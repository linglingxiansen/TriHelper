import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms
import time
from scipy.ndimage import center_of_mass
from scipy import ndimage
# from ultralytics import YOLO
from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.habitat.objectgoal_env21 import ObjectGoal_Env21
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
#from qwen import simple_multimodal_conversation_call
from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
torch.manual_seed(1)
def read_episode_numbers_from_file(file_path):
    """从给定文件中读取 episode 编号，并将它们作为整数列表返回。"""
    episode_numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            episode_numbers.append(int(line.strip()))
    return episode_numbers
class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0

    def reset(self):
        self.total_id += 1
        self.epi_id = 0

    def get_action(self):
        self.epi_id += 1
        if self.epi_id == 1:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        else:
            if self.total_id % 2 == 0:
                return 3
            else:
                return 2

class Sem_Exp_Env_Agent(ObjectGoal_Env21):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.device = args.device
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.red_sem_pred = load_rednet(
            self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        )
        self.red_sem_pred.eval()
        # self.red_sem_pred.to(self.device)
        self.flag = 0
        hulue_done = 0
        self.zhaodaoguole = 0
        self.untrap = UnTrapHelper()
        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.hulue = 0
        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.vlm_count = 0
        self.replan_count = 0
        self.collision_n = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

        self.fail_case = {}
        self.fail_case['collision'] = 0
        self.fail_case['success'] = 0
        self.fail_case['detection'] = 0
        self.fail_case['exploration'] = 0
        self.fail_case['hulue'] = 0

        # self.forward_after_stop_preset = self.args.move_forward_after_stop
        # self.forward_after_stop = self.forward_after_stop_preset

        # self.eve_angle = 0

        self.rgb = None
        self.eve_angle = 0
        self.zhuanquan = 0
        self.call = 0
        self.yidong = '一段距离'
        self.depth_visual = None
        self.jiamubiao=0
        self.dist = 0
        self.loc_r = 240
        self.loc_c = 240    
        self.sem_map_vis = 0
        self.tokenizer = AutoTokenizer.from_pretrained("/hpc2hdd/home/hwang464/Qwen-VL/models--Qwen--Qwen-VL-Chat-Int4", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("/hpc2hdd/home/hwang464/Qwen-VL/models--Qwen--Qwen-VL-Chat-Int4", device_map="cuda", trust_remote_code=True).eval()
        self.history = None
        self.yougoal = 0
        self.xiumiancount=0
        self.pregoal = None
        self.goal = None
    def reset(self):
        self.history = None
        self.jiamubiao=0
        args = self.args
        self.hulue = 0
        # self.forward_after_stop = self.forward_after_stop_preset
        self.replan_count = 0
        hulue_done = 0
        self.yougoal = 0
        self.pregoal = None


        self.xiumiancount=0
        self.dist = 0
        self.loc_r = 240
        self.loc_c = 240    
        self.sem_map_vis = 0
        self.zhaodaoguole = 0
        self.collision_n = 0
        self.prev_blocked = 0
        self._previous_action = -1
        obs, info = super().reset()
        obs = self._preprocess_obs(obs)
        self.flag=0
        self.obs_shape = obs.shape
        self.vlm_count = 0
        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None

        self.eve_angle = 0
        self.eve_angle_old = 0
        info['set_hard_goal'] = 0
        info['set_llm_xiumian']=0
        info['eve_angle'] = self.eve_angle
        self.block_threshold = 6

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        self.goal = None
        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # s_time = time.time()

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), self.fail_case, False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.yougoal = 1
            goal = planner_inputs['goal']
            self.goal = goal
            if np.sum(goal == 1) == 1 and self.args.task_config == "tasks/objectnav_gibson.yaml":
                frontier_loc = np.where(goal == 1)
                self.info["g_reward"] = self.get_llm_distance(planner_inputs["map_target"], frontier_loc)

            # self.collision_map = np.zeros(self.visited.shape)
            self.info['clear_flag'] = 0

        action = self._plan(planner_inputs)

        # c_time = time.time()
        # ss_time = c_time - s_time
        # print('plan map: %.3f秒'%ss_time) 0.19

        if self.collision_n > 20 or self.replan_count > 50:
            self.info['clear_flag'] = 1
            self.collision_n = 0

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        
        if action >= 0:

            # act
            action = {'action': action}
            obs, rew, done, info = super().step(action)
            
            
            if (done and self.info['success'] == 0) :
                if self.info['time'] >= self.args.max_episode_length - 1:
                    self.fail_case['exploration'] += 1
                
                #elif self.hulue  == 1:
                   # self.fail_case['hulue'] += 1
                    #print(self.info['success'])
                elif self.replan_count > 50:
                    self.fail_case['collision'] += 1
                
                else:
                    self.fail_case['detection'] += 1

            if done and self.info['success'] == 1:
                self.fail_case['success'] += 1

            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            self.obs = obs
            self.info = info
            info['eve_angle'] = self.eve_angle
            info['time'] = self.timestep
            info['zhaodaoguole'] = self.zhaodaoguole
            info['jiamubiao'] = self.jiamubiao
            # e_time = time.time()

            # if self.replan_count>20:
            #     info['set_hard_goal']=1
            

            # x1, y1, t1 = self.last_loc
            
            x2 = self.loc_r 
            y2 = self.loc_c 
            if self.yougoal==1 and info['set_hard_goal']==0:
                arr = self.goal>0.5
                #print(arr)
                nonzero_indices = np.nonzero(arr) 

                x = nonzero_indices[0] # 第一维的索引放入x
                y = nonzero_indices[1]

                x1 = x[0]
                y1 = y[0]

                # if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                #     self.col_width += 2
                #     if self.col_width == 7:
                #         self.col_width = min(self.col_width, 5)
                # else:
                #     self.col_width = 1

                dist = pu.get_l2_distance(x1, x2, y1, y2)
                #print(x1,y1,x2,y2,dist)
                self.dist = dist
                #print(goal,goal.shape)
                if dist < 50 and self.replan_count==0 and (planner_inputs["new_goal"]==1 and (self.pregoal==goal).all()):
                   
                    
                    info['set_llm_xiumian'] = 1
                self.yougoal=0
                
                    
            

            if info['set_llm_xiumian']==1:
                
                self.xiumiancount+=1
                if self.xiumiancount==20:
                    info['set_llm_xiumian']=0
                    self.xiumiancount=0

            if info['set_hard_goal']==1:
                # print("time",self.timestep)
                self.xiumiancount+=1
                if self.xiumiancount==30:    
                    info['set_hard_goal']=0
                    self.xiumiancount=0
            # e_time = time.time()
            # ss_time = e_time - c_time
            # print('act map: %.3f秒'%ss_time) 0.23
            mask = np.all(self.sem_map_vis == (242,242,242), axis=-1)

            

            labels, num = ndimage.label(mask)

            area_sizes = np.bincount(labels.ravel())[1:]
            if area_sizes.size > 0:
                max_area_label = np.argmax(area_sizes) + 1 # 标签从1开始
                max_area_size = area_sizes[max_area_label - 1]
            else:
                max_area_label = None
                max_area_size = 0

            if max_area_label is not None:
            # 将最大连接区域变成绿色（0,255,0）
                max_area_pixels = labels == max_area_label
            # sem_map_vis[max_area_pixels] = [0, 255, 0]

            

            # 计算最大连通区域的质心
            centroid = center_of_mass(mask, labels=labels, index=max_area_label)

            true_indices = np.argwhere(self.m1)

            # 计算所有 True 索引的平均值来找到质心
            # np.mean 计算给定轴上的元素平均值，axis=0 表示对列（即每个维度的索引）进行平均
            centroid = np.mean(true_indices, axis=0)
            # 质心返回的是浮点数，我们需要将其四舍五入为最近的整数像素坐标
            centroid_rounded = tuple(np.round(centroid).astype(int))
            # print(centroid_rounded)
            if (0 < centroid_rounded[0] < 480) and (0 < centroid_rounded[1] < 480) and (self.m1[centroid_rounded]) :
                # 质心在区域内
                info['center_row'] = centroid_rounded[0]
                info['center_col'] = centroid_rounded[1]
                # print(centroid_rounded)
            else:
                # 质心不在区域内，使用几何中心
                y, x = np.where(self.m1)
                if len(x) == 0 or len(y) == 0:
                    info['center_row'] = 240
                    info['center_col'] = 240 # 连通区域为空或异常
                else:
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    
                    info['center_row'] = center_x
                    info['center_col'] = center_y
            if planner_inputs["new_goal"]:
                self.pregoal = planner_inputs['goal']
            
            self.sethardgoal=info['set_hard_goal']
            # ss_time = e_time - c_time
            # print('act map: %.3f秒'%ss_time) 0.23

            # info['g_reward'] += rew

            return obs, self.fail_case, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), self.fail_case, False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                self.prev_blocked += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
            else:
                if self.prev_blocked >= self.block_threshold:
                    self.untrap.reset()
                self.prev_blocked = 0    

                
        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        if replan:
            self.replan_count += 1
            print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        
        preeve = self.eve_angle
        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) or self.replan_count > 50:
            action = 0  # Stop
            
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x > map_pred.shape[0]: eve_start_x = map_pred.shape[0] 
            if eve_start_y > map_pred.shape[0]: eve_start_y = map_pred.shape[0] 
            if eve_start_x < 0: eve_start_x = 0 
            if eve_start_y < 0: eve_start_y = 0 
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
        if self.prev_blocked >= self.block_threshold and preeve==0:
            self.eve_angle=0
            if self._previous_action == 1:
                action = self.untrap.get_action()
            else:
                action = 1
        self._previous_action = action
        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        
        # print("obs: ", obs)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        rgb_vis = rgb[:, :, ::-1]

        depth = obs[:, :, 3:4]
        semantic = obs[:,:,4:5].squeeze()
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1

            sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
            sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
            sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
            sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
            sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1

        # vlm
        #print("goal:", self.info['goal_cat_id'],self.info['goal_name'])
        goal_cat_id = self.info['goal_cat_id']
        goal_name = self.info['goal_name']
        #print(goal_cat_id,goal_name)
        no_vlm=1
        # self.jiamubiao=0
        if self.timestep<300 and no_vlm and np.any(sem_seg_pred[:,:,goal_cat_id]==1) and self.flag == 0 :
            #self.count += 1

            if goal_name=='tv_monitor':
                ask = '电视'

            elif goal_name=='chair':
                ask = '椅子'

            elif goal_name=='sofa':
                ask = '沙发'

            elif goal_name=='bed':
                ask='床'

            elif goal_name=='plant':
                ask='植物或花瓶'
            else:
                ask = '马桶'
            
            
            cv2.imwrite(f'path/ask{goal_name}-{self.episode_no}-{self.timestep}-{self.rank}.png', rgb_vis)
            query = self.tokenizer.from_list_format([
     # Either a local path or an url
    {'image': f'path/ask{goal_name}-{self.episode_no}-{self.timestep}-{self.rank}.png'},
    {'text': f'这张图中有{ask}吗'},
])
            response, self.history = self.model.chat(self.tokenizer, query=query, history=self.history)
            print(response)
            self.history = None
            if response is not None:
                
                # print(text_result)
                if '0' in response or '没' in response or '不' in response:
                    # sem_seg_pred[:,:,16] =  sem_seg_pred[:,:,goal_cat_id]
                    sem_seg_pred[:,:,goal_cat_id] = 0
                    self.jiamubiao = 1
                    self.zhaodaoguole = 1
                    # self.count = 0
                
                elif '1' in response or '存在' in response or '有' in response:
                    self.flag = 1
                    self.jiamubiao = 0
                    # self.count = 0
                else:
                    self.flag = 0
            # rgb = obs[:, :, :3]
            rgb_vis = rgb[:, :, ::-1]
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
            
            
            
            
            # exit()



        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth, use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy()
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        # print(semantic_pred,red_semantic_pred)
        # print(red_semantic_pred.shape,red_semantic_pred.max(),red_semantic_pred.min())
        # print(semantic_pred.shape,semantic_pred.max(),semantic_pred.min())

        normalized_img = ((red_semantic_pred - red_semantic_pred.min()) / red_semantic_pred.max() * 255).astype(np.uint8)  # Normalizing to 0-255

        # Apply a colormap for visualization
        colored_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)
        fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-red_semantic_test.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
        # cv2.imwrite(fn, colored_img)

        # normalized_img2 = ((semantic_pred - semantic_pred.min()) / semantic_pred.max() * 255).astype(np.uint8)  # Normalizing to 0-255

        # Apply a colormap for visualization
        # colored_img2 = cv2.applyColorMap(normalized_img2, cv2.COLORMAP_JET)
        # fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-semantic_test.png'.format(
        #         dump_dir, self.rank, self.episode_no,
        #         self.rank, self.episode_no, self.timestep)
        # cv2.imwrite(fn, colored_img2)

        for class_idx in range(semantic_pred.shape[2]):
                    # Create a binary mask where the class label is 1
                    if np.any(semantic_pred[:, :, class_idx] == 1):
                        class_mask = semantic_pred[:, :, class_idx] == 1

                        # Convert the mask to uint8 format
                        class_mask_uint8 = (class_mask * 255).astype(np.uint8)

                        # Save the mask as an image
                        fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-semantic_test-class_{}.png'.format(
                        dump_dir, self.rank, self.episode_no,
                        self.rank, self.episode_no, self.timestep,class_idx)
                        # cv2.imwrite(fn, class_mask_uint8)









        return red_semantic_pred, semantic_pred

# Assuming semantic_pred is your (480, 640, 16) array
        

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        local_w = inputs['map_pred'].shape[0]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        map_edge = inputs['map_edge']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        edge_mask = map_edge == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 9
        # sem_map[m1] = 9
        self.m1 = m1
        true_indices = np.argwhere(self.m1)

        # 计算所有 True 索引的平均值来找到质心
        # np.mean 计算给定轴上的元素平均值，axis=0 表示对列（即每个维度的索引）进行平均
        centroid = np.mean(true_indices, axis=0)
        # 质心返回的是浮点数，我们需要将其四舍五入为最近的整数像素坐标
        centroid_rounded = tuple(np.round(centroid).astype(int))
        sem_map[centroid_rounded[0]:centroid_rounded[0]+5,centroid_rounded[1]:centroid_rounded[1]+5]=3


        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        if np.sum(goal) == 1:
            f_pos = np.argwhere(goal == 1)
            # fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]))
            # goal_fmb = skimage.draw.circle_perimeter(int((fmb[0]+fmb[1])/2), int((fmb[2]+fmb[3])/2), 23)
            goal_fmb = skimage.draw.circle_perimeter(int(f_pos[0][0]), int(f_pos[0][1]), int(local_w/4-2))
            goal_fmb[0][goal_fmb[0] > local_w-1] = local_w-1
            goal_fmb[1][goal_fmb[1] > local_w-1] = local_w-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)


        self.sem_map_vis = sem_map_vis
        mask = np.all(self.sem_map_vis == (242,242,242), axis=-1)


        labels, num = ndimage.label(mask)

        area_sizes = np.bincount(labels.ravel())[1:]
        if area_sizes.size > 0:
            max_area_label = np.argmax(area_sizes) + 1 # 标签从1开始
            max_area_size = area_sizes[max_area_label - 1]
        else:
            max_area_label = None
            max_area_size = 0

        if max_area_label is not None:
                    # 将最大连接区域变成绿色（0,255,0）
            max_area_pixels = labels == max_area_label


        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images :
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)


