# TriHelper
Repository for TriHelper: Zero-Shot Object Navigation with Dynamic Assistance (IROS2024)

## Installation
The code has been tested only with Python 3.7 on Ubuntu 20.04.

1. Environments Setup
- Follow [L3MVN](https://raw.githubusercontent.com/ybgdgh/L3MVN/) to install Habitat-lab, Habitat-sim, detectron2, torch and other independences.

2. Dataset
- Download [HM3D](https://aihabitat.org/datasets/hm3d/) to the data path.
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
TriHelper/
  data/
    scene_datasets/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectgoal_hm3d/
        train/
        val/
        val_mini/
```

3. VLM
- Download the [Qwen2-VL-Chat-Int4](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4) and change the model path in sem_exp.py.

## Evaluation
```
python main_llm_zeroshot.py --split val --eval 1 --auto_gpu_config 0 \
-n 1 --num_eval_episodes 2000 --load pretrained_models/llm_model.pt \
--use_gtsem 0 --num_local_steps 10
```

## Citation
If you find this project useful, welcome to cite us.
```bib
@inproceedings{zhang2024trihelper,
  title={Trihelper: Zero-shot object navigation with dynamic assistance},
  author={Zhang, Lingfeng and Zhang, Qiang and Wang, Hao and Xiao, Erjia and Jiang, Zixuan and Chen, Honglei and Xu, Renjing},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10035--10042},
  year={2024},
  organization={IEEE}
}
```


