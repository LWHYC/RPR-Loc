# RPR-Loc
Official code for MICCAI2021. ['Contrastive Learning of Relative Position Regression for One-Shot Object Localization in 3D Medical Images'](https://arxiv.org/abs/2012.07043)
![image](https://github.com/LWHYC/RPR-Loc/blob/main/Framework.png)
## Dataset
You could download the processed StructSeg task1 dataset from [BaiDu Yun](https://pan.baidu.com/s/1VV8VqJ39wKvlF-mh8b6IVg?pwd=ic6g) or Google Drive (wait a minute) in `data/` and unzip it. 
## Training
Run `train/train_position.py`.
The config is in `config`, containing 4 files for coarse/fine & pancreas/head and neck dataset.
