# RPR-Loc
Official code for MICCAI2021. ['Contrastive Learning of Relative Position Regression for One-Shot Object Localization in 3D Medical Images'](https://arxiv.org/abs/2012.07043)
![image](https://github.com/LWHYC/RPR-Loc/blob/main/Framework.png)
## Dataset
You could download the processed dataset from: [StructSeg](https://structseg2019.grand-challenge.org/Home/) task1 (Organ-at-risk segmentation from head & neck CT scans): [BaiDu Yun](https://pan.baidu.com/s/1VV8VqJ39wKvlF-mh8b6IVg?pwd=ic6g) or [Google Drive](https://drive.google.com/file/d/1TlMfWvgSd3kAh3Eq80DVoboZ42FbLMvE/view?usp=sharing) and [TCIA-Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT): [BaiDu Yun](https://pan.baidu.com/s/13dTwbEzn4OtbgxwqlPD1AA?pwd=chnt) or [Google Drive](https://drive.google.com/file/d/1-rPJxWl0nwxPqAHv5s4oj4gRaAGlOXgU/view?usp=sharing) into `data/` and unzip them. For TCIA-Pancreas, please cite the original paper (Deeporgan: Multi-level deep convolutional networks for automated pancreas segmentation).
## Training
The training config is in `config/train/`, containing 4 files for coarse/fine & pancreas/head and neck dataset. You could change the parameters for your own experiments. More details could be found in `config/train/readme.md.

For example, you could `cd train` then run `python train_position.py ../config/train/train_position_pancreas_coarse.txt` to train a coarse RPR model for TCIA-Pancreas dataset.
## Citation
If you use this code in your research, please cite our work:

    @inproceedings{lei2021contrastive,
    title={Contrastive learning of relative position regression for one-shot object localization in 3D medical images},
    author={Lei, Wenhui and Xu, Wei and Gu, Ran and Fu, Hao and Zhang, Shaoting and Zhang, Shichuan and Wang, Guotai},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={155--165},
    year={2021},
    organization={Springer}
    }
