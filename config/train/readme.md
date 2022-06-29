Some import parameters:  
[data]
- `iter_num`: iteration number for each epoch
- `batch_size`: batch size
- `patch_size`: the cropping patch size along axial, coronal, sagittal

[pnetwork]
- `net_type`: the choosen network, e.g., Pnet, Pnet_2. You could create your own networks in `networks/` and add them in `networks/NetFactory.py`
- `distance_ratio`: the upper and lower bound of predicted relative distance, e.g., 300 means the predicted d_qs ranges in [-300, 300].

[training]
- `load_weight`: Load pretrained model or not
- `pnet_load_path`: The path to pretrained model
- `device_ids`: The GPUs id, e.g., `0,` represents using GPU 0 for training and `0,1` represents using GPU 0 and 1 for training
- `small_move`: In training stage, restricting the distance between two random cropped images or not
- `fluct_range`: If `small_move=True`, `fluct_range` decides the max voxel ranges between two random cropped images
- `num_worker`: num_workers for Dataloader
- `load_memory`: For fast training, this work stores data in memory by default. Switch it to `False` if you have limited memory space
