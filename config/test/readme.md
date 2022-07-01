Some important parameters:  
[data]
- query_image_list: txt file list of the query images
- query_label_list: txt file list of the query labels
- support_image_list: txt file list of the support images
- support_label_list: txt file list of the support labels
- label_wanted_ls: the list of wanted labels. For example, the value of pancreas in TCIA-Pancreas labels is 1 thus label_wanted_ls = [1]
- multi_run_ensemble_num: locate each target landmark several times with different random initializations
- fine_detection: apply fine detection or not

[coarse/fine_pnetwork]
- distance_ratio: the upper/lower bound of predicted distances. It should be kept the same with the training time.

[testing]
- coarse_pnet_load_path: the path of coarse model
- fine_pnet_load_path: the path of fine model. If there is no fine model, you could just copy the coarse model.
