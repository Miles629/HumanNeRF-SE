from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']
    # print("1DatasetArgs init!",cfg.category,cfg.task)
    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        # print("1DatasetArgs zju_mocap!")
        for sub in subjects:
            '''!!! use the comments below to revert to the default code if needed !!!'''
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    # "dataset_path": f"dataset/zju_mocap/{sub}_split_train",
                    # "dataset_path": f"dataset/zju_mocap/{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_mono_train",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
#                     "dataset_path": f"dataset/zju_mocap/{sub}_split_eval",
                    # "dataset_path": f"dataset/zju_mocap/{sub}_eval",
                    "dataset_path": f"dataset/zju_mocap/377_mono_train",
#                     "dataset_path": f"dataset/zju_mocap/{sub}_novelpose",
                    
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })
            # dataset_attrs.update({
            #     f"zju_{sub}_train": {
            #         "dataset_path": f"dataset/zju_mocap/processed/{sub}",
            #         "keyfilter": cfg.train_keyfilter,
            #         "ray_shoot_mode": cfg.train.ray_shoot_mode,
            #     },
            #     f"zju_{sub}_test": {
            #         "dataset_path": f"dataset/zju_mocap/processed/{sub}_eval",
            #         "keyfilter": cfg.test_keyfilter,
            #         "ray_shoot_mode": 'image',
            #         "src_type": 'zju_mocap'
            #     },
            # })


    if cfg.category == 'human_nerf' and cfg.task == 'people-snapshot':
        subs = ['f3c','f1c','f3s','f4c','f4s','f6p','f7p','f8p','m1c','m1p','m1s','m2c','m2o','m2p','m2s','m3c','m3o','m3p','m3s','m4c','m4s','m5o','m5s','m9p']
        # print("1DatasetArgs, peoplesnapshot")
        for sub in subs:
            dataset_attrs.update({
                f"{sub}_train": {
                    "dataset_path": f"dataset/people-snapshot/{sub}/",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"{sub}_test": {
                    "dataset_path": f"dataset/people-snapshot/{sub}_eval",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'people-snapshot'
                },
            })


    @staticmethod
    def get(name):
        print("1DatasetArgs,name",name)
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()