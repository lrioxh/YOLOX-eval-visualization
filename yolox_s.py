#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from datetime import datetime
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = f"{os.path.split(os.path.realpath(__file__))[1].split('.')[0]}_"+\
            str(datetime.today()).replace(" ","-").split(".")[0].replace(":","-")

        # Define yourself dataset path
        self.data_dir = "datasets/coco"
        self.test_ann = "instances_val2017.json"
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 80

        self.warmup_epochs = 1
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 16

        self.max_epoch = 5
        self.data_num_workers = 4
        self.multiscale_range = 5
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 2
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 2
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False

        # -----------------  testing config ------------------ #
        # nms threshold
        self.nmsthre = 0.5
        self.test_conf = 0.01

        # draw curve
        self.plot_sample_rate = 3000    # 默认1000



if __name__=="__main__":
    print(f"{os.path.split(os.path.realpath(__file__))[1].split('.')[0]}_"+\
            str(datetime.today()).replace(" ","-").split(".")[0].replace(":","-"))
