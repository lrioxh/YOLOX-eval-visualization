# encoding: utf-8
import os
from datetime import datetime

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1
        self.max_epoch = 5

        
        self.eval_interval = 2
        self.print_interval = 2

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # draw curve
        self.plot_sample_rate = 3000    # 默认1000
        
        self.exp_name = f"{os.path.split(os.path.realpath(__file__))[1].split('.')[0]}_"+\
            str(datetime.today()).replace(" ","-").split(".")[0].replace(":","-")

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            image_sets=[('2007', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'val')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            exp_name=self.exp_name,
            plot_sample_rate=self.plot_sample_rate if self.plot_sample_rate else 1000

        )
