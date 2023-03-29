#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized,ConfusionMatrix, ap_per_class, box_iou

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct

class VOCEvaluator:
    """
    VOC AP Evaluation class.
    """

    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes,exp_name,plot_sample_rate):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)
        self.exp_name=exp_name
        self.plot_sample_rate=plot_sample_rate

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False,
    ):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
        iouv = torch.linspace(0.5, 0.95, 10, device='cpu')
        niou = iouv.numel()
        stats=[]
        seen=0
        cat_ids =self.dataloader.dataset.class_ids
        confusion_matrix = ConfusionMatrix(nc=self.num_classes,cat_ids=cat_ids)
        names_dic={}
        for item in self.dataloader.dataset.cats:
            names_dic[item["id"]]=item["name"]
        names_dic = {cat_ids.index(key): value for key, value in names_dic.items()}
        names = list(names_dic.values())
        s = ('\n%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')    
        save_dir=f"YOLOX_outputs/{self.exp_name}"

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            dt_data=self.convert_to_voc_format(outputs, info_imgs, ids)
            data_list.update(dt_data)
            print(dt_data)

            

            for _id,out in image_wise_data.items():
                seen += 1
                # gtAnn=self.dataloader.dataset.coco.imgToAnns[int(_id)]
                # print(gtAnn)
                gtAnn=self.dataloader.dataset.load_anno_from_ids(int(_id))[0]

                tcls=gtAnn[:,4]
                # print(gtAnn,tcls)
                if out==None: 
                    if len(gtAnn)>0:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                
                if len(gtAnn)>0:
                    # gt=torch.tensor([[(its['category_id'])]+its['clean_bbox'] for its in gtAnn])
                    # dt=out.cpu().numpy()
                    # dt[:,4]=dt[:,4]*dt[:,5]
                    # dt[:,5]=dt[:,6]
                    # dt=torch.from_numpy(np.delete(dt,-1,axis=1))#share mem
                    
                    dt=np.concatenate((out["bboxes"],np.array(out["scores"], ndmin=2).T,np.array(out["categories"], ndmin=2).T),axis=1)
                    dt=torch.from_numpy(dt)
                    gt=np.concatenate((gtAnn[:,4].reshape(-1,1),gtAnn[:,:4]),axis=1)
                    gt=torch.from_numpy(gt)

                    # gt_ids:[0,80],dt在convert_to_coco_format中映射到了[1,90],此处转换回[0,80](相当于多算一步，为了减少对源文件的改动)
                    if max(cat_ids)>self.num_classes:
                        dt[:,5].map_(dt[:,5],lambda d,*y:cat_ids.index(d))

                    # print(_id,dt,gt)
                    confusion_matrix.process_batch(dt, gt)
                    correct = process_batch(dt, gt, iouv)
                stats.append((correct, dt[:, 4], dt[:, 5], tcls))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        tp, fp, p, r, f1, ap, ap_class =ap_per_class(*stats, plot=True, save_dir=save_dir, names=names_dic,sample_rate=self.plot_sample_rate)
        confusion_matrix.plot(save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_classes)
        pf = '\n%20s' + '%11i'  *2 + '%11.3g' * 4  # print format
        s+=pf % ('all',seen, nt.sum(), mp, mr, map50, map)
        for i, c in enumerate(ap_class):
            s+=pf % (names[c],seen, nt[c], p[i], r[i], ap50[i], ap[i])
        logger.info(s)      # log出P，R，mAP50，mAP95

        if return_outputs:
            return eval_results, data_list
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )
        info = time_info + "\n"

        all_boxes = [
            [[] for _ in range(self.num_images)] for _ in range(self.num_classes)
        ]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(f"im_eval: {img_num + 1}/{self.num_images} \r")
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataloader.dataset.evaluate_detections(all_boxes, tempdir)
            return mAP50, mAP70, info
