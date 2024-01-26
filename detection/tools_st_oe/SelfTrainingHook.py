import datetime
import itertools
import logging
import math
import operator
import os
import tempfile
import time
import warnings
from collections import Counter
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager

# inference on datasets
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union

from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from detectron2.engine.train_loop import HookBase
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou
import json

from tools_st_oe.build_pseudo import build_detection_train_loader_pseudo
from core.datasets.setup_datasets import setup_pseudo_label_dataset
import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.data.datasets.pascal_voc import register_pascal_voc

# Project imports
import core.datasets.metadata as metadata

from detectron2.data.catalog import DatasetCatalog
from core.datasets.pascal_voc_oe import register_pascal_voc_oe


__all__ = [
    "SelfTrainingHook",
]

class SelfTrainingHook(HookBase):
    """
    Self-Training hook
    """
    def __init__(self,cfg, pseudo_label_function):
        self.cfg = cfg
        self.func = pseudo_label_function

        # self.num_rounds = cfg.SELF_TRAINING
        self.iterations_per_round = cfg.SELF_TRAINING.ITERATIONS_PER_ROUND
        self.init_oe_portion = cfg.SELF_TRAINING.INIT_OE_PORTION
        self.min_oe_portion = cfg.SELF_TRAINING.MIN_OE_PORTION
        self.oe_portion_step = cfg.SELF_TRAINING.OE_PORTION_STEP

        # set milestone
        self.start_iteration = cfg.AUXI_DATASETS.START_FINETUNE_ITER
        self.max_iteration = cfg.SOLVER.MAX_ITER
        self.mile_stone = [i for i in range(self.start_iteration,self.max_iteration,self.iterations_per_round)]
        self.oe_portion = self.init_oe_portion
        self.start_self_spaced_iteration = cfg.SELF_TRAINING.START_SELF_SPACED
        # json file
        self.raw_results_path = os.path.join(cfg.OUTPUT_DIR,cfg.SELF_TRAINING.RAW_RESULTS_PATH)
        self.filtered_results_path = os.path.join(cfg.OUTPUT_DIR,cfg.SELF_TRAINING.FILTERED_RESULTS_PATH)
        self.pseudo_label_path = os.path.join(cfg.OUTPUT_DIR,cfg.SELF_TRAINING.PSEUDO_LABEL_SAVED_PATH)

        # sample
        self.current_round = 0
        if comm.is_main_process():
            if not os.path.exists(self.raw_results_path):
                os.makedirs(self.raw_results_path)
            if not os.path.exists(self.filtered_results_path):
                os.makedirs(self.filtered_results_path)
            if not os.path.exists(self.pseudo_label_path):
                os.makedirs(self.pseudo_label_path)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):

        if self.trainer.storage.iter in self.mile_stone:
            #### pseudo-label generation
            # evaluation & save energy score
            predictions = self.func(self.current_round)
            if comm.is_main_process():
                self.collect_raw_predictions_and_save(predictions)
                self.select_pseudo_label_and_save()
            comm.synchronize()
            self.register_pseudo_label_datasets()
            data_loader_pseudo = build_detection_train_loader_pseudo(self.cfg, 'pseudo_label_format_{}'.format(
            self.trainer.storage.iter))

            self.trainer._trainer.update_dataloader(data_loader_pseudo)
            self.current_round += 1


    def after_step(self):
        pass


    def collect_raw_predictions_and_save(self, predictions):
        final_output_list = []
        final_max_box_result_list = []
        for pred in predictions:
            instances = pred['output'][0]['instances']
            images_info = pred['images_info'][0]
            if len(instances) !=0:
                result, max_box_result = self.instances_to_json(instances,images_info)
                final_output_list.extend(result)
                final_max_box_result_list.extend(max_box_result)

        with open(os.path.join(self.raw_results_path, 'raw_results_{}_iter.json'.format(self.trainer.storage.iter)), 'w') as fp:
            json.dump(final_output_list, fp, indent=4, separators=(',', ': '))
        with open(os.path.join(self.raw_results_path, 'max_box_raw_results_{}_iter.json'.format(self.trainer.storage.iter)), 'w') as fp:
            json.dump(final_max_box_result_list, fp, indent=4, separators=(',', ': '))

    def select_pseudo_label_and_save(self,):
        with open(os.path.join(self.raw_results_path, 'max_box_raw_results_{}_iter.json'.format(self.trainer.storage.iter)), 'r') as f:
            max_box_json_dicts = json.load(f)
        f.close()

        energyscore_list = [item["energy_score"] for item in max_box_json_dicts if item['max_box_id']!=-1]
        if self.trainer.storage.iter >= self.start_self_spaced_iteration:
            self.oe_portion = max(self.init_oe_portion - (int(
                (self.trainer.storage.iter - self.start_self_spaced_iteration) / self.iterations_per_round) + 1)* self.oe_portion_step,
                              self.min_oe_portion)
        else: self.oe_portion = self.init_oe_portion
        filter_threshold = self.k_parameter(energyscore_list, self.oe_portion)
        self.oe_selection(filter_threshold)

    def register_pseudo_label_datasets(self,):

        max_box_file = open(os.path.join(self.filtered_results_path,'max_box_filtered_results_'+str(self.trainer.storage.iter))+'.json', 'r')
        content = max_box_file.read()
        max_box_json_dicts = json.loads(content)
        max_box_file.close()
        annotations_list = []
        images_list = []
        count = 1
        for ann in max_box_json_dicts:
            annotations_list.append({'image_id': ann['image_id'],
                                     'id': count,
                                     'category_id': ann['category_id'],
                                     'bbox': ann['bbox'],
                                     'area': ann['area'],
                                     'iscrowd': 0,
                                     })
            count+=1
            images_list.append({'id': ann['image_id'],
                                'width': ann['width'],
                                'height': ann['height'],
                                'file_name': ann['file_name'],
                                'license': 1})
        licenses = [{'id': 1,
                     'name': 'none',
                     'url': 'none'}]
        if 'voc' in self.cfg.DATASETS.TRAIN[0]:
            categories = [{'id': 1, 'name': "person", 'supercategory': 'entity'},
                          {'id': 2, 'name': "bird", 'supercategory': 'entity'},
                          {'id': 3, 'name': "cat", 'supercategory': 'entity'},
                          {'id': 4, 'name': "cow", 'supercategory': 'entity'},
                          {'id': 5, 'name': "dog", 'supercategory': 'entity'},
                          {'id': 6, 'name': "horse", 'supercategory': 'entity'},
                          {'id': 7, 'name': "sheep", 'supercategory': 'entity'},
                          {'id': 8, 'name': "airplane", 'supercategory': 'entity'},
                          {'id': 9, 'name': "bicycle", 'supercategory': 'entity'},
                          {'id': 10, 'name': "boat", 'supercategory': 'entity'},
                          {'id': 11, 'name': "bus", 'supercategory': 'entity'},
                          {'id': 12, 'name': "car", 'supercategory': 'entity'},
                          {'id': 13, 'name': "motorcycle", 'supercategory': 'entity'},
                          {'id': 14, 'name': "train", 'supercategory': 'entity'},
                          {'id': 15, 'name': "bottle", 'supercategory': 'entity'},
                          {'id': 16, 'name': "chair", 'supercategory': 'entity'},
                          {'id': 17, 'name': "dining table", 'supercategory': 'entity'},
                          {'id': 18, 'name': "potted plant", 'supercategory': 'entity'},
                          {'id': 19, 'name': "couch", 'supercategory': 'entity'},
                          {'id': 20, 'name': "tv", 'supercategory': 'entity'},
                          {'id': 21, 'name': "background",'supercategory': 'entity'},
                          {'id': 22, 'name': "OOD",'supercategory':'entity'}
                          ]

        json_dict_val = {'info': {'year': 2023},
                         'licenses': licenses,
                         'categories': categories,
                         'images': images_list,
                         'annotations': annotations_list}

        val_file_name = os.path.join(self.pseudo_label_path, 'pseudo_label_format_{}.json'.format(self.trainer.storage.iter))
        with open(val_file_name, 'w') as outfile:
            json.dump(json_dict_val, outfile, indent=4, separators=(',', ': '))
        outfile.close()
        setup_pseudo_label_dataset('pseudo_label_format_{}'.format(self.trainer.storage.iter), os.path.join(self.cfg.OE_DATASET_PATH, "JPEGImages"), val_file_name)

    def k_parameter(self, energyscore_list, oe_portion):

        energyscore_list.sort()
        len_energyscore = len(energyscore_list)
        select_size = int(math.floor(len_energyscore * oe_portion))
        threshold = energyscore_list[select_size-1]
        return threshold

    def oe_selection(self,filter_threshold):
        max_box_file = open(
            os.path.join(self.raw_results_path, 'max_box_raw_results_{}_iter.json'.format(self.trainer.storage.iter)),
            'r')
        content = max_box_file.read()
        max_box_json_dicts = json.loads(content)
        max_box_file.close()
        filtered_max_box_json_dicts = []
        for item in max_box_json_dicts:
            if item['energy_score'] < filter_threshold:
                # max_box_json_dicts.remove(item)
                filtered_max_box_json_dicts.append(item)
        with open(os.path.join(self.filtered_results_path,'max_box_filtered_results_'+str(self.trainer.storage.iter)+'.json'), 'w') as fp:
            json.dump(filtered_max_box_json_dicts, fp, indent=4, separators=(',', ': '))
        fp.close()

    def instances_to_json(self, instances, images_info, cat_mapping_dict=None):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances): detectron2 instances
            img_id (int): the image id
            cat_mapping_dict (dict): dictionary to map between raw category id from net and dataset id. very important if
            performing inference on different dataset than that used for training.

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.cpu().tolist()
        classes = instances.pred_classes.cpu().tolist()
        # inter_feat = instances.inter_feat.cpu().tolist()

        energy_score = instances.energy_scores.cpu().tolist()
        # import ipdb; ipdb.set_trace()
        sizes = instances.box_sizes.cpu().tolist()
        # max_box_id = sizes.argmax().item() if len(sizes)>0 else -1

        max_box_id = sizes.index(max(sizes)) if len(sizes)>0 else -1
        # max_box_id = instances.max_box_id

        # classes = classes
        # breakpoint()
        # pred_cls_probs = instances.pred_cls_probs.cpu().tolist()


        results = []
        max_result = []
        for k in range(num_instance):
            result = {
                "image_id": images_info['image_id'],
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "energy_score": energy_score[k],
                "area": sizes[k]
            }
            results.append(result)
        if max_box_id != -1:
            max_result.append(
                {
                    "file_name": images_info['file_name'],
                    "image_id": images_info['image_id'],
                    "height":images_info['height'],
                    "width":images_info['width'],
                    "max_box_id": max_box_id,
                    "category_id": self.cfg.MODEL.ROI_HEADS.NUM_CLASSES + 2,
                    "bbox": boxes[max_box_id],
                    "score": scores[max_box_id],
                    "energy_score": energy_score[max_box_id],
                    "area": sizes[max_box_id]
                }
            )
        # max_result.append(max_result)
        return results, max_result

