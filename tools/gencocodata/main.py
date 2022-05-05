#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
# TODO: 使用 plain_train_net.py 作为训练脚本以获得更多拓展功能

import logging
import os
from collections import OrderedDict
import torch
import platform

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # print('evaluator_type', evaluator_type)
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


# TODO: 研究内建数据集或者bitmask格式数据集输入格式
def register_dataset():
    dataset_dir = os.path.join(root_dir, '../../dataset', dataset, process)
    train_label_path = os.path.join(dataset_dir, train_on, 'label.json')
    train_image_dir = os.path.join(dataset_dir, train_on, 'image')
    test_label_path = os.path.join(dataset_dir, test_on, 'label.json')
    test_image_dir = os.path.join(dataset_dir, test_on, 'image')

    register_coco_instances('dataset_train', {}, train_label_path, train_image_dir)
    register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)

    # dataset_dir = os.path.join(root_dir, '../../dataset', dataset)
    # traintest_label_path = os.path.join(dataset_dir, 'traintest', 'label.json')
    # traintest_image_dir = os.path.join(dataset_dir, 'traintest', 'image')
    # train_label_path = os.path.join(dataset_dir, 'train', 'label.json')
    # train_image_dir = os.path.join(dataset_dir, 'train', 'image')
    # test_label_path = os.path.join(dataset_dir, 'test', 'label.json')
    # test_image_dir = os.path.join(dataset_dir, 'test', 'image')
    #
    # if dataset == "fruits_nuts":
    #     register_coco_instances('dataset_train', {}, traintest_label_path, traintest_image_dir)
    #     register_coco_instances('dataset_test', {}, traintest_label_path, traintest_image_dir)
    # if dataset == "face":
    #     register_coco_instances('dataset_train', {}, traintest_label_path, traintest_image_dir)
    #     register_coco_instances('dataset_test', {}, traintest_label_path, traintest_image_dir)
    # if dataset == "skin":
    #     register_coco_instances('dataset_train', {}, test_label_path, test_image_dir)
    #     register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)
    # if dataset == "test":
    #     register_coco_instances('dataset_train', {}, train_label_path, train_image_dir)
    #     register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)
    # if dataset == "test_2":
    #     register_coco_instances('dataset_train', {}, test_label_path, test_image_dir)
    #     register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)
    # if dataset == "skin_new":
    #     register_coco_instances('dataset_train', {}, train_label_path, train_image_dir)
    #     register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)

    DatasetCatalog.get("dataset_train")
    MetadataCatalog.get("dataset_train")
    DatasetCatalog.get("dataset_test")
    MetadataCatalog.get("dataset_test")


# TODO: 重新规划各级参数配置方式和优先级等等
def setup(args):
    """
    Create configs and perform basic setups.
    """
    # 所有任务共享的基本参数配置 detectron2/config/defaults.py
    cfg = get_cfg()
    # 从yaml文件中读取基于某类_BASE_结构的配置并覆盖，然后从yaml文件中读取基于某个模型的配置并覆盖
    cfg.merge_from_file(args.config_file)
    # 临时的参数设置
    cfg.DATASETS.TRAIN = ('dataset_train',)
    cfg.DATASETS.TEST = ('dataset_test',)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 5

    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    SIZE = 1024
    cfg.INPUT.MAX_SIZE_TRAIN = SIZE
    cfg.INPUT.MIN_SIZE_TRAIN = SIZE
    cfg.INPUT.MAX_SIZE_TEST = SIZE
    cfg.INPUT.MIN_SIZE_TEST = SIZE
    # cfg.INPUT.CROP.ENABLED = False
    # cfg.INPUT.CROP.TYPE = "absolute"
    # cfg.INPUT.CROP.SIZE = [SIZE, SIZE]

    # cfg.TEST.AUG.ENABLED = False
    # cfg.TEST.AUG.MIN_SIZES = (SIZE,)
    # cfg.TEST.AUG.MAX_SIZE = SIZE
    # cfg.TEST.AUG.FLIP = True

    # cfg.MODEL.RESNETS.DEPTH = 50
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # 周期性操作
    cfg.SOLVER.CHECKPOINT_PERIOD = 250
    cfg.TEST.EVAL_PERIOD = 250
    cfg.VIS_PERIOD = 0

    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = (5000)  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    # 从命令行中解析额外的参数并覆盖
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# TODO: main、训练和测试方式研究
def main(args):
    register_dataset()

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


# TODO: config_file和opts规范化
if __name__ == "__main__":
    root_dir = os.path.abspath('..')
    dataset = 'skin'  # fruits_nuts, face, skin, test, skin_01
    process = 'crop_1024'  # 'original', 'crop_1024', 'crop_512', 'crop_2048'
    train_on = 'train'  # 'traintest', 'train', 'test'
    test_on = 'test'  # 'traintest', 'train', 'test'

    args = default_argument_parser().parse_args()
    # linux平台cmd运行时的命令
    # python main.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    # windows平台ide运行时的参数配置
    if platform.system() == 'Windows':
        args.config_file = '../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        args.eval_only = False  # 仅测试或者训练后测试
        args.resume = False  # 从 cfg.OUTPUT_DIR 指定的路径中加载上一次已训练完成的 checkpoint 和 checkpointabels 或者从 cfg.MODEL.WEIGHTS 指定的路径中加载 checkpoint
        args.opts = ['OUTPUT_DIR', './output', 'MODEL.WEIGHTS', 'E:/project/detectron2-master/0_note/result/model_final_f10217.pkl']
        # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"  # initialize from model zoo
        # cfg.MODEL.WEIGHTS = None './output/model_final.pth'  # No checkpoint found. Initializing model from scratch
        # E:/project/detectron2-master/0_note/result/model_final_f10217.pkl
        # '../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
