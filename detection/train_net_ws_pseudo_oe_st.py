"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import sys
import os

import core

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results

# Project imports
from default_trainer_st_oe import DefaultTrainer_OE_ST
from core.setup_st_oe import setup_config, setup_arg_parser

# AUXI dataset
from tools_st_oe.build_pseudo import build_auxi_data_inference_loader


class Trainer(DefaultTrainer_OE_ST):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)
    @classmethod
    def auxi_data_inference_loader(cls, cfg, dataset_name, current_round):
        return build_auxi_data_inference_loader(
            cfg, dataset_name, current_round)


def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed, is_testing=False, ood=False)

    trainer = Trainer(cfg)
    # print(trainer.model)

    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
