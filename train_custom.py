from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
import argparse
import json


def main(args):
    if args.classes:
        if os.path.isfile(args.classes):
            classes = json.load(open(args.classes))
        else:
            classes = args.classes.split(",")
        metadata = {"thing_classes": classes}
    else:
        metadata = {}
    register_coco_instances(args.dataset_name + "_train", metadata, args.train_anno_json, args.train_image_dir)
    register_coco_instances(args.dataset_name + "_test", metadata, args.test_anno_json, args.test_image_dir)

    # MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
    #
    # for d in ["train", "val"]:
    #     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    #     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.DATASETS.TRAIN = (args.dataset_name + "_train",)
    cfg.DATASETS.TEST = (args.dataset_name + "_test",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.img_per_batch
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image  # faster, and good enough for this toy dataset (default: 512)
    if metadata:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume_train)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--train_anno_json', default=None, type=str)
    parser.add_argument('--test_anno_json', default=None, type=str)
    parser.add_argument('--train_image_dir', default=None, type=str)
    parser.add_argument('--test_image_dir', default=None, type=str)
    parser.add_argument('--classes', default=None, type=str)
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--resume_train', default=False, action='store_true')
    parser.add_argument('--img_per_batch', default=2, type=int)
    parser.add_argument('--batch_size_per_image', default=512, type=int)

    main(parser.parse_args())
