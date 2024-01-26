import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.data.datasets.pascal_voc import register_pascal_voc

# Project imports
import core.datasets.metadata as metadata

from detectron2.data.catalog import DatasetCatalog
from core.datasets.pascal_voc_oe import register_pascal_voc_oe
from collections import ChainMap


def setup_all_datasets(dataset_dir, oe_dataset_dir, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_voc_dataset(dataset_dir)
    setup_ImageNet1k_Val_OE_dataset(oe_dataset_dir)
    
    setup_coco_dataset(
        dataset_dir,
        image_root_corruption_prefix=image_root_corruption_prefix)
    setup_coco_ood_dataset(dataset_dir)

    setup_balanced_coco_ood_datasets(dataset_dir)

    setup_balanced_openim_ood_dataset(dataset_dir)
    setup_balanced_coco_ood_val_datasets(dataset_dir)

def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_balanced_openim_ood_dataset(dataset_dir):
    """
    sets up openimages out-of-distribution dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.

    Only validation is supported.
    """
    test_image_dir = os.path.join(dataset_dir + 'ood_classes_rm_overlap', 'images')

    test_json_annotations = os.path.join(
        dataset_dir + 'ood_classes_rm_overlap', 'COCO-Format', 'balanced_openimages_coco_format.json')

    register_coco_instances(
        "balanced_openimages_ood",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "balanced_openimages_ood").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "balanced_openimages_ood").thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_balanced_coco_ood_datasets(dataset_dir):
    test_image_dir = os.path.join(dataset_dir,'Images')
    test_json_annotations = os.path.join(
        dataset_dir,'COCO-Format', 'val_coco_format.json')
    register_coco_instances(
        "balanced_coco_ood",
        {},
        test_json_annotations,
        test_image_dir
    )
    MetadataCatalog.get('balanced_coco_ood').thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get('balanced_coco_ood').thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_balanced_coco_ood_val_datasets(dataset_dir):
    test_image_dir = os.path.join(dataset_dir,'Images')
    test_json_annotations = os.path.join(
        dataset_dir,'COCO-Format', 'val_coco_format.json')
    register_coco_instances(
        "balanced_coco_val_ood",
        {},
        test_json_annotations,
        test_image_dir
    )
    MetadataCatalog.get('balanced_coco_val_ood').thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get('balanced_coco_val_ood').thing_dataset_id_to_contiguous_id = metadata.OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_voc_dataset(dataset_dir):
    train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    train_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = metadata.VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_coco_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    # test_json_annotations = os.path.join(
    #     dataset_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_ood_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    MetadataCatalog.get(
        "coco_ood_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_ood_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_ImageNet1k_Val_OE_dataset(oe_dataset_dir):

    dataset_name = 'ImageNet1k_Val_NonoverlapOE'
    split = 'oe_nonoverlap'
    year = '2012'
    register_pascal_voc_oe(dataset_name, oe_dataset_dir, split, year)

def setup_pseudo_label_dataset(dataset_name, images_path, pseudo_label_path):
    register_coco_instances(
        dataset_name,
        {},
        pseudo_label_path,
        images_path)
    MetadataCatalog.get(
        dataset_name).thing_classes = metadata.VOC_OE_THING_CLASSES
    MetadataCatalog.get(
        dataset_name).thing_dataset_id_to_contiguous_id = metadata.VOC_OE_THING_DATASET_ID_TO_CONTIGUOUS_ID
