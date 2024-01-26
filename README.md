# WSOE


Implementation for WSOE: Weakly Supervised Outlier Exposure for Object-level Out-of-distribution Detection (under review).

The codebase is heavily based on [VOS](https://github.com/deeplearning-wisc/vos), [ProbDet](https://github.com/asharakeh/probdet) and [Detectron2](https://github.com/facebookresearch/detectron2).
## Requirements
```
pip install -r requirements.txt
```
In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset Preparation

**PASCAL VOC**

Download the processed VOC 2007 and 2012 dataset from [here](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         └── val_coco_format.json

**Auxiliary OOD Dataset**

In this work, ImageNet1k Validation Set serves as an auxiliary OOD dataset. Download preprocessed ImageNet from [here](https://drive.google.com/file/d/1GsTVHYaKWM40VlsFGpq_x9drhh3HYrKl/view?usp=drive_link)

**Imba-COCO**

Download COCO2017 dataset from the [official website](https://cocodataset.org/#home). 

Download the OOD dataset (json file) [here](https://drive.google.com/file/d/1GsTVHYaKWM40VlsFGpq_x9drhh3HYrKl/view?usp=drive_link). 


Put the processed OOD json file to ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            └── instances_val2017_ood_rm_overlap.json
         ├── train2017
         └── val2017

**Balanced-COCO**

Download our Balanced-COCO [here](https://drive.google.com/file/d/1sTEGHOBuHZ9UMTCFfgwcuQ9sO3Ba0U2C/view?usp=drive_link)

The dataset folder should have the following structure:
<br>

     └── Balanced_COCO_DATASET_ROOT
         |
         ├── COCO-Format
         └── Images

**Balanced-OpenImages**

Download our Balanced-OpenImages [here](https://drive.google.com/file/d/1mtKt1jHYt7q9jhymMdjmwM9zxnVFxfBa/view?usp=drive_link)

The dataset folder should have the following structure:
<br>

     └── Balanced_OpenImages_DATASET_ROOT
         |
         └── ood_classes_rm_overlap

**OOD Validation Set**
Download our OOD Validation Set [here](https://drive.google.com/file/d/19AtuRDLISe5wfRw9MIZWgRRqC_mmeIKv/view?usp=drive_link)
<br>

    └── Validation_Set_Root
         |
         ├── COCO-Format
         └── Images

## Training and Envaluation

Firstly, enter the detection folder by running
```
cd detection
```

To train the model, firstly modify dataset address by changing "dataset-dir" and "oe-dataset-dir" according to your local dataset address. "dataset-dir" contains ID data, and "oe-dataset-dir" contains auxiliary OOD data. 

**ResNet50 as Backbone Network**

Training
```
python train_net_ws_pseudo_oe_st.py
    --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted
    --oe-dataset-dir path/to/datasets/ImageNet1k_Val_OE 
    --config-file VOC-Detection/oe/wsoe.yaml
    --num-gpus 2 --random-seed 0
```

Envaluation
```
# evaluate ID dataset
python apply_net.py
       --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted\
       --test-dataset voc_custom_val --config-file VOC-Detection/oe/wsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

## evaluate balanced_openimages_ood
python apply_net.py
       --dataset-dir path/to/datasets/OpenImages/
       --test-dataset balanced_openimages_ood --config-file VOC-Detection/oe/wsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0
    
### evaluate coco_ood_val     
python apply_net.py
       --dataset-dir path/to/datasets/coco/
       --test-dataset coco_ood_val --config-file VOC-Detection/oe/wsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

# evaluate balanced_coco_ood
python apply_net.py
       --dataset-dir path/to/datasets/BalancedBenchmark
       --test-dataset balanced_coco_ood --config-file VOC-Detection/oe/wsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0
```

```
cd ..
python voc_coco_plot.py 
    --name wsoe.yaml 
    --ood-dataset coco_ood_val --model oe 
    --thres xxx --energy 1 --seed 0

python voc_coco_plot.py 
    --name wsoe.yaml 
    --ood-dataset balanced_coco_ood --model oe 
    --thres xxx --energy 1 --seed 0
    
python voc_coco_plot.py --name wsoe.yaml 
    --ood-dataset balanced_openimages_ood --model oe 
    --thres xxx --energy 1 --seed 0

```
Here the threshold is determined according to [ProbDet](https://github.com/asharakeh/probdet). It will be displayed in the screen as you finish evaluating on the in-distribution dataset.

**RegNetX-4.0GF as Backbone Network**

Training
```
cd detection

python train_net_ws_pseudo_oe_st.py
    --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted
    --oe-dataset-dir path/to/datasets/ImageNet1k_Val_OE 
    --config-file VOC-Detection/oe/wsoe_regnetx.yaml
    --num-gpus 2 --random-seed 0
```
Envaluation

```
#envaluate ID dataset
python apply_net.py
       --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted\
       --test-dataset voc_custom_val --config-file VOC-Detection/oe/wsoe_regnetx.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

#evaluate coco_ood_val     
python apply_net.py
       --dataset-dir path/to/datasets/coco/
       --test-dataset coco_ood_val --config-file VOC-Detection/oe/wsoe_regnetx.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

cd ..
python voc_coco_plot.py 
    --name wsoe_regnetx.yaml 
    --ood-dataset coco_ood_val --model oe 
    --thres xxx --energy 1 --seed 0
```

**Fully Supervised Outlier Exposure**

Training
```
cd detection

python train_net_ws_pseudo_oe_st.py
    --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted
    --oe-dataset-dir path/to/datasets/ImageNet1k_Val_OE 
    --config-file VOC-Detection/oe/fsoe.yaml
    --num-gpus 2 --random-seed 0
```
Envaluation

```
#envaluate ID dataset
python apply_net.py
       --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted\
       --test-dataset voc_custom_val --config-file VOC-Detection/oe/fsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

#evaluate coco_ood_val     
python apply_net.py
       --dataset-dir path/to/datasets/coco/
       --test-dataset coco_ood_val --config-file VOC-Detection/oe/fsoe.yaml
       --inference-config Inference/standard_nms.yaml --random-seed 0
       --image-corruption-level 0 --visualize 0

cd ..
python voc_coco_plot.py 
    --name fsoe 
    --ood-dataset coco_ood_val --model oe 
    --thres xxx --energy 1 --seed 0
```

## Envaluation with Pretrained Weights

In order to perform envaluation, we can do the following:
1. Download pretrained models and put them into wsoe/detection/data/checkpoints. The pretrained models can be downloaded from [wsoe](https://drive.google.com/file/d/1KoSFB-uOq8ta8jZDzn9RgFkTFTg-AJEV/view?usp=drive_link), [wsoe_regnetx](https://drive.google.com/file/d/1RmDtc9_i6ymisXmvLTWHLmFeCiUPTIYa/view?usp=drive_link), [fsoe](https://drive.google.com/file/d/11iO9bBE-FRM8XC_0NxqgKUUfp47lzgR2/view?usp=drive_link).
2. For wsoe, modify the necessary parameters in the configuration file `detection/configs/VOC-Detection/oe/wsoe.yaml`. More importanly, modify the folder paths for model weights to your local path, i.e. `detection/data/checkpoints/wsoe_resnet.pth`.
3. Perform envaluation with these scripts:

    ```
    cd detection

    #envaluate ID dataset
    python apply_net.py
        --dataset-dir path/to/datasets/VOCdatasets/VOC_0712_converted\
        --test-dataset voc_custom_val --config-file VOC-Detection/oe/wsoe.yaml
        --inference-config Inference/standard_nms.yaml --random-seed 0
        --image-corruption-level 0 --visualize 0

    #evaluate coco_ood_val     
    python apply_net.py
        --dataset-dir path/to/datasets/coco/
        --test-dataset coco_ood_val --config-file VOC-Detection/oe/wsoe.yaml
        --inference-config Inference/standard_nms.yaml --random-seed 0
        --image-corruption-level 0 --visualize 0

    cd ..
    python voc_coco_plot.py 
        --name wsoe 
        --ood-dataset coco_ood_val --model oe 
        --thres xxx --energy 1 --seed 0
    ```


