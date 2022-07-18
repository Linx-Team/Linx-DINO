# Project - Linx

preparing for object detection challenge

![Linx eye](https://user-images.githubusercontent.com/15726007/177371786-167eb74b-3953-435a-9aa2-d1bb779f1688.png)

# Installation

## 0. Clone this project
```bash
git clone https://github.com/Linx-Team/Linx-DINO.git
cd Linx-DINO
```

## 1. Create Conda Environment

```bash
conda create --name linx python=3.8 -y
conda activate linx
```

## 2. Install PyTorch

- with GPU (for details, see https://pytorch.org)

```bash
# CUDA 11.1
# cuda 11.1 version에 대해서는 문제가 있어 아래 명령으로 변경
# conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

- with CPU (for details, see https://pytorch.org)

```bash
# conda
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch
```

## 3. Install requirements
```bash
pip install -r requirements.txt
```

## 4. Install MS coco api
```bash
git clone https://github.com/cocodataset/cocoapi
cd cocoapi
cd PythonAPI
make
python setup.py install
pip show pycocotools # v 2.0 설치 확인 
```

## 5. Setup coco dataset 
- download [COCO 2017](https://cocodataset.org/) dataset 
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip -qq train2017.zip
unzip -qq val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
```

- organize them as following:
```
COCO_DIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

## 6. Setup backbone pretrained models
- case: swin transformer
```bash
wget -c https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
```
- move the downloade weight file in the backbone directory as following
```
BACKBONE_DIR/
  └── swin
```

## 7. Compiling CUDA operators
```sh
cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## 8. Run(Debug) 
- scrips/ directory 참조
```sh
# main.py 단독 - config 수정 이후  - scripts/ 참조
#distributed를 사용하는 경우 - mproc 갯수는 사용가능한 cuda device 숫자보다 작게, 
python -m torch.distributed.launch --nproc_per_node=4 main.py --output_dir logs/dino/swin -c config/DINO/DINO_4scale_swin.py --coco_path /home/ubuntu/.linx/datasets/coco_2017 --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=/home/ubuntu/.linx/backbones/swin
```

This implementation is mainly from DINO "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)" - the current SOTA model of object detection

