# Project - Linx

preparing for object detection challenge

![Linx eye](https://user-images.githubusercontent.com/15726007/177371786-167eb74b-3953-435a-9aa2-d1bb779f1688.png)

# Installation

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

## 3. Install MS coco api
```bash
git clone https://github.com/cocodataset/cocoapi
cd cocoapi
cd PythonAPI
make
python setup.py install
pip show pycocotools # v 2.0 설치 확인 
```


## 4. Install requirements
```bash
pip install -r requirements.txt
```

## 5. Setup coco dataset 
- download [COCO 2017](https://cocodataset.org/) dataset 
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

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
python -m torch.distributed.launch --nproc_per_node=4 \ #mproc 갯수는 사용가능한 cuda device 숫자보다 작게
    main.py \
	--output_dir logs/dino/swin \
	-c config/DINO/DINO_4scale_swin.py \ 
	--coco_path  /Path/to/COCO_DIR \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=/Path/To/BACKBONE_DIR
```

This implementation is variation of DINO "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)" - the current SOTA model of object detection


## LICNESE
DINO is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
