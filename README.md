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
# CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch
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

## 6. Compiling CUDA operators
```sh
cd models/dn_dab_deformable_detr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## 7. Run(Debug)
```sh
python main.py -m dn_dab_deformable_detr \
              --output_dir logs/dn_dab_deformable_detr/R50 \
              --batch_size 1 \
              --coco_path ~/.linx/datasets/coco_2017 \
              --transformer_activation relu \
              --use_dn \
              --device cpu
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
