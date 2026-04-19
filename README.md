
markdown
# PGDT: Physics-Guided Dual-Teacher Framework for Semi-Supervised SAR Ship Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.3.0-green.svg)](https://github.com/open-mmlab/mmdetection)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official PyTorch implementation of the paper **"Physics-Guided Dual-Teacher Framework for Semi-Supervised SAR Ship Detection"** (accepted by *IEEE Geoscience and Remote Sensing Letters*). 

This project is built upon the open-source detection toolbox [MMDetection](https://github.com/open-mmlab/mmdetection).

## 💡 Introduction
Existing semi-supervised object detection (SSOD) methods are prone to self-reinforcing errors in SAR imagery due to their homogeneous network architectures and confirmation bias. 

To mitigate this, we propose the **Physics-Guided Dual-Teacher (PGDT)** framework. Distinct from conventional ensembles, PGDT establishes a heterogeneous expert system:
- **Semantic Teacher**: A data-driven branch for high-level contextual abstraction.
- **Physics Teacher**: A parameter-free expert leveraging IS-Transform for objective scattering validation.
- **Dual-Arbitration Mechanism**: Employs an adaptive curriculum schedule, synergistically utilizing *Physics Veto* to prune clutter-induced false positives and *Physics Mining* (with *Teacher-Guided Refinement*) to retrieve stealthy targets.

## 🛠️ Installation

### Tested Environment
- **OS**: Ubuntu
- **GPU**: 1 × NVIDIA GeForce RTX 3090 (24GB)
- **CUDA**: 12.4
- **MMDetection**: 3.3.0

**Step 1. Environment Preparation**

First, create a clean conda environment and **ensure it is activated** before proceeding.

```shell
conda create -n pgdt python=3.8 -y
conda activate pgdt
```
**Step 2. Install Core Dependencies**

Install PyTorch and torchvision. Please ensure the `pytorch-cuda` version matches your local driver.

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3. Install MMDetection Framework**

Install the MMLab family libraries and build MMDetection from source.

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
```

## 📂 Data 
We evaluate PGDT on the **SSDD** and **HRSID** datasets. Please download the datasets and organize them in the `data/` folder as follows:
```text
PGDT-SAR/
  └── data/
      ├── SSDD/
      │   ├── annotations/
      │   │   ├── instances_train_5percent.json
      │   │   ├── instances_train_10percent.json
      │   │   ├── instances_unlabeled_95percent.json
      │   │   ├── instances_unlabeled_90percent.json
      │   │   └── test.json
      │   └── images/
      │       ├── train/
      │       └── test/
      └── HRSID/
          ├── annotations/
          │   ├── instances_train_5percent.json
          │   ├── instances_train_10percent.json
          │   ├── instances_unlabeled_95percent.json
          │   ├── instances_unlabeled_90percent.json
          │   └── test2017.json
          └── images/
```

## 🚀 Getting Started

### Training
To train the PGDT model on a single GPU, select the config file based on your desired dataset and labeled data ratio (5% or 10%):

**For HRSID (10% Labeled):**
```shell
python tools/train.py configs/pgdt/pgdt_faster_rcnn_r50_fpn_10percent_hrsid.py
```
*Note: To train with 5% labeled data, simply replace `10percent` with `5percent` in the command above.*

**For SSDD (10% Labeled):**
```shell
python tools/train.py configs/pgdt/pgdt_faster_rcnn_r50_fpn_10percent_ssdd.py
```

### Evaluation
To evaluate the trained model on the test set:
```shell
python tools/test.py configs/pgdt/pgdt_faster_rcnn_r50_fpn_10percent_hrsid.py work_dirs/pgdt_faster_rcnn_r50_fpn_10percent_hrsid/latest.pth
```

## 📝 Citation
If you find this project useful in your research, please consider citing our paper:
```bibtex

```

## 🙏 Acknowledgement
This project is built upon the foundational framework of [MMDetection](https://github.com/open-mmlab/mmdetection). Furthermore, a large part of the semi-supervised training codebase is inspired by and borrowed from [SoftTeacher](https://github.com/microsoft/SoftTeacher). We sincerely thank the original authors for their outstanding open-source contributions!

## ✉️ Contact
For any questions, please feel free to open an issue or contact `kklt_zhang@dlmu.edu.cn`.
```



