<div align="center">
    <img src="https://github.com/AffectAI/MGNN/raw/main/resources/AffectAI_logo.png" width="600"/>
    <div>&nbsp;</div>

# **DepMGNN: Matrixial Graph Neural Network for Video-based Automatic Depression Assessment**

</div>

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🥳 🚀 What's New](#--whats-new-)
- [📖 Introduction](#-introduction-)
- [🛠️ Installation](#️-installation-)
- [👨‍🏫 Get Started](#-get-started-)
- [👀 Models](#-models-)
- [🙌 Results](#-results-)
- [🖊️ Citation](#️-citation-)

## 🥳 🚀 What's New [🔝](#-table-of-contents)

👏👏👏**Congratulations (2024.12.10)**: Our work **DepMGNN: Matrixial Graph Neural Network for Video-based Automatic Depression Assessment** has been accepted by AAAI-2025 and selected for oral presentation!

## 📖 Introduction [🔝](#-table-of-contents)

<div align="center">
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/vector_graph.gif" height="300px">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/matrixial_graph.gif" height="300px">
  <p style="font-size:1.5vw;"> Existing vector-style graph (left) and Our matrixial-style graph (right)</p>
</div>

<div align="center">
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/mgnn.gif" height="350px"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/updated_mgnn.gif" height="350px"/>
    <p style="font-size:1.5vw;">Our clip-level spatio-temporal matrixial graph (left) and the updated matrixial graph by our MGNN (right)</p>
</div>

<div align="center">
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/main.png" height="250px"/>
    <p style="font-size:1.5vw;">Pipeline of our DepMGNN</p>
</div>


## 🛠️ Installation [🔝](#-table-of-contents)

MGNN is built on top of [mmaction2](https://github.com/open-mmlab/mmaction2) and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html).

Please refer to their official tutorials for detailed installation instructions.

<details close>
<summary>Quick instructions</summary>

```shell
conda create -n MGNN python=3.9 -y
conda activate MGNN

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0

pip install torch-geometric==2.4.0

pip install einops
pip install timm
pip install seaborn

git clone https://github.com/AffectAI/MGNN.git
cd MGNN
pip install -v -e .
```

</details>

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

### Step 1: Preparation
1. Apply for and download the AVEC2013, AVEC2014, and First Impression datasets from their official websites.
2. Crop the face from original videos use [face_detect.py](https://github.com/AffectAI/MGNN/blob/main/face_detect.py)
    <details close>
    <summary>Quick instructions</summary>

    ```shell
    pip install pyfacer

    python face_detect.py
    ```
    
    </details>
3. Place the cropped face frames in the corresponding folder under [./datasets](https://github.com/AffectAI/MGNN/tree/main/datasets). The corresponding dataset labels have been uploaded to the directory.

4. Download the pretrained resnet-50 model on vggface2 and put it into ./pretrained_models.

### Step 2: Training

```shell
# Training on AVEC 2014
bash ./tools/dist_train.sh configs/depression/mgnn_depression_avec2014_res50.py num_gpus --seed 0

# Training on AVEC 2013
bash ./tools/dist_train.sh configs/depression/mgnn_depression_avec2013_res50.py num_gpus --seed 0

# Training on First Impression dataset
bash ./tools/dist_train.sh configs/depression/mgnn_personality_first_impression_res50.py num_gpus --seed 0

```

### Step 3: Testing

```shell
# Testing on AVEC 2014 Northwind and Freeform
bash ./tools/dist_test.sh configs/depression/mgnn_depression_avec2014_res50_test_fusion.py your/model/path/your_model.pth 1

# Testing on AVEC 2013
bash ./tools/dist_test.sh configs/depression/mgnn_depression_avec2013_res50.py your/model/path/your_model.pth 1

# Testing on First Impression dataset
bash ./tools/dist_test.sh configs/depression/mgnn_personality_first_impression_res50.py your/model/path/your_model.pth 1

```


## 👀 Models [🔝](#-table-of-contents)

1. Pretrained models: [vggface2 pretrained resnet-50 model](https://pan.baidu.com/s/1JsGJb9kpH6wOMYls7Y5aIg?pwd=9gp4)

2. MGNN AVEC 2014: [MGNN (resnet-50)](https://pan.baidu.com/s/1qvLcD0cvH4RhjvO309ZDRQ?pwd=yrjk)

3. MGNN AVEC 2013: [MGNN (resnet-50)](https://pan.baidu.com/s/1fYSpHmkbCQx4Yh-8xNJ2pg?pwd=gvrv)

4. MGNN First Impression: [MGNN (resnet-50)](https://pan.baidu.com/s/1xPieIIr5t2GlpzG1AwflsA?pwd=45t2)


## 🙌 Results [🔝](#-table-of-contents)

<div align="center">
  <img src="https://github.com/AffectAI/MGNN/raw/main/resources/results.png" height="380px">
</div>

## 🖊️ Citation [🔝](#-table-of-contents)

If you find this project useful in your research, please consider cite:

```BibTeX

```
