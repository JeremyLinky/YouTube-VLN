

<p align="center">
  <img src="readme/lily.png" align="center" width="20%">
  
  <h3 align="center"><strong>Learning Vision-and-Language Navigation from YouTube Videos</strong></h3>

  <p align="center">
      <a href="https://scholar.google.com/citations?user=GPsw8IIAAAAJ" target='_blank'>Kunyang Lin</a><sup>1 2*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=KkpEXpsAAAAJ" target='_blank'>Peihao Chen</a><sup>1*</sup>&nbsp;&nbsp;&nbsp;
      <a href="" target='_blank'>Diwei Huang</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="" target='_blank'>Thomas H. Li</a><sup>6</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=EVsoTGkAAAAJ" target='_blank'>Mingkui Tan</a><sup>1 5†</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=PTeSCbIAAAAJ" target='_blank'>Chuang Gan</a><sup>3 4</sup>&nbsp;&nbsp;&nbsp;
    <br>
  <sup>1</sup>South China University of Technology&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Key Laboratory of Big Data and Intelligent Robot, Ministry of Education&nbsp;&nbsp;&nbsp;
  <sup>3</sup>UMass Amherst&nbsp;&nbsp;&nbsp;
  <sup>4</sup>MIT-IBM Watson AI Lab&nbsp;&nbsp;&nbsp;
  <sup>5</sup>Key Laboratory of Big Data and Intelligent Robot, Ministry of Education&nbsp;&nbsp;&nbsp;
  <sup>6</sup>Peking University Shenzhen Graduate School
  </p>

</p>


<p align="center">
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/ICCV-2023-blue.svg">
  </a>
  <a href="https://arxiv.org/abs/2307.11984" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2307.11984-red.svg">
  </a>
  <a href="LICENSE" target='_blank'>
    <img src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>


## Getting started

This project is developed with Python 3.6.13, Pytorch 1.10.1. Please install dependencies by follows:

```bash
conda env create -f env.yaml
conda activate lily
```
or install the environment by
```bash
pip install -r requirements.txt
```
Some packages may be missed you need to refer to the *requirements.txt* to install manually. 

## Preparing dataset

We provide the detailed construction process of our proposed YouTube-VLN dataset in [YouTube_VLN.md](scripts/README.md). The whole process may take a certain amount of time. If you want to directly use the generated dataset for training, please download the following data:blush:.

<!-- *data/YouTube-VLN/youtube_img_features* -->

1、Download the image features (totally 11 files) and put them into *data/YouTube-VLN/youtube_img_features*:

[image features 0、](https://drive.google.com/drive/folders/142JLO8G8CFetK1-EkChswH9x4IxoHaLD?usp=sharing)
[image features 1、](https://drive.google.com/drive/folders/1K2egIRzhuXWlVY6Kcn4hpa6DjinzaHQq?usp=drive_link)
[image features 2、](https://drive.google.com/drive/folders/1wiECk6-8yb5vcXQ3aWdXxtK51XEP_eUE?usp=sharing)
[image features 3、](https://drive.google.com/drive/folders/1waioWqRrybVFBRuABOrNZUzgbCCkwyW-?usp=sharing)
[image features 4、](https://drive.google.com/drive/folders/1PnQxp2hfFVdAx7YWhlJikQPYo554sHN9?usp=sharing)
[image features 5、](https://drive.google.com/drive/folders/1b5b6Y5qvi20zZ9s1D3ANMEx6nRBCr8j1?usp=sharing)
[image features 6、](https://drive.google.com/drive/folders/1y61b4bE4l3FaDOVeJAcLCw2jvJ-QeRh3?usp=sharing)
[image features 7、](https://drive.google.com/drive/folders/1e0j6cehnlIf60hW5ji0eenPtmjTgFRyM?usp=sharing)
[image features 8、](https://drive.google.com/drive/folders/1jC360MNm66-GXS45uur271cK7HbB2VBK?usp=sharing)
[image features 9、](https://drive.google.com/drive/folders/1eFQKtivQmw41NjDk3AubE2F9pHtHg0NT?usp=sharing)
[image features 10](https://drive.google.com/drive/folders/1nJsK2b1HZfqeKCFFzjtDCzKZkR0LR3to?usp=sharing)


<!-- *data/YouTube-VLN/ytb/merge+ytb_train.json* -->

2、Download the [trainset and testset](https://drive.google.com/drive/folders/1jbmPm2m8xNRSvX7AzkRo1GaMOyfouHgA?usp=sharingand) put them into *data/YouTube-VLN/ytb*.


3、Download the checkpoint of VilBERT pre-trained on [Conceptual Captions](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin) and then put it into *data/YouTube-VLN*.

<!-- *data/pretrained_model.bin* -->

4、Download the [matterport-ResNet-101-faster-rcnn features](https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip) and unzip it and then put it into *data/YouTube-VLN*.

5、Download the [instruction template](https://drive.google.com/file/d/1skdU4Kvs3E1jvqBSBvtsLsxMXYbtQ7fp/view?usp=sharing) and then put it into *data/task*.

6、Follow [download.py](scripts/download.py) to download the other data of tasks.
```bash
python scripts/download.py
```

## Training

### 1. Pre-traing Lily using YouTube-VLN

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 1234 \
    -m pretrain \
    --pre_dataset ytb \
    --from_pretrained data/pretrained_model.bin \
    --save_name ytbvln_2e5_500_MRT \
    --prefix merge+ \
    --separators \
    --masked_vision \
    --masked_language \
    --ranking \
    --traj_judge \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 500 \
    --save_epochs 100
```

### 2. Fine-tune with masking loss

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 5555 \
    -m train \
    --from_pretrained result/ytbvln_2e5_500_MRT/data/best_ranking.bin \
    --save_name ytbvln_2e5_500_MRT_ranking_30M \
    --masked_vision \
    --masked_language \
    --batch_size 12 \
    --num_epochs 30
```

### 3. Fine-tune with ranking loss and shuffling loss

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_port 5555 \
    -m train \
    --from_pretrained result/ytbvln_2e5_500_MRT_ranking_30M/data/29.bin \
    --save_name ytbvln_2e5_500_MRT_ranking_30M_30RS \
    --shuffle_visual_features \
    --ranking \
    --batch_size 16 \
    --num_epochs 30
```

### 4. Fine-tune with ranking loss and shuffling loss using speaker augmented data 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_port 5555 \
    -m train \
    --from_pretrained result/ytbvln_2e5_500_MRT_ranking_30M/data/29.bin \
    --save_name ytbvln_2e5_500_MRT_ranking_30M_30RSA \
    --prefix aug+ \
    --beam_prefix aug_ \
    --shuffle_visual_features \
    --ranking \
    --batch_size 16 \
    --num_epochs 30
```



## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --from_pretrained result/ytbvln_2e5_500_MRT_ranking_30M_30RSA/data/best_unseen.bin \
  --save_name ytbvln_2e5_500_MRT_ranking_30M_30RSA \
  --split val_unseen

python scripts/calculate-metrics.py results/ytbvln_2e5_500_MRT_ranking_30M_30RSA/test_val_unseen/_results_val_unseen.json
```

Here we provide our [trained model](https://drive.google.com/file/d/1Yosh7fyeYGFidO4QqD0nooXNQ11j_eu6/view?usp=sharing), feel free to test it.

## Citation
If you find this work helpful, please kindly consider citing our paper:

```bibtex
@article{lin2023ytbvln,
  title = {Learning Vision-and-Language Navigation from YouTube Videos},
  author = {Lin, Kunyang and Chen, Peihao and Huang, Diwei and Li, Thomas H. and Tan, Mingkui and Gan, Chuang},
  journal = {arXiv preprint arXiv:2307.11984}, 
  year = {2023},
}
```

```bibtex
@misc{lin2023ytbvln,
  title = {Learning Vision-and-Language Navigation from YouTube Videos},
  author = {Lin, Kunyang and Chen, Peihao and Huang, Diwei and Li, Thomas H. and Tan, Mingkui and Gan, Chuang},
  howpublished = {\url{https://github.com/JeremyLinky/YouTube-VLN}}, 
  year = {2023},
}
```


## Acknowledgements
Our code is partially modified from  [Airbert](https://github.com/airbert-vln/airbert), [video-dqn](https://github.com/uiuc-robovision/video-dqn) and [Probes-VLN](https://github.com/liangcici/probes-vln). Thanks for their awesome works and please consider citing them at the same time. 

## Contact
For any questions, please feel free to file an issue or contact:revolving_hearts::
```
Kunyang Lin: imkunyanglin@gmail.com
Diwei Huang: sediweihuang@gmail.com
```