## TRAINING UNBIASED DIFFUSION MODELS FROM BIASED DATASET (TIW-DSM) (ICLR 2024) <br><sub>Official PyTorch implementation of the TIW-DSM </sub>



**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88), [Byeonghu Na](https://sites.google.com/view/byeonghu-na), Minsang Park, [JoonHo Jang](https://sites.google.com/view/joonhojang), [Dongjun Kim](https://sites.google.com/view/dongjun-kim), [Wanmo Kang](https://sites.google.com/site/wanmokang), and [Il-Chul Moon](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3)**   

| [openreview](https://openreview.net/forum?id=39cPKijBed) | arxiv | datasets | checkpoints |

--------------------

## Overview
![Teaser image](./figures/Figure1_v2.PNG)



### 1) Prepare a pre-trained score network
  - Download **edm-cifar10-32x32-uncond-vp.pkl** at [EDM](https://github.com/NVlabs/edm) for unconditional model.
  - Download **edm-cifar10-32x32-cond-vp.pkl** at [EDM](https://github.com/NVlabs/edm) for conditional model.
  - Place **EDM checkpoint** at the directory specified below.  
 
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
  │   ├── pretrained_score/edm-cifar10-32x32-cond-vp.pkl
  ├── ...
  ```

### 2) Generate fake samples
  - To draw 50k unconditional samples, run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond_vanilla --dg_weight_1st_order=0
   ```
  - To draw 50k conditional samples, run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-cond-vp.pkl --outdir=samples/cifar_cond_vanilla --dg_weight_1st_order=0
   ```

### 3) Prepare real data
  - Download [DG/data/true_data.npz](https://drive.google.com/drive/folders/18qh5QGP2gLgVjr0dh2g8dfBYZoGC0uVT) for unconditional model.
  - Download [DG/data/true_data_label.npz](https://drive.google.com/drive/folders/18qh5QGP2gLgVjr0dh2g8dfBYZoGC0uVT) for conditional model.
  - Place **real data** at the directory specified below.
  ```
  ${project_page}/DG/
  ├── data
  │   ├── true_data.npz
  │   ├── true_data_label.npz
  ├── ...
  ```

### 4) Prepare a pre-trained classifier
  - Download [DG/checkpoints/ADM_classifier/32x32_classifier.pt](https://drive.google.com/drive/folders/1gb68C13-QOt8yA6ZnnS6G5pVIlPO7j_y)
  - We train 32 resolution classifier from [ADM](https://github.com/openai/guided-diffusion).
  - Place **32x32_classifier.pt** at the directory specified below.
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── ADM_classifier/32x32_classifier.pt
  ├── ...
  ```

### 5) Train a discriminator
  - Download pre-trained unconditional checkpoint [DG/checkpoints/discriminator/cifar_uncond/discriminator_60.pt](https://drive.google.com/drive/folders/1Mf3F1yGfWT8bO0_iOBX-PWG3O-OLROE2) for the test.
  - Download pre-trained conditional checkpoint [DG/checkpoints/discriminator/cifar_cond/discriminator_250.pt](https://drive.google.com/drive/folders/1P1u7cz7kY1BJDPVrPNiFcksy_HCHY_bI) for the test.
  
  - Place **pre-trained discriminator** at the directory specified below.
  ```
  ${project_page}/DG/
  ├── checkpoints/discriminator
  │   ├── cifar_uncond/discriminator_60.pt
  │   ├── cifar_cond/discriminator_250.pt
  ├── ...
  ```
  - To train the unconditional discriminator from scratch, run:
   ```
   python3 train.py
   ```
   - To train the conditional discriminator from scratch, run:
   ```
   python3 train.py --savedir=/checkpoints/discriminator/cifar_cond --gendir=/samples/cifar_cond_vanilla --datadir=/data/true_data_label.npz --cond=1 
   ```

### 6) Generate discriminator-guided samples
  - To generate unconditional discriminator-guided 50k samples, run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond
   ```
  - To generate conditional discriminator-guided 50k samples, run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-cond-vp.pkl --outdir=samples/cifar_cond --dg_weight_1st_order=1 --cond=1 --discriminator_ckpt=/checkpoints/discriminator/cifar_cond/discriminator_250.pt --boosting=1
   ```
  
### 7) Evaluate FID
  - Download stat files at [DG/stats/cifar10-32x32.npz](https://drive.google.com/drive/folders/1xTdHz2fe71yvO2YpVfsY3sgH5Df7_b6y)
  - Place **cifar10-32x32.npz** at the directory specified below.
  ```
  ${project_page}/DG/
  ├── stats
  │   ├── cifar10-32x32.npz
  ├── ...
  ```
  - Run: 
  ```
  python3 fid_npzs.py --ref=/stats/cifar10-32x32.npz --num_samples=50000 --images=/samples/cifar_uncond/
   ```
  ```
  python3 fid_npzs.py --ref=/stats/cifar10-32x32.npz --num_samples=50000 --images=/samples/cifar_cond/
   ```

## Experimental Results
### EDM-G++
|FID-50k |Cifar-10|Cifar-10(conditional)|FFHQ64|
|------------|------------|------------|------------|
|EDM|2.03|1.82|2.39|
|EDM-G++|1.77|1.64|1.98|

### Other backbones
|FID-50k  |Cifar-10|CelebA64|
|------------|------------|------------|
|Backbone|2.10|1.90|
|Backbone-G++|1.94|1.34|

Note that we use [LSGM](https://github.com/NVlabs/LSGM) for Cifar-10 backbone, and [Soft-Truncation](https://github.com/Kim-Dongjun/Soft-Truncation) for CelebA64 backbone. <br>
See [alsdudrla10/DG_imagenet](https://github.com/alsdudrla10/DG_imagenet) for the results and released code on ImageNet256.

### Samples from unconditional Cifar-10
![Teaser image](./figures/Figure3.PNG)

### Samples from conditional Cifar-10
![Teaser image](./figures/Figure4.PNG)


## Reference
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
kim2024training,
title={Training Unbiased Diffusion Models From Biased Dataset},
author={Yeongmin Kim and Byeonghu Na and JoonHo Jang and Minsang Park and Dongjun Kim and Wanmo Kang and Il-chul Moon},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
}
```
This work is heavily built upon the code from
 - *Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577,2022.*
 - *Dongjun Kim\*, Yeongmin Kim\*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining generative process with discriminator guidance in score-based diffusion models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 16567–16598. PMLR, 23–29 Jul 2023*


