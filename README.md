## TRAINING UNBIASED DIFFUSION MODELS FROM BIASED DATASET (TIW-DSM) (ICLR 2024) <br><sub>Official PyTorch implementation of the TIW-DSM </sub>



**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88), [Byeonghu Na](https://sites.google.com/view/byeonghu-na), Minsang Park, [JoonHo Jang](https://sites.google.com/view/joonhojang), [Dongjun Kim](https://sites.google.com/view/dongjun-kim), [Wanmo Kang](https://sites.google.com/site/wanmokang), and [Il-Chul Moon](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3)**   

| [openreview](https://openreview.net/forum?id=39cPKijBed) | [arxiv](https://arxiv.org/abs/2403.01189) | [datasets](https://drive.google.com/drive/u/0/folders/1RakPtfp70E2BSgDM5xMBd2Om0N8ycrRK)  | [checkpoints](https://drive.google.com/drive/u/0/folders/1vYLH8UNlXWZarn0IOtiPuU8FvBFqJvTP) |

--------------------

## Overview
![Teaser image](./figures/figure1.PNG)

## Requirements
The requirements for this code are the same as those outlined for [EDM](https://github.com/NVlabs/edm).

## Datasets
  - Download from [datasets](https://drive.google.com/drive/u/0/folders/1RakPtfp70E2BSgDM5xMBd2Om0N8ycrRK) with simmilar directory structure
## Training
  ### Time-dependent discriminator 
  - CIFAR10 LT / 5% setting
  ```
  python train_classifier.py
   ```
  - CIFAR10 LT / 10% setting
  ```
  python train_classifier.py --savedir=/checkpoints/discriminator/cifar10/unbias_1000/ --refdir=/datasets/cifar10/discriminator_training/unbias_1000/real_data.npz --real_mul=10
   ```
  - CelebA64 / 5% setting
  ```
  python train_classifier.py --feature_extractor=/checkpoints/discriminator/feature_extractor/64x64_classifier.pt --savedir=/checkpoints/discriminator/celeba64/unbias_8k/ --biasdir=/datasets/celeba64/discriminator_training/bias_162k/fake_data.npz --refdir=/datasets/celeba64/discriminator_training/unbias_8k/real_data.npz --img_resolution=64
   ```

  ### Diffusion model with TIW-DSM objective
  ```
  blabla
   ```

## Sampling
  - blabla

## Evaluation
  - blabla


## Reference
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
kim2024training,
title={Training Unbiased Diffusion Models From Biased Dataset},
author={Yeongmin Kim and Byeonghu Na and Minsang Park and JoonHo Jang and Dongjun Kim and Wanmo Kang and Il-chul Moon},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=39cPKijBed}
}
```
This work is heavily built upon the code from
 - *Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577,2022.*
 - *Dongjun Kim\*, Yeongmin Kim\*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining generative process with discriminator guidance in score-based diffusion models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 16567–16598. PMLR, 23–29 Jul 2023*


