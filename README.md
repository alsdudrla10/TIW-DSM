## TRAINING UNBIASED DIFFUSION MODELS FROM BIASED DATASET (TIW-DSM) (ICLR 2024) <br><sub>Official PyTorch implementation of the TIW-DSM </sub>



**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88), [Byeonghu Na](https://sites.google.com/view/byeonghu-na), Minsang Park, [JoonHo Jang](https://sites.google.com/view/joonhojang), [Dongjun Kim](https://sites.google.com/view/dongjun-kim), [Wanmo Kang](https://sites.google.com/site/wanmokang), and [Il-Chul Moon](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3)**   

| [openreview](https://openreview.net/forum?id=39cPKijBed) | arxiv | datasets | checkpoints |

--------------------

## Overview
![Teaser image](./figures/1igure1.PNG)



## Datasets
  - blabla
## Training
  ### Time-dependent discriminator 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond_vanilla --dg_weight_1st_order=0
   ```

  ### Diffusion model with TIW-DSM objective
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-cond-vp.pkl --outdir=samples/cifar_cond_vanilla --dg_weight_1st_order=0
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
author={Yeongmin Kim and Byeonghu Na and JoonHo Jang and Minsang Park and Dongjun Kim and Wanmo Kang and Il-chul Moon},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
}
```
This work is heavily built upon the code from
 - *Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577,2022.*
 - *Dongjun Kim\*, Yeongmin Kim\*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining generative process with discriminator guidance in score-based diffusion models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 16567–16598. PMLR, 23–29 Jul 2023*


