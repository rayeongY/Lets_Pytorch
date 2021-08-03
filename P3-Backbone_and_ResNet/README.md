# BackBone Network and ResNet

## `timm`

> `timm` is a deep-learning library created by Ross Wightman and is a collection of SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations and also training/validating scripts with ability to reproduce ImageNet training results.

`timm`은 SOTA(_State-Of-The-Art, 최고 수준의_)
+ 컴퓨터 비전 모델,
+ 레이어,
+ 유틸리티,
+ 옵티마이저,
+ 스케줄러,
+ 데이터로더,
+ 어그먼테이션 및 학습/테스트 스크립트   

를 모아놓은 딥러닝 라이브러리다. `timm`을 이용하면 굉장히 쉽게 모델을 만들 수 있다!   
아래와 같은 방식으로 모델 및 백본 네트워크를 쉽고/빠르고/간편하게 구성할 수 있다.

```python
import timm
import torch

model = timm.create_model('resnet18',
                          pretrained=...,
                          features_only=...,
                          ...)
```

이외에도 `timm`은 다양한 지원을 하고 있으며, [여기](https://fastai.github.io/timmdocs/)에서 볼 수 있다.

## Backbone Network

### Neck?


## ResNet