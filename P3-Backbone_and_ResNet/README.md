# BackBone Network and ResNet

## 시작하기에 앞서
이 문서를 작성하기 위해 Backbone 및 Neck에 관해 찾던 중, 꽤나 도움이 되는 글을 보았다. 나처럼 아주아주 지식이 부족한 사람은 [이 포스트](https://velog.io/@hewas1230/ObjectDetection-Architecture)를 먼저 읽고 와도 좋을 것이다.   

정도에 따라 다르겠지만, 일반적으로 딥러닝 모델의 구조는 매우 복잡하다. _modeling_ 할 신경망이 어느 정도의 인공 지능을 가지느냐에 따라 구현해야할 모듈-_기능_-이 늘어나기 때문이다.   
우리가 학부 과정에서 배우는 기초 CNN 등의 모델은 보통   

+ 입력: 이미지 데이터
    + Task 0: 입력받은 데이터(이미지)에서 특징정보를 추출함
    + Task 1: 특징정보를 기준으로 이미지 자체를 분류함

에서 끝난다. 그러나, 예를 들어 일반적인 **Object Detecion** 모델을 구성한다고 하면

+ 입력: 이미지 데이터   
    + Task 0: 입력받은 데이터(이미지)에서 특징정보를 추출함
    + Task 1: 특징정보를 이용해서 객체를 찾아냄(탐색, Locatlization)
    + Task 2: 찾아낸 객체의 클래스를 구분함(분류, Classification)
    + etc.

...정도의 흐름을 따를 것이다. 말로는 참 쉽다. 아무튼, 이런 **"Deep"** 한 모델은 학습 효율도 떨어지는데, 이때 Backbone Network를 사용하면 도움이 된다고 한다.   


#
## Backbone Network

> **Backbone**은 '등뼈'라는 사전적 정의를 가지고 있다.




### Neck?


#
## ResNet

이 신경망은 **Residual Function**을 학습하기에 ResNet이라는 이름을 갖게 되었다. 자세한 설명은 하위 항목에서 써보겠다.

### Residual Function이란?



#
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


## 참고
https://velog.io/@hewas1230/ObjectDetection-Architecture   
https://ehdgns.tistory.com/109   
https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8