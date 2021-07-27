# Define a NN(Neural Network) with with PyTorch

PyTorch에서 신경망을 구성하는 방법을 알아본다.


## Introduce

PyTorch에서는 기본적으로 `torch.nn`패키지를 사용해 신경망을 구성할 수 있다.   
`torch.nn` 이외에도 신경망을 만들고 훈련시키는 여러 모듈과 클래스를 제공한다.

* `nn.Module` 은 **Layer(계층)**, 그리고 **output**을 반환하는 `forward(input)` 함수를 포함한다.



## Iplementation Steps

1. **Import** necessary libraries to get Data
2. **Define** and **Initialize** NN
3. **Detail** how Data go through the Model
4. (If needed) Apply Data to the Model and **Test** 


### 1. **Import** necessary libraries to get Data

가장 먼저, Pytorch 프레임워크를 사용하고 Pytorch의 신경망 모듈을 사용하기 위해 아래 라이브러리를 가져온다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F     # 이건... 잘?
```

### 2. **Define** and **Initialize** NN

다음으로 신경망을 정의하고 초기화하는 단계를 거친다.
신경망 구성은 신경망에 대응되는 **클래스**를 정의하는 방식으로 이루어진다.   

`nn.Module`을 참고하는 신경망 클래스 `Net`을 정의하기 위해선 `__init__(self)`함수를 작성해야 한다.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1st Conv2d Layer
        ## input: 1 input channer(image)
        ## output: 32 of 3 * 3 Conv feature maps
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # 2nd Conv2d Layer
        ## input: 32 inputs from conv1 layer
        ## output: 64 of 3 * 3 Conv feature maps
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Set Dropout
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # FC1
        self.fc1 = nn.Linear(9216, 128)
        
        # FC2
        ## Print 10 labels
        self.fc2 = nn.Linear(128, 10)

model = Net()
print(model)
```

즉, `__init__()` 함수에서 신경망의 디자인을 결정한다고 보면 된다. 위 예시 코드에서는 아주 기본적인 Layer 정의가 이루어졌지만...

```python
import timm

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.backbone = timm.create_model(...)
        self.state_channels = self.backbone.feature_info.channels()
        self.num_classes = "##"
        self.refinement = nn.Sequential(nn.Conv2d(...),
                                        nn.ReLU(...))
        self.gap = nn.AdaptiveAvgPool2d(...)
        self.refinement_h = nn.Sequential(nn.Conv2d(...),
                                          nn.ReLU(...))
        self.classifier = nn.Conv2d(...)
```

이런식으로 Layer 생성자를 받는 변수를 정의하는 등 활용 방법이 무궁무진한 것 같다.   

여하튼, 여기까지가 신경망-디자인-정의였으니 이렇게 정의된 신경망이 서로 **어떻게** 연결되어 있는지, 데이터가 신경망을 **어떤** 흐름으로 지나가는지 정의해야 한다.

### 3. **Detail** how Data go through the Model

`__init__()`함수에서 신경망의 디자인을 정의했다면(신경망에 어떤 Layer들이 존재하는지 정의했다면), 이제 각 계층 요소를 *어떻게* 연결짓고 데이터가 입력부터 출력까지 *어떻게* 흘러가는지 결정할 차례다.   

#### Function 'forward'

신경망 클래스의 `forward`함수에서 신경망의 흐름을 정의한다.   
여기서 `forward`란, **feed-forward** 알고리즘-네트워크에서 따온 이름이다. 

```python
class Net(nn.Module):
    def __init__(self):
        ...

    # x: data
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)     # compress x with start_dim=1
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
```

### 4. (If needed) Test

```python
data = torch.rand((1, 1, 28, 28))
model = Net()
result = model(data)
```