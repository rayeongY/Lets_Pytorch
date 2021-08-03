# Data Management with YAML

딥러닝 모델을 구현하려면 데이터셋에 대한 정보, 모델에 대한 정보 등 매우 다양하며 구조적으로 복잡한 데이터를 처리해야한다. YAML이 제공하는 `.yaml` 파일과 **parser** 라이브러리의 `parse_yaml` 모듈을 사용하면 이 처리 과정이 꽤나 간단해진다.


## Set parameters with argparse library

Python 코드에서 명령행 파싱을 하려면 표준 라이브러리의 **argparse** 모듈을 사용하면 된다.   
기본 골격은 아래와 같다.

```python
import argparse

## Parsing arguments Example
parser = argparse.ArgumentParser()
parser.add_argument('--parameterA', type=..., default=..., help=..., required=...)
...
args = parser.parse_args()

paramA = args.parameterA
```

1. 라이브러리를 `import`
2. `parser` 객체 생성
3. `add_argument` 함수로 필요한 인자와 해당 인자의 세부 사항 추가
4. `parser`의 `parse_args` 함수를 호출해 인자를 파싱하고 `args`에 저장
5. `args.(parameter 이름)` 표현으로 파싱한 인자(parameter)를 사용

이런 식으로 간단하게 프로그램에서 사용할 parameter 데이터를 관리할 수 있다. 있지만... argument 데이터 자체가 복잡해지면, 어떤 식으로 코드를 작성해야할지 막막해지는게 사실이다.

## Set arguments with Configuration Files 

예를 들어 딥러닝 모델이 처리할 Dataset에 대한 정보를 argument로 받는다고 하자. 이때 처리할 만한 정보로는

+ Dataset의 이름
+ 파일 시스템에서 Dataset이 존재하는 경로
+ Dataset의 Data가 가질 수 있는 class 종류
+ Dataset의 Data에 대한 정보(최대 크기, 최대 길이 등)

등이 있을 것이다. 이정도로 끝나면 위에서 기술한 `arg_parse()` 만으로 argument를 관리할 수 있을지도 모른다. 그러나 복잡한 모델을 구성할수록 프로그램이 인자로 받는 parameter 데이터가 많아지므로 `add_argument()` 함수로 직접-하나하나 argument를 추가하기엔 무리가 있을 것이다.   

때문에 마크업 언어로 argument에 대한 Configuration File(프로그램의 매개 변수 및 초기 설정을 구성하는 데 사용되는 파일)을 작성하고, 파싱한 argument에 configuration 파일을 파싱한 결과(예: dictionary)를 세팅하는 방법을 알아야 한다.


## YAML이란?

**YAML**은 많이 사용되는 데이터 직렬화 양식 중 하나로 JSON, XML과 같은 마크업 언어이다.   
사람이 읽고 사용하기에 친화적인 편이라 configuration file 작성에 자주 쓰인다.


## YAML로 작성된 .yaml 파일에서 parameter 정보를 parsing하기

YAML 문법은 꽤나 간단하다.

```yaml
$ vi ./configs/datasets/people_dataset.yaml

DATASET:
  NAME: "birthday_id"
  PATH: "./dataset/birthday_id"
  CLASSES: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  LENGTH: 6
```

위의 `.yaml` 파일은 이렇게 해석할 수 있을 것이다.

+ Dataset의 이름은 `birthday_id`이다.
+ Dataset은 parser 코드와 동일한 디렉토리에 존재하는 *birthday_id* 폴더에 있다.
+ Dataset의 Data는 0부터 9까지의 int로 구성된다.
+ Dataset의 Data는 6자리이다.

`parser` 라이브러리는 `.yaml` 파일을 파싱하는 함수 `parse_yaml`을 제공한다.

```python
from common.parser import parse_yaml
import argparse

# Define parser
parser = argparse.ArgumentParser()

# Set arguments
parser.add_argument('--dataset', type=str, default="./configs/datasets/people_dataset.yaml")
args = parser.parse_args()

# Parse yaml files
dataset_option = parse_yaml(args.dataset)

# Use arguments
dataset_name = dataset_option["NAME"]
dataset_path = dataset_option["PATH"]
```

`.yaml` 파일을 불러와 `parse_yaml()` 함수로 `.yaml` 파일을 파싱한 다음, 딕셔너리를 사용하는 방식으로 argument의 각 영역에 접근할 수 있다.


## Reference
https://pynative.com/python-yaml/
