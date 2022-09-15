# PUE-AI
[![Pull Requests][pr-shield]][pr-url]
> **🏃 땅따먹기 기반 운동 장려 앱 NEMODU**  
>
> DND 7기 <br>
> 프로젝트 기간 : 2022.07 ~ <br> <br>
> **Healthy Pleasure,** 즐거운 건강관리를 위해 <br>
> 나의 일상 속 움직임을 기록하고, 친구와 재미있게 운동할 수 있는 앱 서비스
>

## Goals
As we go through the COVID-19, most of the 'meeting places' are rapidly moving to various virtual conference spaces started from Zoom. As such, there are many side effects, A typical example is Zoom Fatigue, which causes a lot of fatigue in virtual conferences than usual conversations. We think the main causes of that are


## File structure
<div align="left">
    
    PUE-AI
    |
    ├─examples
    │  │  aimodel.py
    │  │  AIServer.py
    │  │  main.py
    │  │  test_Chatbot.py
    │  │  test_NER.py
    │  │
    │  └─trainning
    │          Training_Chatbot_Transformer.py
    │          Training_EmoClass_KoBERT.py
    │          Training_NER_KoBERT.py
    │
    ├─resources
    │  │  config.json
    │  │
    │  ├─converters
    │  │      letter_to_index.pickle
    │  │      NER_label_to_index.pickle
    │  │      tokenizer.pickle
    │  │
    │  ├─training_data
    │  │      delete_before_use.txt
    │  │
    │  └─weights
    │      │  delete_before_use.txt
    │      │
    │      ├─Topic_weights
    │      │      delete_before_use.txt
    │      │
    │      └─Transformer_weights
    │              delete_before_use.txt
    │
    └─submodules
       │  emo_classifier.py
       │  gd_generator.py
       │  subtopic_classifier.py
       │  topic_classifier.py
       └─  __init__.py
    
</div>


## Environments
### Development
#### Language
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>


#### IDE
<div>
    <img src="https://img.shields.io/badge/VisualStudioCode-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white"/>
     <img src = "https://img.shields.io/badge/PyCharm-000000.svg?style=for-the-badge&logo=PyCharm&logoColor=white"/>&nbsp 
</div>

#### Library
<div>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
     <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>&nbsp
</div>

#### Package Management
<div>
    <img src="https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white"/>&nbsp
</div>

#### GPU Server
<div>
    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=for-the-badge&logo=Google-Colab&logoColor=white"/>&nbsp
</div>

### Communication
<div>
    <img src="https://img.shields.io/badge/ClickUp-7B68EE.svg?style=for-the-badge&logo=ClickUp&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>&nbsp
</div>



## Project Settings
#### Install library dependencies

```bash
> pip install -r requirements.txt
```

#### Test AI Code

```bash
> import aimodel
> import tensorflow as tf
> from __init__ import setup_environ, download_weights
> setup_environ()
> download_weights()

> with tf.device("/device:CPU:0"):
>    DoDam = AIModel()
>    UserName = "민채"
>    while True:
>        sample = input("입력 : ")
>        output = DoDam.run(UserName, sample)
>        print("출력 : {}" .format(output))
```

## Feature
> 사용자 입력 대화를 분석, 분류하여 결과에 따라 일반대화를 생성하고 하나의 자료로 만들어 냅니다. 

보다 자세한 기능 설명은 [**AI_Wiki_Specification**](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Specification)을 참고해주세요.

### 1. 대화 분석
> 사전 학습된 AI 모델을 통해 사용자 입력 대화에 담긴 주제와 감정을 분석하여 지정된 레이블로 분류합니다.
- 입력 대화를 AI 모델 훈련을 위해 tensor 형태로 만드는 데이터 인코딩을 수행합니다.
- TFBertModel layer와 classifier layer를 쌓아 Model 제작
- 인코딩한 데이터를 모델에 입력합니다.
- 모델의 결과값인 확률 벡터 중 최대 값을 가리키는 요소를 추출 후, label dictionary로 label 반환합니다.

### 2. 대화 분류
> 분석 결과로 얻은 주제와 감정을 바탕으로 대화 타입을 분류합니다.
- 감정 정보가 부정이고, 주제 정보가 상담 시나리오 관련 주제이면 상담 대화로 분류합니다.
- 감정 정보가 중립 및 긍정이고, 주제 정보가 일상 대화 관련 주제이면 일상 대화로 분류합니다.


### 3. 대화 생성 및 자료구성
> 사전 학습된 AI 모델을 통해 사용자 입력에 대응하는 적절한 답변을 만들고 대화 내역을 하나의 자료로 만들어 냅니다.
- 충분히 일상대화를 학습한 AI 모델에 입력 대화를 tensor 형태로 넣습니다.
- Decoder에 Bert의 인코딩된 입력과, Bert의 출력을 입력하여 attention mechanism 및 FFNN 을 통해 답변 token들을 차례로 도출합니다.
- 모든 대화 정보 및 타입, 일상 대화 답변을 OrderedDict 자료형에 저장하여 서버에 반환합니다.

## BERT

## Developers
<div align="left">
    <table border="1">
        <th><a href="https://github.com/HeoYoon1">허윤</a></th>
        <th><a href="https://github.com/pangthing">박광명</a></th>
        <tr>
            <td>
                <img src="https://github.com/HeoYoon1.png" width='80' />
            </td>
            <td>
                <img src="https://github.com/pangthing.png" width='80' />
            </td>
        </tr>
    </table>
</div>



## DOCUMENTARY

### Open source Github
Klue-BERT : https://github.com/KLUE-benchmark/KLUE</br>
pytorch : https://github.com/pytorch</br>
tensorflow : https://github.com/tensorflow/tensorflow

You can also see the [**AI_Wiki**](https://github.com/PUE-AI-ChatBot/PUE-AI.wiki.git).

## LICENSE
Preparing...


[pr-shield]: https://img.shields.io/github/issues-pr/Study-CodingTest/Study?style=for-the-badge
[pr-url]: https://github.com/PUE-AI-ChatBot/PUE-FE
