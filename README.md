# 💬 PUE: AI 심리 상담 챗봇
[![Pull Requests](https://img.shields.io/github/issues-pr/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)](https://github.com/PUE-AI-ChatBot/PUE-AI/pulls)
[![GitHub issues](https://img.shields.io/github/issues/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)](https://github.com/PUE-AI-ChatBot/PUE-AI/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)
>  **:two_hearts: AI 심리상담 챗봇 PUE**  
>
> Open Source <br>
> 프로젝트 시작 : 2022.08 <br> <br>
> 힘든 사람 누구에게나 ***친구가 되어줄*** <br>
> ***따듯한 위로를 전하는*** AI 챗봇 서비스 <br> 
>

## 🥇 Goals

코로나 블루로 우울감을 호소하는 사람이 많아지자 상담사 고용을 늘렸지만 여전히 응답률이 저조하였습니다. <br>
이를 위해 정보 제공용 챗봇을 뛰어넘어 가벼운 심리 상담이 가능한 챗봇을 만들고자 하였습니다. <br>
상담이 여려운 시간대에도 상담이 가능하며 챗봇 상담의 높은 접근성으로 기존 상담에 대한 인식을 개선하고자 개발하게 되었습니다. <br>

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


## 🔨 Environments
### Development
#### Language
<img src="https://img.shields.io/badge/python-3.9-blue?style=for-the-badge&logo=appveyor"/>

#### Library
<div>
  <img src="https://img.shields.io/badge/tensorflow-2.10.0-brightgreen?style=for-the-badge&logo=appveyor"/>
     <img src="https://img.shields.io/badge/transformers-4.21.3-yellow?style=for-the-badge&logo=appveyor"/>&nbsp
</div>

#### IDE
<div>
    <img src="https://img.shields.io/badge/VisualStudioCode-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white"/>
     <img src = "https://img.shields.io/badge/PyCharm-000000.svg?style=for-the-badge&logo=PyCharm&logoColor=white"/>&nbsp 
</div>

#### Package Management & GPU Server
<div>
    <img src="https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=for-the-badge&logo=Google-Colab&logoColor=white"/>&nbsp
</div>


### Communication
<div>
    <img src="https://img.shields.io/badge/ClickUp-7B68EE.svg?style=for-the-badge&logo=ClickUp&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>&nbsp
</div>



## ⚙️ Project Settings
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

## 📜 Feature
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

## 💻 Developers
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



## 📚 Documentations

### Open Source Github
- Klue-BERT : https://github.com/KLUE-benchmark/KLUE

- pytorch : https://github.com/pytorch

- tensorflow : https://github.com/tensorflow/tensorflow

### Wiki
- [Specification](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Specification)

- [Coding Convention](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Coding-Convention)

- [Workflow](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Workflow)

## 🔒 LICENSE
Preparing...



