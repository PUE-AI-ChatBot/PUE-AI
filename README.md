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


## File structure
<div align="left">
    ```
    
    PUE-AI
    │
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
    
    ```
</div>


## Environments
<table>
<tr>
 <td align="center">언어</td>
 <td>
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
 </td>
</tr>
<tr>
 <td align="center">IDE</td>
 <td>
    <img src="https://img.shields.io/badge/VisualStudioCode-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white"/>
     <img src = "https://img.shields.io/badge/PyCharm-000000.svg?style=for-the-badge&logo=PyCharm&logoColor=white"/>&nbsp </td>
    
</tr>
<tr>
 <td align="center">라이브러리</td>
 <td>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
     <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>&nbsp
</tr>
<tr>
 <td align="center">패키지관리</td>
 <td>
    <img src="https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white"/>&nbsp
  </td>
</tr>
<tr>
 <td align="center">GPU 서버</td>
 <td>
    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=for-the-badge&logo=Google-Colab&logoColor=white"/>&nbsp
  </td>
</tr>
<tr>
 <td align="center">협업툴</td>
 <td>
    <img src="https://img.shields.io/badge/ClickUp-7B68EE.svg?style=for-the-badge&logo=ClickUp&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white"/>&nbsp
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>&nbsp
 </td>
</tr>
</table>


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

## REFERENCE
You can also see the [**AI_Wiki**](https://github.com/PUE-AI-ChatBot/PUE-AI.wiki.git).

## LICENSE



[pr-shield]: https://img.shields.io/github/issues-pr/Study-CodingTest/Study?style=for-the-badge
[pr-url]: https://github.com/PUE-AI-ChatBot/PUE-FE
