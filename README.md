# π¬ PUE: AI μ¬λ¦¬ μλ΄ μ±λ΄
[![Pull Requests](https://img.shields.io/github/issues-pr/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)](https://github.com/PUE-AI-ChatBot/PUE-AI/pulls)
[![GitHub issues](https://img.shields.io/github/issues/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)](https://github.com/PUE-AI-ChatBot/PUE-AI/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/PUE-AI-ChatBot/PUE-AI?style=for-the-badge)
>  **:two_hearts: AI μ¬λ¦¬μλ΄ μ±λ΄ PUE**  
>
> Open Source <br>
> νλ‘μ νΈ μμ : 2022.08 <br> <br>
> νλ  μ¬λ λκ΅¬μκ²λ ***μΉκ΅¬κ° λμ΄μ€*** <br>
> ***λ°λ―ν μλ‘λ₯Ό μ νλ*** AI μ±λ΄ μλΉμ€ <br> 
>
## π₯ Goals


μ½λ‘λ λΈλ£¨λ‘ μ°μΈκ°μ νΈμνλ μ¬λμ΄ λ§μμ§μ μλ΄μ¬ κ³ μ©μ λλ Έμ§λ§ μ¬μ ν μλ΅λ₯ μ΄ μ μ‘°νμμ΅λλ€. <br>
μ΄λ₯Ό μν΄ μ λ³΄ μ κ³΅μ© μ±λ΄μ λ°μ΄λμ΄ κ°λ²Όμ΄ μ¬λ¦¬ μλ΄μ΄ κ°λ₯ν μ±λ΄μ λ§λ€κ³ μ νμμ΅λλ€. <br>
μλ΄μ΄ μ¬λ €μ΄ μκ°λμλ μλ΄μ΄ κ°λ₯νλ©° μ±λ΄ μλ΄μ λμ μ κ·Όμ±μΌλ‘ κΈ°μ‘΄ μλ΄μ λν μΈμμ κ°μ νκ³ μ κ°λ°νκ² λμμ΅λλ€. <br>

## File structure
<div align="left">
    
    PUE-AI
    |
    ββexamples
    β  β  aimodel.py
    β  β  AIServer.py
    β  β  main.py
    β  β  test_Chatbot.py
    β  β  test_NER.py
    β  β
    β  ββtrainning
    β          Training_Chatbot_Transformer.py
    β          Training_EmoClass_KoBERT.py
    β          Training_NER_KoBERT.py
    β
    ββresources
    β  β  config.json
    β  β
    β  ββconverters
    β  β      letter_to_index.pickle
    β  β      NER_label_to_index.pickle
    β  β      tokenizer.pickle
    β  β
    β  ββtraining_data
    β  β      delete_before_use.txt
    β  β
    β  ββweights
    β      β  delete_before_use.txt
    β      β
    β      ββTopic_weights
    β      β      delete_before_use.txt
    β      β
    β      ββTransformer_weights
    β              delete_before_use.txt
    β
    ββsubmodules
       β  emo_classifier.py
       β  gd_generator.py
       β  subtopic_classifier.py
       β  topic_classifier.py
       ββ  __init__.py
    
</div>


## π¨ Environments
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



## βοΈ Project Settings
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
>    UserName = "λ―Όμ±"
>    while True:
>        sample = input("μλ ₯ : ")
>        output = DoDam.run(UserName, sample)
>        print("μΆλ ₯ : {}" .format(output))
```
## π Feature
> μ¬μ©μ μλ ₯ λνλ₯Ό λΆμ, λΆλ₯νμ¬ κ²°κ³Όμ λ°λΌ μΌλ°λνλ₯Ό μμ±νκ³  νλμ μλ£λ‘ λ§λ€μ΄ λλλ€. 
λ³΄λ€ μμΈν κΈ°λ₯ μ€λͺμ [**AI_Wiki_Specification**](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Specification)μ μ°Έκ³ ν΄μ£ΌμΈμ.
### 1. λν λΆμ
> μ¬μ  νμ΅λ AI λͺ¨λΈμ ν΅ν΄ μ¬μ©μ μλ ₯ λνμ λ΄κΈ΄ μ£Όμ μ κ°μ μ λΆμνμ¬ μ§μ λ λ μ΄λΈλ‘ λΆλ₯ν©λλ€.
- μλ ₯ λνλ₯Ό AI λͺ¨λΈ νλ ¨μ μν΄ tensor ννλ‘ λ§λλ λ°μ΄ν° μΈμ½λ©μ μνν©λλ€.
- TFBertModel layerμ classifier layerλ₯Ό μμ Model μ μ
- μΈμ½λ©ν λ°μ΄ν°λ₯Ό λͺ¨λΈμ μλ ₯ν©λλ€.
- λͺ¨λΈμ κ²°κ³Όκ°μΈ νλ₯  λ²‘ν° μ€ μ΅λ κ°μ κ°λ¦¬ν€λ μμλ₯Ό μΆμΆ ν, label dictionaryλ‘ label λ°νν©λλ€.
### 2. λν λΆλ₯
> λΆμ κ²°κ³Όλ‘ μ»μ μ£Όμ μ κ°μ μ λ°νμΌλ‘ λν νμμ λΆλ₯ν©λλ€.
- κ°μ  μ λ³΄κ° λΆμ μ΄κ³ , μ£Όμ  μ λ³΄κ° μλ΄ μλλ¦¬μ€ κ΄λ ¨ μ£Όμ μ΄λ©΄ μλ΄ λνλ‘ λΆλ₯ν©λλ€.
- κ°μ  μ λ³΄κ° μ€λ¦½ λ° κΈμ μ΄κ³ , μ£Όμ  μ λ³΄κ° μΌμ λν κ΄λ ¨ μ£Όμ μ΄λ©΄ μΌμ λνλ‘ λΆλ₯ν©λλ€.
### 3. λν μμ± λ° μλ£κ΅¬μ±
> μ¬μ  νμ΅λ AI λͺ¨λΈμ ν΅ν΄ μ¬μ©μ μλ ₯μ λμνλ μ μ ν λ΅λ³μ λ§λ€κ³  λν λ΄μ­μ νλμ μλ£λ‘ λ§λ€μ΄ λλλ€.
- μΆ©λΆν μΌμλνλ₯Ό νμ΅ν AI λͺ¨λΈμ μλ ₯ λνλ₯Ό tensor ννλ‘ λ£μ΅λλ€.
- Decoderμ Bertμ μΈμ½λ©λ μλ ₯κ³Ό, Bertμ μΆλ ₯μ μλ ₯νμ¬ attention mechanism λ° FFNN μ ν΅ν΄ λ΅λ³ tokenλ€μ μ°¨λ‘λ‘ λμΆν©λλ€.
- λͺ¨λ  λν μ λ³΄ λ° νμ, μΌμ λν λ΅λ³μ OrderedDict μλ£νμ μ μ₯νμ¬ μλ²μ λ°νν©λλ€.
## π» Developers
<div align="left">
    <table border="1">
        <th><a href="https://github.com/HeoYoon1">νμ€</a></th>
        <th><a href="https://github.com/pangthing">λ°κ΄λͺ</a></th>
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

## π Documentations

### Open Source Github
- Klue-BERT : https://github.com/KLUE-benchmark/KLUE
- pytorch : https://github.com/pytorch
- tensorflow : https://github.com/tensorflow/tensorflow

### Wiki
- [Specification](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Specification)
- [Coding Convention](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Coding-Convention)
- [Workflow](https://github.com/PUE-AI-ChatBot/PUE-AI/wiki/Workflow)

## π LICENSE
This Project is [MIT licensed](https://github.com/dnd-side-project/dnd-7th-3-frontend/blob/main/LICENSE).




