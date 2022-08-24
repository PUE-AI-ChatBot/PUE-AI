## setup
from setup import setup_environ

setup_environ()

## device 관련 설정
import os

# CPU만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# GPU log 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from submodules.emo_classifier import *
from submodules.ner_classifier import *
from submodules.gc_transformer import *
from collections import OrderedDict

## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들

class AIModel:

    def __init__(self):
        self.get_converters()

    def get_converters(self):
        self._mTokenizer = mTokenizer
        self._mGC_tokenizer = mGC_tokenizer

    def model_loader(self):
        self.GC_model = load_general_corpus_model()
        self.NER_model = load_NER_model()
        self.EMO_model = load_Emo_model()

##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):

        Data = OrderedDict()

        GeneralAnswer = GC_predict(inputsentence, self.GC_model, self._mTokenizer)
        NEROut = ner_predict(self.NER_model,[inputsentence])
        EmoOut = emo_predict(self.EMO_model,[inputsentence])

        NER = {}
        for (word, tag) in NEROut:
            if tag != "O":
                NER[word] = tag

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["NER"] = NER
        Data["Emotion"] = EmoOut
        Data["Type"] = "General"
        Data["System_Corpus"] = GeneralAnswer

        return Data

DoDam = AIModel()

DoDam.model_loader()

UserName = "민채"

while True:
    sample = input("입력 : ")
    output = DoDam.run(UserName, sample)
    print("출력 : {}" .format(output))