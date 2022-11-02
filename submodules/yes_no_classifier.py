import re
import os
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


max_len = 30
def sentiment_predict(loaded_model, new_sentence):
    tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/Y_tokenizer.pickle", 'rb'))
    okt = Okt()
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측

    if score > 0.5 or new_sentence in ["그래"]:
        return "yes"
    else:
        return "no"

def load_yes_no_model():
    print("########Loading Yes No model!!!########")
    new_model = load_model(os.environ['CHATBOT_ROOT'] + "/resources/weights/Yes_no_weights/Yes_no_weights.h5")

    return new_model

