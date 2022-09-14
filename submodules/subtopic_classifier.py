import os
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from eunjeon import Mecab

m_index = {0:"연예및결혼_금전", 1:"연예및결혼_성격", 2:"연예및결혼_기타"}#index for married
f_index = {0:"가족_부모", 1:"가족_형제자매", 2:"가족_소외감"}#index for famliy
a_index = {0:"군대_간부", 1:"군대_부조리", 2:"군대_훈련"}#index for army
c_index = {0:"회사_아르바이트", 1:"회사_직장", 2:"회사_진로"}#index for company

#step 1 : 소분류 LDA 모델을 불러옵니다.
def load_Sub_Topic_model():
    print("########Loading Sub Topic model!!!########")
    model = []
    model.append(LdaModel.load(os.environ['CHATBOT_ROOT']+"/resources/weights/Subtopic_model/" + "LDA_model_연애_결혼"))
    model.append(LdaModel.load(os.environ['CHATBOT_ROOT']+"/resources/weights/Subtopic_model/" + "LDA_model_군대"))
    model.append(LdaModel.load(os.environ['CHATBOT_ROOT']+"/resources/weights/Subtopic_model/" + "LDA_model_가족"))
    model.append(LdaModel.load(os.environ['CHATBOT_ROOT']+"/resources/weights/Subtopic_model/" + "LDA_model_회사_아르바이트"))

    return model

def Sub_Topic_predict(model, sentences, topic_name):
    m = Mecab()
    bow = []
    user_M = m.pos(sentences)
    for word, pos in user_M:
        if pos.startswith("N") or pos.startswith("V"):
            bow.append(word)
    if topic_name == "연예/결혼":
        id2word = model[0].id2word
        bow1 = id2word.doc2bow(bow)
        topic_distribution = model[0].get_document_topics(bow1)
        temp1 = []
        temp2 = []
        for i, j in topic_distribution:
            temp1.append(i)
            temp2.append(j)
        topic_num = temp1[np.argmax(temp2)]
        topic_name = m_index[topic_num]
        return topic_name
    elif topic_name == "회사/학교/진로":
        id2word = model[3].id2word
        bow1 = id2word.doc2bow(bow)
        topic_distribution = model[3].get_document_topics(bow1)
        temp1 = []
        temp2 = []
        for i, j in topic_distribution:
            temp1.append(i)
            temp2.append(j)
        topic_num = temp1[np.argmax(temp2)]
        topic_name = c_index[topic_num]
        return topic_name
    elif topic_name == "군대":
        id2word = model[1].id2word
        bow1 = id2word.doc2bow(bow)
        topic_distribution = model[1].get_document_topics(bow1)
        temp1 = []
        temp2 = []
        for i, j in topic_distribution:
            temp1.append(i)
            temp2.append(j)
        topic_num = temp1[np.argmax(temp2)]
        topic_name = a_index[topic_num]
        return topic_name
    elif topic_name == "가족":
        id2word = model[2].id2word
        bow1 = id2word.doc2bow(bow)
        topic_distribution = model[2].get_document_topics(bow1)
        temp1 = []
        temp2 = []
        for i, j in topic_distribution:
            temp1.append(i)
            temp2.append(j)
        topic_num = temp1[np.argmax(temp2)]
        topic_name = f_index[topic_num]
        return topic_name
    else:
        return topic_name
