"""
last modified : 220823
modified by : Heo_Yoon
contents : (new)theme_classifier
new_dependencies(ex. lib) : (new_folder)resources/the_weights
                            , (new lib) eunjeon, gensim
"""

from submodules import LdaModel, datapath, corpora, os, np, theme_mapping_by_index
from eunjeon import Mecab



def the_predict(model, sentences):
    # 문장을 형태소 단위로 분리
    m = Mecab()
    key = m.pos(sentences)
    print(key)
    key_list = []
    #품사가 명사, 동사 이며 1글자 이상인 단어만 취함
    for L, pos in key:
        if pos.startswith("N") or pos.startswith("V"):
            if len(L) > 1:
                key_list.append(L)
    print(key_list)
    #vocab과 비교
    bow = model.id2word.doc2bow(key_list)
    #일치 단어 없을 시 무주제
    if not bow:
        return "무주제"
    #일치 단어 있을 시 model 입력
    a = model.get_document_topics(bow)
    b = []
    for i in a:
        b.append(i[1])
    #가장 높게 예측된 주제 index
    topic_index = a[np.argmax(b)][0]
    #예측값
    prediction = theme_mapping_by_index[topic_index]

    return prediction


def load_THE_model():
    print("########Loading THE model!!!########")
    file_dir = datapath(os.environ['CHATBOT_ROOT']+"/resources/weights/THE_weights/The_model(LDA)")
    new_model = LdaModel.load(file_dir)

    return new_model
