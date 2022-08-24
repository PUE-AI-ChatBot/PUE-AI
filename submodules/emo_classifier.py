# from submodules import emotion_mapping_by_index, mTokenizer, emotion_labels, SequenceClassification, np, os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import os
import numpy as np

mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")
emotion_labels = {"불만": 0, "중립": 1, "당혹": 2, "기쁨": 3, "걱정": 4, "질투": 5, "슬픔": 6, "죄책감": 7, "연민": 8}
emotion_mapping_by_index = dict((value, key) for (key, value) in emotion_labels.items())

def load_Emo_model():
    print("########Loading EMO model!!!########")
    new_model = EmotionClassificationBertModel("klue/bert-base", num_labels=len(emotion_labels))
    # new_model.build(input_shape=[((None, 128)), ((None, 128)), ((None, 128))])
    new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/Emo_weights/Emo_weights")

    return new_model

class EmotionClassificationBertModel(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(EmotionClassificationBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.drop = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                                activation='softmax',
                                                name='classifier')

    def call(self, inputs, training=None, mask=None):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #Bert의 값이 encoder (pooler)값이 반환됨 64*29
        output = outputs[1]
        dropped = self.drop(output, training=False)
        prediction = self.classifier(dropped)

        return prediction

def emo_predict(model, sentences, max_len=128):

    # 예측에 필요한 데이터폼 생성
    input = emo_make_datasets(sentences, max_len)
    raw_output = model.predict(input)
    output = np.argmax(raw_output, axis=-1)

    prediction = emotion_mapping_by_index[output[0]]

    return prediction

def emo_make_datasets(sentences, max_len=128):
    input_ids, attention_masks, token_type_ids = [], [], []

    for sentence in sentences:
        # 문장별로 정수 인코딩 진행
        input_id = mTokenizer.encode(sentence, max_length=max_len)
        # encode한 정수들의 수만큼 1로 할당
        attention_mask = [1] * len(input_id)
        # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
        token_type_id = [0] * max_len

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)

    # 패딩
    input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
    attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)