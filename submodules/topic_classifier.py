import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import shape_list, BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
import tensorflow as tf
import json
import glob
import random

index_mapping_by_Topics = {"가족":1, "회사/학교/진로":2, "군대":3, "연애/결혼":4, "건강":5, "반려동물":6, "이슈":7, "방송/미디어 컨텐츠":8, "취미":9, "계절/날씨":10, "식음료":11}
Topics_mapping_by_index = {value:key for key, value in index_mapping_by_Topics.items()}


def load_Topic_model():
    print("########Loading Topic model!!!########")
    tag_size = len(index_mapping_by_Topics)
    new_model = TopicBertModel("klue/bert-base", num_labels=tag_size+1)
    new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/Topic_weights/Topic_weights.h5")

    return new_model

def TopicBertModel(model_name, num_labels):

    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="input_ids")
    attention_masks = tf.keras.Input(shape=(128,), dtype=tf.int64, name="attention_masks")
    token_type_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="token_type_ids")

    bertModel = TFBertModel.from_pretrained(model_name, from_pt=True)
    bertout = bertModel(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
    dropout = tf.keras.layers.Dropout(bertModel.config.hidden_dropout_prob)(bertout[1])
    classifier = tf.keras.layers.Dense(num_labels,
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                        activation='softmax',
                                        name='outputs')(dropout)

    my_model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=classifier, name="TopicClassifier")

    return my_model

def Topic_predict(model, input, tokenizer, converter=Topics_mapping_by_index, max_len=128):
    input_ids_pred, attention_masks_pred, token_type_ids_pred = topic_make_datasets_for_prediction(input, max_len, tokenizer)
    y_predicted = model(inputs=[input_ids_pred, attention_masks_pred, token_type_ids_pred], training=False)
    label_predicted = np.argmax(y_predicted, axis = -1)

    result = []
    for prob_vector, label_index in zip(y_predicted, label_predicted):
        result = (converter[label_index], [round(value*100, 1) for value in np.array(prob_vector)][1:])

    return result, y_predicted

def topic_make_datasets_for_prediction(sentences, max_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []
    tokenizer.pad_token

    for sentence in sentences:
    # 문장별로 정수 인코딩 진행
        input_id = tokenizer.encode(sentence, max_length=max_len)
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

    return input_ids, attention_masks, token_type_ids

def make_datasets(sentences, max_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []
    tokenizer.pad_token

    for sentence, _ in sentences:
        # 문장별로 정수 인코딩 진행
        input_id = tokenizer.encode(sentence, max_length=max_len)
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

    return input_ids, attention_masks, token_type_ids