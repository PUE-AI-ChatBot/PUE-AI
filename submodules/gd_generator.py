import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import os
import pickle

mGC_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/tokenizer.pickle", 'rb'))
mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")

def load_general_corpus_model():

    D_MODEL = 768
    NUM_LAYERS = 6
    NUM_HEADS = 24
    DFF = 2048
    DROPOUT = 0.1
    VOCAB_SIZE = 32000

    # print(VOCAB_SIZE)
    print("########Loading GD model!!!########")

    # new_model = GeneralCorpusBertModel("klue/bert-base", num_layers=NUM_LAYERS, d_model=D_MODEL, 
    #                                     dff=DFF, num_heads=NUM_HEADS, num_labels=VOCAB_SIZE)

    # new_model.build(input_shape=(1, 1))

    new_model = GeneralDialogBertModel("klue/bert-base", num_layers=NUM_LAYERS, d_model=D_MODEL, 
                                        dff=DFF, num_heads=NUM_HEADS, num_labels=VOCAB_SIZE, dropout=DROPOUT)

    new_model.load_weights(os.environ['CHATBOT_ROOT'] + "/resources/weights/GeneralDialog_weights/General_weights.h5")

    return new_model

def GeneralDialogBertModel(model_name, num_layers, d_model, dff, num_heads, num_labels, dropout):

    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="input_ids")
    attention_masks = tf.keras.Input(shape=(128,), dtype=tf.int64, name="attention_masks")
    token_type_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="token_type_ids")

    # decoder 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # bert 모델

    bertModel = TFBertModel.from_pretrained(model_name, from_pt=True)
    bertout = bertModel(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
    dropout1 = tf.keras.layers.Dropout(bertModel.config.hidden_dropout_prob)(bertout[0])
    context_vec = tf.keras.layers.Dense(d_model, name="dec_input")(dropout1)

    # decoder 모델
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(input_ids)
    dec_outputs = decoder(vocab_size=num_labels, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                            )(inputs=[dec_inputs, bertout[0], look_ahead_mask, dec_padding_mask])
    predictions = tf.keras.layers.Dense(units=num_labels, name="outputs")(dec_outputs)

    my_model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids, dec_inputs], 
                            outputs=predictions, name="GeneralDialogSequenceModel")

    return my_model


# class GeneralCorpusBertModel(tf.keras.Model):
#     def __init__(self, model_name, num_layers, d_model, dff, num_heads, num_labels):
#         super(GeneralCorpusBertModel, self).__init__()
        
#         ## bert encoder
#         self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
#         self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
#         self.context_vec = tf.keras.layers.Dense(d_model, name="dec_input")

#         ## decoder
#         self.look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')
#         self.dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')
#         self.decoder = decoder(vocab_size=num_labels, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=0.1,)
#         self.classifier = tf.keras.layers.Dense(num_labels,
#                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
#                                                 name='classifier')

#         # self.input_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="input_ids")
#         # self.attention_masks = tf.keras.Input(shape=(128,), dtype=tf.int64, name="attention_masks")
#         # self.token_type_ids = tf.keras.Input(shape=(128,), dtype=tf.int64, name="token_type_ids")

#         # self.dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

#         # self.bertModel = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
#         # self.bertout = self.bertModel(input_ids=self.input_ids, attention_mask=self.attention_masks, token_type_ids=self.token_type_ids)
#         # self.dropout1 = tf.keras.layers.Dropout(self.bertModel.config.hidden_dropout_prob)(self.bertout[0])
#         # self.context_vec = tf.keras.layers.Dense(d_model, name="dec_input")(self.dropout1)

#         # self.look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(self.dec_inputs)
#         # self.dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(self.input_ids)
#         # self.dec_outputs = decoder(vocab_size=VOCAB_SIZE, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=0.1,
#         #                         )(inputs=[self.dec_inputs, self.bertout[0], self.look_ahead_mask, self.dec_padding_mask])
#         # self.predictions = tf.keras.layers.Dense(units=VOCAB_SIZE, name="outputs")(self.dec_outputs)

#         # self.model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids, dec_inputs], outputs=predictions, name="GeneralDialogSequenceModel")

    
#     def make_attention_masks(self, enc_inputs):
#         attention_masks = []
#         for enc_input in enc_inputs:
#           count = 0
#           for index in enc_input:
#             if index != 0:
#               count += 1
#           attention_mask = [1] * count
#           attention_masks.append(attention_mask)

#         attention_masks = pad_sequences(attention_masks, padding='post', maxlen=128)
#         attention_masks = np.array(attention_masks, dtype=int)
        
#         return attention_masks

#     def make_token_type_ids(self, enc_inputs):
#         print(enc_inputs)
#         token_type_ids = [0] * 128
#         token_type_ids = np.array(token_type_ids, dtype=int)

#         return token_type_ids

#     def call(self, inputs):
#         (input_ids, attention_mask, token_type_ids), decoder_input = inputs

#         ## bert encoder
#         bertout = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         dropped = self.dropout(bertout[0], training=False)
#         encoder_out = self.context_vec(dropped)

#         ## decoder
#         look_ahead_mask = self.look_ahead_mask(decoder_input)
#         dec_padding_mask = self.dec_padding_mask(input_ids)
#         decoder_out = self.decoder([decoder_input, encoder_out, look_ahead_mask, dec_padding_mask])
#         prediction = self.classifier(decoder_out)

#         return prediction

def GC_predict(sentence, model, tokenizer):
    prediction = make_datasets_for_prediction(sentence, model, tokenizer)

    # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
    # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if (i < tokenizer.vocab_size) and (i != 2)])

    # print('Input: {}'.format(sentence))
    # print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

def make_datasets_for_prediction(sentence, model, tokenizer):

    SEP = [2]
    CLS = [3]
    
    tokenized = tokenizer.encode(sentence)

    # 입력 문장에 시작 토큰과 종료 토큰을 추가
    sentence = tf.expand_dims( tokenized + [0] * (128-len(tokenized)), axis=0 )
    position = tf.expand_dims( [1] * len(tokenized) + [0] * (128-len(tokenized)), axis=0 )
    segment = tf.expand_dims( [0] * 128, axis=0 )

    output = tf.expand_dims(SEP, 0)

    # print(sentence)
    # print(position)
    # print(segment)
    # print(output)

    # 디코더의 예측 시작
    for i in range(128):
        predictions = model(inputs=[sentence, position, segment, output], training=False)

        # 현재 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, CLS[0]):
            break

        # 현재 시점의 예측 단어를 output(출력)에 연결한다.
        # output은 for문의 다음 루프에서 디코더의 입력이 된다.
        output = tf.concat([output, predicted_id], axis=-1)
        # print(output)

    # 단어 예측이 모두 끝났다면 output을 리턴.
    return tf.squeeze(output, axis=0)

##Transformer임 일반대화를 위함임!
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs,*args, **kwargs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

##클래스로 만들고 불러오기


def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    # Q와 K의 곱. 어텐션 스코어 행렬.
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링
    # dk의 루트값으로 나눠준다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)

    # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model을 num_heads로 나눈 값. 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units=d_model)

      # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs,*args, **kwargs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]

# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")

    # 패딩 마스크(두번째 서브층)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
            'mask': look_ahead_mask # 룩어헤드 마스크
        })

    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
            'mask': padding_mask # 패딩 마스크
        })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

