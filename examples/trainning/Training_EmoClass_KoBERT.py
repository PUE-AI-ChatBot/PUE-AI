import numpy as np
from sklearn.model_selection import train_test_split
from bertmodels import AIModel
import pandas as pd
import os


train_ner_df = pd.read_csv(os.environ['CHATBOT_ROOT']+"/resources/training_data/감성대화말뭉치_최종_전반.csv", encoding='CP949')

X_train = train_ner_df['문장']
y_train = train_ner_df['감정']

y_train_encoded = y_train

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_encoded, test_size=0.1, random_state=0)

y_train = np.array([AIModel.emotion_labels[content] for content in y_train])

# tokenizer load

max_len = 128
main_model = AIModel()

#방식변경
train_X = main_model.EMO_make_datasets(X_train ,max_len=max_len)
test_X = main_model.EMO_make_datasets(X_test, max_len=max_len)

model = SequenceClassification("klue/bert-base", num_labels=len(AIModel.emotion_labels))
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

history = model.fit(
    train_X, y_train, epochs=2, batch_size=2, validation_split=0.1
)