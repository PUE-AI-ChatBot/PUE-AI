from submodules.emo_classifier import *
from submodules.ner_classifier import *
from submodules.gd_generator import *
from submodules.topic_classifier import *
# import numpy as np
# from transformers import BertTokenizer
# import pickle
# import os

# mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")
# mGC_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/tokenizer.pickle", 'rb'))
# mNER_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/letter_to_index.pickle", 'rb'))
# VOCAB_SIZE = mGC_tokenizer.vocab_size + 2

# emotion_labels = {"불만": 0, "중립": 1, "당혹": 2, "기쁨": 3, "걱정": 4, "질투": 5, "슬픔": 6, "죄책감": 7, "연민": 8}

# emotion_mapping_by_index = dict((value, key) for (key, value) in emotion_labels.items())
# index_mapping_by_NER = {'O': 0, 'B-LC': 1, 'I-LC': 2, 'B-QT': 3, 'I-QT': 4, 'B-OG': 5, 'I-OG': 6, 'B-DT': 7, 'I-DT': 8, 
#                         'B-PS': 9, 'I-PS': 10, 'B-TI': 11, 'I-TI': 12}
# NER_mapping_by_index = {0 : '0', 1 : 'B-LC', 2 : 'I-LC', 3 : 'B-QT', 4 : 'I-QT', 5 : 'B-OG', 6 : 'I-OG', 7 : 'B-DT', 
#                         8 : 'I-DT', 9 : 'B-PS', 10 : 'I-PS', 11 : 'B-TI', 12 : 'I-TI', 13 : 'UNK'}
# NER_labels = dict((value, key) for (key, value) in NER_mapping_by_index.items())
