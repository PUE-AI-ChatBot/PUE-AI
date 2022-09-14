## setup
## device 관련 설정
import os
try:
    from submodules import *
    from collections import OrderedDict
except :
    from .submodules import *
    from collections import OrderedDict

## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들

class AIModel:

    def __init__(self):
        self.get_converters()
        self.dialog_buffer = []
        self.model_loader()

    def get_converters(self):
        self._mTokenizer = mTokenizer
        self._mGC_tokenizer = mGC_tokenizer
        self._topic_converter = Topics_mapping_by_index

    def model_loader(self):
        self.GC_model = load_general_corpus_model()
        self.NER_model = load_NER_model()
        self.EMO_model = load_Emo_model()
        self.Topic_model = load_Topic_model()
        self.ST_model = load_Sub_Topic_model()

    def manage_dailogbuffer(self):
        if len(self.dialog_buffer) < 3:
            return False

        elif len(self.dialog_buffer) == 3:
            return True

        elif len(self.dialog_buffer) == 5:
            while len(self.dialog_buffer) != 3:
                self.dialog_buffer.pop(0)
            return True

        # elif len(self.dialog_buffer) > 4:
        #     while len(self.dialog_buffer) != 1:
        #         self.dialog_buffer.pop(0)
        #     return False

    def get_results(self, inputsentence):
        dialogs = ""
        for dialog in self.dialog_buffer:
            dialogs += dialog

        GeneralAnswer = GC_predict(inputsentence, self.GC_model, self._mTokenizer)
        NEROut = ner_predict(self.NER_model,[inputsentence])
        EmoOut = emo_predict(self.EMO_model,[inputsentence])

        NER = {}
        for (word, tag) in NEROut:
            NER[word] = tag

        if self.manage_dailogbuffer() is True:
            (initial_topic_output, initial_label_prob, topic_percentage), topic_prob_vec = Topic_predict(self.Topic_model, [dialogs], self._mTokenizer)
            if EmoOut in ('불만', '당혹', '걱정', '질투', '슬픔', '죄책감', '연민') and (float(initial_label_prob*100) < 99.0) :
                topic_index = np.argmax(topic_prob_vec[0][:7])
                altered_topic_output = self._topic_converter[topic_index]
                Topic = altered_topic_output
                Topic = Sub_Topic_predict(self.ST_model, inputsentence, Topic)

            else:
                altered_topic_output = 'None'
                Topic = initial_topic_output
                Topic = Sub_Topic_predict(self.ST_model, inputsentence, Topic)
        else:
            Topic = "None"

        if EmoOut in ('중립', '기쁨'):
            DialogType = "General"
        # elif EmoOut in ('불만', '당혹', '걱정', '질투', '슬픔', '죄책감', '연민'):
        #     DialogType = "Scenario"
        else :
            DialogType = "Scenario"

        self.dialog_buffer.append(GeneralAnswer)
        return GeneralAnswer, NER, EmoOut, Topic, DialogType

##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):

        Data = OrderedDict()

        self.dialog_buffer.append(inputsentence)

        GeneralAnswer, Name_Entity, Emotion, Topic, Type = self.get_results(inputsentence)

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["NER"] = Name_Entity
        Data["Emotion"] = Emotion
        Data["Topic"] = Topic
        Data["Type"] = Type
        Data["System_Corpus"] = GeneralAnswer

        return Data
if __name__ == "__main__" :
    from setup import setup_environ
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    setup_environ()
    with tf.device("/device:CPU:0"):
        DoDam = AIModel()
        UserName = "민채"
        while True:
            sample = input("입력 : ")
            output = DoDam.run(UserName, sample)
            print("출력 : {}" .format(output))