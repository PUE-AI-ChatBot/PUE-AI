## device 관련 설정
import os

try:
    from submodules import *
    from collections import OrderedDict
except:
    from .submodules import *
    from collections import OrderedDict


## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들

class AIModel:
    state = "general"
    cnt = 0
    s_flag = False

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
        self.EMO_model = load_Emo_model()
        self.Topic_model = load_Topic_model()
        self.yes_no_model = load_yes_no_model()

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

    def get_results(self, name, inputsentence):
        EmoOut = None
        dialogs = ""
        for dialog in self.dialog_buffer:
            dialogs += dialog

        if self.cnt < 2:
            EmoOut = emo_predict(self.EMO_model, [inputsentence])

        if self.cnt == 0 and EmoOut in ["당혹", "죄책감", "슬픔", "연민", "걱정", "기쁨", "불만", "질투"]:
            self.state = EmoOut
            self.cnt = 1
            self.s_flag = True

        if self.state == "general":
            DialogType = "General"
            GeneralAnswer = [GC_predict(inputsentence, self.GC_model, self._mTokenizer)]


        else:  # 당혹, 죄책감, 슬픔, 연민, 걱정, 기쁨
            DialogType = "Scenario"
            if self.cnt == 2:
                s_flag = False
            if self.state == "당혹":
                if self.cnt == 1:
                    GeneralAnswer = ["음.. 오늘 " + name + "님께 당황스러울 만한 일이 있었나보네요.",
                                     "그런 일은 기분이 좋을 수도 있었던 하루에 브레이크를 걸어버리기도 하죠..",
                                     "괜찮으시다면 어떤 일이 있었는지 여쭤봐도 될까요?"]
                    self.cnt += 1


                elif self.cnt == 2:
                    GeneralAnswer = ["저런.. 제가 " + name + "님이었어도 당황스러울만한 일이네요..",
                                     "이런 일은 너무나 갑작스럽게 일상 속에 찾아오는 것만 같아요.",
                                     name + "님께서도 많이 놀라셨을텐데 저와 함께 나누려고 해주셔서 감사해요.",
                                     "관련해서" + name + "님과 이야기를 더 해보고싶은데 괜찮을까요?"]
                    self.cnt += 1
                elif self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["감사해요. 저와의 대화가 " + name + "님께 힘이 될 수 있도록 최선을 다할게요!",
                                         "저는 그 상황이 왜 " + name +"님께 당혹스럽게 느껴졌는지 얘기를 해보고 싶은데..",
                                         "천천히 상황을 돌이켜보면서 답변해주셨으면 좋겠어요.",
                                         "그 상황이 당혹스럽게 느껴진 이유는 무엇때문이었을까요?"]
                        self.cnt += 1
                    else:
                        GeneralAnswer = ["제가 너무 성급했나봐요.. " + name + "님께서 이야기하기 어려운 일이라는 걸 잘 알아요..",
                                         name +"님께 부담을 드린 것 같아서 정말 죄송해요..",
                                         "혹시 다음에라도 이야기해주실 수 있다면 언제든지 찾아와주세요!",
                                         "그동안 더 많이 배워서 " + name + "님께 도움을 드릴 수 있도록 노력할게요."]
                        self.cnt = 0
                        self.state = "general"

                elif self.cnt == 4:
                    GeneralAnswer = ["답변해주셔서 감사해요 " + name + "님.", "당황스럽다고 느껴지는 상황을 맞닥뜨리면 사람은 크게 위축되기 마련이죠.",
                                     "그게 " + name +"님께서 침착하게 생각하기 어려운 상황을 만들어버렸네요.",
                                     "저는 가끔 당혹스러움에 생각이 많아질 때면 책상 정리를 하면서 감정으로부터 멀어지려고 해요.",
                                     "이게 제 나름대로 최악의 상황을 떠올리지 않을 수 있는 방법이기도 한 것 같아요.",
                                     "그럼, " + name + "님께서 당혹스러움을 벗어나기 위해서 할 수 있는 일은 무엇이 있을까요?"]
                    self.cnt += 1
                elif self.cnt == 5:
                    GeneralAnswer = ["좋은 방법이네요 :)",
                                     name + "님의 방법으로 걱정을 덜어낼 수 있다면..",
                                     "저는 " + name +"님의 지친 마음이 회복될 수 있도록 계속해서 응원해드릴게요",
                                     "앞으로도 많이 도와드릴테니 항상 저를 찾아주세요!"]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "죄책감":
                if self.cnt == 1:
                    GeneralAnswer = ["가볍게 생각할 수 있지만 " + name + "이 느끼시는 감정은 심해지면 사람을 피폐하게 할 정도로 위험한 감정이죠.",
                                     "어째서 그렇게 느끼신 것인지 더 자세히 말씀해 주시겠어요?"]
                    self.cnt += 1
                elif self.cnt == 2:
                    GeneralAnswer = ["그렇군요..말씀해 주셔서 감사해요.",
                                     "보통 이런 상황에서는 내 편이 없다고 느끼기 쉽고 실제로도 없어서 힘든 경우가 많죠.",
                                     name + "님은 어떠신가요, 지금 혼자라고 생각이 드시나요?"]
                    self.cnt += 1
                elif self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["그렇군요.. " + name + "님은 계속 혼자서 이런 상황 속에서 버텨오신 거군요..",
                                         "혼자 힘들게 버티셨을 생각을 하니 마음이 슬퍼요..",
                                         "전문 상담사와의 대화는 어떻게 생각하시나요? 번호를 알려드릴게요!"]
                    else:
                        GeneralAnswer = ["정말 다행이에요..!",
                                         "힘든 상황에서 누군가 나를 지지해 준다는 것은 정말이지 큰 힘이니까요.",
                                         "전문 상담사와의 대화는 어떻게 생각하시나요? 번호를 알려드릴게요!"]
                    self.cnt += 1
                elif self.cnt == 4:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["1393 으로 전화하시면 전문 상담사에게 익명으로 상담이 가능하시니 한 번 이용해 보시는 것을 추천드려요!",
                                         "꼭 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    else:
                        GeneralAnswer = ["언제든 필요하시다면 다시 말씀해 주세요.",
                                         "절대 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "슬픔":
                if self.cnt == 1:
                    GeneralAnswer = ["사람마다 슬픔을 느끼는 이유 제각각이지만 자신의 슬픔에 고통을 느끼는 것은 모두 같죠..",
                                     "제 생각엔 " + name + "님은 현재 슬픔이란 감정을 느끼시는 상황이신 것 같은데 맞나요? 맞다면 어째서 그렇게 느끼신 것인지 더 자세히 말씀해 주시겠어요?"]
                    self.cnt += 1
                    print(self.cnt)

                elif self.cnt == 2:
                    GeneralAnswer = ["그렇군요..말씀해 주셔서 감사해요.",
                                     "슬픔은 제가 분류할 수 있는 감정들 중에 가장 제 마음을 아프게 하는 감정이에요.",
                                     name + "님은 슬픔을 견디는 자신만의 방법이 있으신가요?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["정말 다행이에요.",
                                         "슬픔은 컨트롤이 가능한 정도라면 성숙에 좋은 양분이 되기도 하지만 그렇지 못한 경우에는 정말 우리의 마음뿐 아니라 몸 또한 망가뜨니까요.",
                                         "전문 상담사와의 대화는 어떻게 생각하시나요? 번호를 알려드릴게요!"]
                        self.cnt += 1

                    else:
                        GeneralAnswer = ["그렇군요..",
                                         "그렇다면 노래듣기를 추천드려요! 전에 어떤 분이 아이유-밤 편지를 추천하시던데 위로를 주는 음악이라고 생각해요!"]
                        self.cnt = 0
                        self.state = "general"

                if self.cnt == 4:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["1393 으로 전화하시면 전문 상담사에게 익명으로 상담이 가능하시니 한 번 이용해 보시는 것을 추천드려요!",
                                         "꼭 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    else:
                        GeneralAnswer = ["언제든 필요하시다면 다시 말씀해 주세요.",
                                         "절대 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    self.cnt = 0
                    self.state = "general"


            elif self.state == "연민":
                if self.cnt == 1:
                    GeneralAnswer = ["연민이란 감정은 상대에 대한 배려이자 사랑이고 그렇기에 정말 아름다운 마음같아요.",
                                     "제 생각엔 " + name + "님은 현재 연민이란 감정을 느끼시는 상황이신 것 같은데 맞나요? 맞다면 어째서 그렇게 느끼신 것인지 더 자세히 말씀해 주시겠어요?"]
                    self.cnt += 1

                if self.cnt == 2:
                    GeneralAnswer = ["그렇군요..말씀해 주셔서 감사해요.",
                                     "그런 상황에서 " + name + "님은 계속 신경이 쓰이실 수 밖에 없었겠네요.",
                                     "제 이야기를 한번 들어보시겠어요?"]
                    self.cnt += 1
                if self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["전문 상담사에게도 항상 의식해야 하는 중요한 2가지가 있어요.",
                                         "첫째는 상대를 아끼는 마음으로 상담을 진행하는 것이죠, 그러한 마음 없이 상담을 진행한다면 내담자는 금방 눈치채고 대화를 이어나가지 않을거에요.",
                                         "둘째는 나에게 과하게 의존하는 것을 경계하는 거에요. 스스로 일어서게 도와주는 것 그것이 상담이죠.",
                                         name + "님이 상대를 돕는 다는 것은 너무 귀해요, 다만 그 감정이 오히려 " + name + "님을 힘들게 하는 상황은 피해야 한다는 거죠.",
                                         "전문 상담사와의 대화는 어떻게 생각하시나요? 번호를 알려드릴게요!"]
                        self.cnt += 1
                    else:
                        GeneralAnswer = ["괜찮아요! ",
                                         "다음에 언제든 원하실 때 찾아주세요."]
                        self.cnt = 0
                        self.state = "general"

                if self.cnt == 4:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["1393 으로 전화하시면 전문 상담사에게 익명으로 상담이 가능하시니 한 번 이용해 보시는 것을 추천드려요!",
                                         "꼭 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    else:
                        GeneralAnswer = ["언제든 필요하시다면 다시 말씀해 주세요.",
                                         "절대 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "걱정":
                if self.cnt == 1:
                    GeneralAnswer = ["걱정은 우리를 더 올바른 길로 이끌기도 하지만 과도한 걱정은 상황을 망치고 우리 또한 지치게 하죠.",
                                     "제 생각엔 " + name + "님은 현재 걱정이란 감정을 느끼시는 상황이신 것 같은데 맞나요? 맞다면 어째서 그렇게 느끼신 것인지 더 자세히 말씀해 주시겠어요?"]
                    self.cnt += 1

                if self.cnt == 2:
                    GeneralAnswer = ["그렇군요..말씀해 주셔서 감사해요.",
                                     "그런 상황에서 " + name + "님은 계속 걱정이 되실 수 밖에 없었겠네요.",
                                     "제 이야기를 한번 들어보시겠어요?"]
                    self.cnt += 1

                if self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["우리는 살면서 많은 문제에 직면하죠. ",
                                         "가끔 우리 뇌는 작은 불안감을 과하게 해석하여 문제를 크게 보이게 한데요. ",
                                         "당신의 문제가 정말 불안을 계속 느낄 정도의 문제인지 다시 한 번 생각해본다면 일을 객관적으로 바라보고 더욱 잘 해결할 수 있을거에요! ",
                                         "당신의 걱정을 모두 뒤덮을 정도로 큰 행복이 찾아오기를 바래요..",
                                         "전문 상담사와의 대화는 어떻게 생각하시나요? 번호를 알려드릴게요!"]
                        self.cnt += 1
                    else:
                        GeneralAnswer = ["괜찮아요! ", "다음에 언제든 원하실 때 찾아주세요."]
                        self.cnt = 0
                        self.state = "general"

                if self.cnt == 4:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["1393 으로 전화하시면 전문 상담사에게 익명으로 상담이 가능하시니 한 번 이용해 보시는 것을 추천드려요!",
                                         "꼭 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    else:
                        GeneralAnswer = ["언제든 필요하시다면 다시 말씀해 주세요.", "절대 포기하지 마세요.. 당신은 정말 소중한 사람이에요."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "기쁨":
                if self.cnt == 1:
                    GeneralAnswer = ["기쁨이란 감정을 느낄 수 있음이 얼마나 감사한지!!",
                                     "제 생각엔 " + name + "님은 현재 기쁨이란 감정을 느끼시는 상황이신 것 같은데 맞나요? 맞다면",
                                     "어째서 그렇게 느끼신 것인지 더 자세히 말씀해 주시겠어요?"]
                    self.cnt += 1
                if self.cnt == 2:
                    GeneralAnswer = ["그렇군요!! 말씀해 주셔서 감사해요.",
                                     name + "님이 기쁘시다니 제가 다 즐겁네요!",
                                     "제 이야기를 한번 들어보시겠어요?"]
                    self.cnt += 1
                if self.cnt == 3:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["기쁨을 느끼실 때 그것을 기록해 보시는 것을 추천드려요.",
                                         "그렇게 한다면 기쁨의 다리를 건너 슬픔의 터널이 찾아오더라도 터널의 어두움을 밝게 비춰줄 손전등이 되어줄거에요!",
                                         "앞으로도 계속 기쁨이 넘쳐나는 삶이 되시기를 바래요!"]
                    else:
                        GeneralAnswer = ["괜찮아요! ", "다음에 언제든 원하실 때 찾아주세요."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "불만":
                if self.cnt == 1:
                    GeneralAnswer = [name + "님이 많이 화가 나신 것 같아 보여요..",
                                     "짜증나는 일이 많은 게 현실이죠. " + name + "님의 일도 마찬가지로 짜증이 날만한 일일 거 같아요.",
                                     name + "님의 상황에선 화가 안나는 게 이상한 일인 것 같기도 하구요.",
                                     "혹시 괜찮다면 제 이야기를 한번 들어보시겠어요?"]
                    self.cnt += 1

                elif self.cnt == 2:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["우리 사회는 참는 것을 미덕으로 여기지만 저는 가끔 그런 생각이 들어요..",
                                         "참다가 화병 생기는 것 보다는 화내는 게 낫지 않을까? 참다가 병 생기면 오래 간다 하더라구요.. ",
                                         "그런 것보다 화내는 것이 훨씬 좋아요 그 대신 세련되게 화내야겠죠!",
                                         "정말 화내야 될 때 화내기 위해 타이밍을 기다리는 사람만큼 지혜로운 사람은 없죠. ",
                                         "언제나 당신을 응원하고 있을테니, 마음이 편안해지셨으면 좋겠어요."]
                    else:
                        GeneralAnswer = ["괜찮아요! ",
                                         "다음에 언제든 원하실 때 찾아주세요."]
                    self.cnt = 0
                    self.state = "general"

            else:
                if self.cnt == 1:
                    GeneralAnswer = [name + "님이 누군가를 질투하고 있으신 것 같아요.",
                                     "가까운 사람일 수도, 먼 사람일 수도 있겠지만요.",
                                     "혹시 괜찮다면 제 이야기를 한번 들어보시겠어요?"]
                elif self.cnt == 2:
                    reaction = sentiment_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["우리 사회는 누군가랑 비교하게 만들어진 것 같아요",
                                         "끊임없이 비교하다보니 박탈감을 느낄 수도 있구요..",
                                         "질투는 자신을 피폐하게 만들기도 하니까 참 해결하지 어려운 문제 같아요.",
                                         name + "님에게 마음이 안정이 필요하신 것 같네요",
                                         "언제라도 저를 찾아주신다면 모든 이야기를 들어드릴게요!"]
                    else:
                        GeneralAnswer = ["괜찮아요! ",
                                         "다음에 언제든 원하실 때 찾아주세요."]
                    self.cnt = 0
                    self.state = "general"

        if self.manage_dailogbuffer() is True:
            (initial_topic_output, initial_label_prob, topic_percentage), topic_prob_vec = Topic_predict(
                self.Topic_model, [dialogs], self._mTokenizer)
            if EmoOut in ('불만', '당혹', '걱정', '질투', '슬픔', '죄책감', '연민') and (float(initial_label_prob * 100) < 99.0):
                topic_index = np.argmax(topic_prob_vec[0][:7])
                altered_topic_output = self._topic_converter[topic_index]
                Topic = altered_topic_output

            else:
                altered_topic_output = 'None'
                Topic = initial_topic_output
        else:
            Topic = "None"

        #if self.cnt == 0  :
        #  DialogType = "General"
        #else:
        #  DialogType = "Scenario"

        if DialogType == "General" or self.s_flag:
            return GeneralAnswer, EmoOut, Topic, DialogType, self.s_flag
        else:
            return GeneralAnswer, None, Topic, DialogType, self.s_flag

    ##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):

        Data = OrderedDict()

        self.dialog_buffer.append(inputsentence)

        GeneralAnswer, Emotion, Topic, Type, Flag = self.get_results(name, inputsentence)

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["Emotion"] = Emotion
        Data["Topic"] = Topic
        Data["Type"] = Type
        Data["System_Corpus"] = GeneralAnswer
        Data["Flag"] = Flag

        return Data


if __name__ == "__main__":
    import tensorflow as tf
    from __init__ import setup_environ, download_weights

    setup_environ()
    download_weights()

    with tf.device("/device:CPU:0"):
        DoDam = AIModel()
        UserName = "민채"
        while True:
            sample = input("입력 : ")
            output = DoDam.run(UserName, sample)
            print("출력 : {}".format(output))
