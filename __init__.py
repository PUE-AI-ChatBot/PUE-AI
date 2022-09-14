import os,json

def setup_environ():
    this_dir, this_filename = os.path.split(__file__)

    os.environ['CHATBOT_ROOT'] = this_dir

    print("Environment Variable Set Successfully. root: %s" % (os.environ['CHATBOT_ROOT']))

def download_weights():

    print("Check each weights version and update if they have to.")

    config_url = 'https://drive.google.com/u/0/uc?id=1lSjmYdxyLMwsVjTuFPoZ2zabQbP2f56e&export=download'

    import gdown
    this_dir = os.environ['CHATBOT_ROOT']
    Emo_version = GD_version = "1.0.0"

    if os.path.isfile(this_dir+"/resources/config.json"):
        with open(this_dir+"/resources/config.json",'r') as f:
            loaded = json.load(f)
            Emo_version = loaded["EMO-weights-version"]
            GD_version = loaded["GD-weights-version"]
            SUB_version = loaded["TOPIC-weights-version"]

    output = this_dir.replace("\\", "/") + "/resources/config.json"
    gdown.download(config_url, output, quiet=False)

    with open(this_dir + "/resources/config.json", 'r') as f:
        loaded = json.load(f)
        Emo_flag = not loaded["EMO-weights-version"] == Emo_version
        GD_flag = not loaded['GD-weights-version'] == GD_version
        SUB_flag = not loaded["TOPIC-weights-version"] == SUB_version

    weight_path = this_dir.replace("\\","/") + "/resources/weights"

    if not os.path.exists(weight_path+"/Emo_weights") :
        os.makedirs(weight_path+"/Emo_weights")

    if not os.path.isfile(weight_path+"/Emo_weights/Emo_weights.index") or Emo_flag:
        print("Downloading Emo pretrained index...")
        output = weight_path+"/Emo_weights/Emo_weights.index"
        gdown.download(loaded["EMO-index-url"], output, quiet=False)

    if not os.path.isfile(weight_path+"/Emo_weights/Emo_weights.data-00000-of-00001") or Emo_flag:
        print("Downloading Emo pretrained weights...")
        output = weight_path+"/Emo_weights/Emo_weights.data-00000-of-00001"
        gdown.download(loaded["EMO-data-url"], output, quiet=False)

    if not os.path.exists(weight_path + "/GeneralDialog_weights"):
        os.makedirs(weight_path + "/GeneralDialog_weights")

    if not os.path.isfile(weight_path + "/GeneralDialog_weights/General_weights.h5") or GD_flag:
        print("Downloading Transformer pretrained index...")
        output = weight_path + "/GeneralDialog_weights/General_weights.h5"
        gdown.download(loaded["GD-h5-url"], output, quiet=False)

    if not os.path.exists(weight_path + "/Topic_weights"):
        os.makedirs(weight_path + "/Topic_weights")

    if not os.path.isfile(weight_path + "/Topic_weights/Topic_weights.h5") or GD_flag:
        print("Downloading Topic_weights pretrained index...")
        output = weight_path + "/Topic_weights/Topic_weights.h5"
        gdown.download(loaded["TOPIC-h5-url"], output, quiet=False)

    category = ["연애_결혼", "가족", "군대", "회사_아르바이트"]
    theme_weights = [("LDA_model_{0}","model"), ("LDA_model_{0}.expElogbeta.npy","npy"), ("LDA_model_{0}.state","state"),
                     ("LDA_model_{0}.id2word","id2word")
                      ]
    url_name = ["married", "family", "army", "company"]

    if not os.path.exists(weight_path+"/Subtopic_model"):
        os.makedirs(weight_path+"/Subtopic_model")

    for k, j in enumerate(url_name):
        for file_name,file_extension in theme_weights:
            if not os.path.isfile(weight_path + "/Subtopic_model/" + file_name.format(category[k])) or SUB_flag:
                print("Downloading Sub topic models...")

                output = weight_path + "/Subtopic_model/" + file_name.format(category[k])

                gdown.download(loaded[str(j) + "-" + file_extension + "-url"], output, quiet=False)

    print("Setup has just overed!")


setup_environ()
download_weights()
