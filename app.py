import numpy as np
import flask
from flask import Flask, render_template
from flask_cors import CORS
from src.models.tweet import TweetClassifier
from src.utils.preprocessing import txt_cleanning


APP_CONFIG = {"lstm_model_path": "model_bin/lstm.pt",  # <--- lstm's model path
              "snn_model_path": "model_bin/snn.pt"     # <--- snn's model path
              }

app = Flask("TweetClassifier")
CORS(app)


def init_app_config():
    print("Models are loading. This might take a few minutes.")
    APP_CONFIG["lstm_model"] = TweetClassifier("lstm")
    APP_CONFIG["snn_model"] = TweetClassifier("snn")
    APP_CONFIG["lstm_model"].load(APP_CONFIG["lstm_model_path"])
    APP_CONFIG["snn_model"].load(APP_CONFIG["snn_model_path"])
    print("Models loaded successfully.")


@app.route('/predict')
def my_form(ret=None):
    """ deploy front-end
    """
    if not ret:
        ret = {0: None, 1: "", 2: "lstm_model"}
    return render_template('index.html', r=ret)


@app.route("/predict", methods=["POST"])
def predictLSTM():
    """ classify tweet
    Parameters:
    -----------
    model: <str> "lstm_model" or "snn_model"
    tweet: <str> tweet content to be classified

    Returns:
    -------
    updated html that includes prediction
    """
    tweet = flask.request.form.get("tweet")
    model = flask.request.form.get("model")
    clean_tweet = np.asarray([txt_cleanning(tweet)])
    pred = APP_CONFIG["lstm_model"].predict(clean_tweet)
    ret = {0: pred, 1: tweet, 2: model}
    return my_form(ret)


if __name__ == "__main__":
    # initialize models
    init_app_config()
    app.run(host="0.0.0.0", port=5010)
