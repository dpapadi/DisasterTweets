The goal of this project is the development and deployment of a text classifier that can identify disastrous tweets. (inspired by Kaggle's contest: https://www.kaggle.com/c/nlp-getting-started) \
Two models were trained for this task.

1. An LSTM classifier
2. A Fully Connected Neural Classifier

The model's were deployed using a flask server.\
In order to use the server, several steps are necessary.\
First: \
Make sure you run python version lower than 3.7 (tensorflow is not supported for python==3.7)
```
cd <project root>
```
Then:
1. Install requirements:
```
pip install -r requirements.txt
```
2. Install GloVe embeddings:
```
mkdir glove
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove/glove.twitter.27B.zip -d glove/
```
3. Install Universal Sentence Encoder
```
mkdir useqa3
curl -L "https://tfhub.dev/google/universal-sentence-encoder-qa/3?tf-hub-format=compressed" | tar -zxvC useqa3/
```
4. Train the model
```
mkdir model_bin

python train.py --data data/train_simple.csv --model snn --store model_bin/snn.pt
python train.py --data data/train_simple.csv --model lstm --store model_bin/lstm.pt
```
5. Deploy the model
```
python app.py
```
6. Use the UI on:
```
http://0.0.0.0:5010/predict
```
