The goal of this project is the development and deployment of a text classifier that can identify disastrous tweets. Two models were trained for this task.\
(inspired by Kaggle's contest: https://www.kaggle.com/c/nlp-getting-started)
1. An LSTM classifier
2. A Fully Connected Neural Classifier
The model's were deployed using a flask server.

In order to use the server, several steps are necessary:\
First:
```
cd <project root>
```
Then:
1. Install requirements:
```
pip install -r requirements.txt
```

2. Train the model
```
python train.py --data data/train_simple.csv --model snn --store model_bin/snn.pt

python train.py --data data/train_simple.csv --model lstm --store model_bin/lstm.pt
```

3. Deploy the model
```
python app.py
```
4. Use the UI on:
```
http://0.0.0.0:5010/predict
```
