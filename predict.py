import argparse
import pandas as pd
from src.utils.preprocessing import txt_cleanning
from src.models.tweet import TweetClassifier


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        help="data path", default=".")
    parser.add_argument("--model", type=str,
                        help="model type: lstm | snn")
    parser.add_argument("--load", type=str,
                        help="path to load the model")
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    model_type = args.model

    model = TweetClassifier(model_type)
    load_path = args.load

    data_df = pd.read_csv(args.data, header=None, sep="\t")
    X = data_df[0].apply(txt_cleanning).values
    model.load(load_path)

    if model_type == "snn":
        X = X.model.transform(X)

    print("Predictions:")
    print(model.predict(X, transform=False))
