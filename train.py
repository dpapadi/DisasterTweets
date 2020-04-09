import argparse
import pandas as pd
from src.utils.preprocessing import txt_cleanning
from src.models.tweet import TweetClassifier


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        help="train data path | must be .csv format of <text>, <label>", default=".")
    parser.add_argument("--model", type=str,
                        help="model type: lstm | snn")
    parser.add_argument("--store", type=str,
                        help="path to store the model")
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    model_type = args.model

    model = TweetClassifier(model_type)
    save_path = args.store

    data_df = pd.read_csv(args.data, header=None)
    X = data_df[0].apply(txt_cleanning).values
    y = data_df[1].values
    if model_type == "snn":
        X = model.model.transform(X)

    model.fit(X, y, epoch=80, transform=False)
    model.save(save_path)
