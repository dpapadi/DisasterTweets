import pandas as pd
from pathlib import Path
import tensorflow_hub as hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.model_utils import get_use_embs

torch.manual_seed(0)


class SimpleNNClassifier(torch.nn.Module):
    def __init__(self, model_hp):
        super(SimpleNNClassifier, self).__init__()
        self.load_module(model_hp["module_path"])
        self.fc1 = nn.Linear(model_hp["input_size"], model_hp["fc1_out"])
        self.dropout1 = nn.Dropout(p=model_hp["dropout1"])
        self.fc2 = nn.Linear(model_hp["fc1_out"], model_hp["fc2_out"])
        self.dropout2 = nn.Dropout(p=model_hp["dropout2"])
        self.fc3 = nn.Linear(model_hp["fc2_out"], model_hp["fc3_out"])
        self.fc4 = nn.Linear(model_hp["fc3_out"], 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        out = torch.sigmoid(self.fc4(x))
        return out

    def predict(self, x):
        return self.forward(x)

    def transform(self, X):
        print("Creating feature embeddings.")
        embs = pd.Series(X).apply(lambda x: get_use_embs(x, self._module))
        print("Feature embeddings created.")
        return embs.values

    def load_module(self, path):
        if not Path(path).exists():
            raise(ValueError("Path {} does not exist. Module not loaded".format(path)))
        self._module = hub.load(path)
        return self
