import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.model_utils import init_embeddings

torch.manual_seed(0)


class LSTMClassifier(nn.Module):
    def __init__(self, model_hp):
        super(LSTMClassifier, self).__init__()
        self.oov_id = model_hp["oov_id"]
        self.embedding_dim = model_hp["emb_dim"]
        # utils function for embeddings
        self.embeddings, self.vocabulary = init_embeddings(self.embedding_dim)
        self.hidden_size = model_hp["hidden_size"]
        self.layers = model_hp["lstm_layers"]
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, model_hp["fc1_out"])
        self.dropout1 = nn.Dropout(p=model_hp["dropout1"])
        self.fc2 = nn.Linear(model_hp["fc1_out"], 1)
        self.embeddings.weight.requires_grad = False

    def forward(self, sentence_arr):
        batch_size = sentence_arr.shape[0]
        h0 = Variable(torch.zeros(self.layers, batch_size, self.hidden_size)).requires_grad_()
        c0 = Variable(torch.zeros(self.layers, batch_size, self.hidden_size)).requires_grad_()
        inpt = self.create_input(sentence_arr)
        out, (out_state, final_cell_state) = self.lstm(inpt, (h0.detach(), c0.detach()))
        x = F.relu(out[:, -1, :])
        x = self.dropout1(F.relu(self.fc1(x)))
        out = torch.sigmoid(self.fc2(x))
        return out

    def predict(self, X):
        return self.forward(X)

    def create_input(self, sentence_arr):
        """
        code inspired from https://stackoverflow.com/questions/53316174/
        using-pre-trained-word-embeddings-how-to-create-vector-for-unknown-oov-token
        """
        sentence_arr = sentence_arr
        mask = (sentence_arr == self.oov_id).long()
        embed_known = self.embeddings((1 - mask) * sentence_arr)
        embed_random = mask.unsqueeze(-1) * torch.nn.Parameter(data=torch.rand(embed_known.shape))
        emb = embed_known + embed_random
        emb = emb.detach()
        return emb
