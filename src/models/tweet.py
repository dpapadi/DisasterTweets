import torch
import numpy as np
from src.utils.model_config import model_hp
from src.utils.model_utils import create_dataloader, get_train_test_sets
from src.utils.nn_training import train, get_accuracy
from src.models.lstm import LSTMClassifier
from src.models.snn import SimpleNNClassifier


torch.manual_seed(0)

MODELS = {"snn": SimpleNNClassifier,
          "lstm": LSTMClassifier
          }


class TweetClassifier():
    """ A class that wraps the twitter classification model.
    """
    def __init__(self, model_type, hparams=None, sentence_length=15):
        """ object initiation
        Parameters
        ----------
        model_type: "snn" or "lstm"
        hparams: dict that contains the parameters of the model. Parameters are also
                stored in the src.utils.model_config module. It is recommended to leave this
                value to None and modify parameters there
        sentence_length: only applicable with the "lstm" model type.  sets a fixed length for tweets
        """
        self.model_type = model_type
        if not hparams:
            hparams = model_hp[self.model_type]
        self.model = MODELS[self.model_type](hparams)
        if model_type == "lstm":
            self.sentence_length = sentence_length
        print("Classifier initialized!")

    def fit(self, X, y, X_val=None, y_val=None, shuffle=True, test_size=0.20, batch_size=32, epoch=50,
            path=None, verbose=False, transform=True):
        """ train the model
        Parameters
        ----------
        X: <numpy array> tweet text
        y: <numpy array> tweet labels
        X_val: <numpy array> tweet text for validation
        y_val: <numpy array> tweet labels for validation
        shuffle: <boolean> whether or no to shuffle the training set during training
        test_size: <float> what it the training/test set ratio. Only applicable if there is no validation set
        bach_size: <int> size of the batches used in training
        epoch: <int> number of epochs(training duration)
        path: <str> path to save the model during training
        verbose: <boolean> print information during training
        transform: <boolean> transform input using USE encoder. Only applicable if self.model_type == "snn"
        """
        X, vocabulary, sentence_length = self._init_params(X, transform)
        if X_val is not None and y_val is not None:
            X_val, _, _ = self._init_params(X_val, transform)
        X_train, X_test, y_train, y_test = get_train_test_sets(X, y, X_val, y_val, test_size=test_size, random_state=1)

        train_dl = create_dataloader(X_train, y_train, shuffle=True, batch_size=batch_size,
                                     vocabulary=vocabulary, sentence_length=sentence_length)

        test_dl = create_dataloader(X_test, y_test, shuffle=True, batch_size=batch_size,
                                    vocabulary=vocabulary, sentence_length=sentence_length)

        self.model = train(self.model, self.model_type, train_dl, test_dl, epoch, path, verbose)
        return self

    def predict(self, X, batch_size=64, transform=True):
        """ get predictions
        Parameters
        ----------
        X: <numpy array> tweet text
        bach_size: <int> size of the batches used in training
        transform: <boolean> transform input using USE encoder. Only applicable if self.model_type == "snn"

        Returns:
        --------
        <numpy array> or <int>: predictions
        """
        X, vocabulary, sentence_length = self._init_params(X, transform=transform)
        self.model.eval()
        dataloader = create_dataloader(X, shuffle=False, batch_size=batch_size,
                                       vocabulary=vocabulary, sentence_length=sentence_length)
        outputs = []
        for x in dataloader:
            outputs += self.model.predict(x).round().tolist()
        return np.array(outputs).squeeze()

    def save(self, path):
        """ save model state
        Parameters
        ----------
        path: <str> path to save the model state
        """
        torch.save({"state_dict": self.model.state_dict()}, path)
        print("model saved in {}".format(path))
        return "1"

    def load(self, path):
        """ load model state
        Parameters
        ----------
        path: <str> path to load the model state from
        """
        self.model.load_state_dict(torch.load(path)['state_dict'])
        print("model loaded from {}".format(path))
        return self

    def get_accuracy(self, X, y, batch_size=64, transform=True):
        """ calculate model's accuracy
        Parameters
        ----------
        X: <numpy array> tweet text
        y: <numpy array> tweet labels
        bach_size: <int> size of the batches used in training
        transform: <boolean> transform input using USE encoder. Only applicable if self.model_type == "snn"

        Returns:
        --------
        <float>: accuracy of the model
        """
        X, vocabulary, sentence_length = self._init_params(X, transform)
        dataloader = create_dataloader(X, y, shuffle=False, batch_size=batch_size,
                                       vocabulary=vocabulary, sentence_length=sentence_length)
        return get_accuracy(self.model, dataloader)

    def _init_params(self, X, transform):
        """ hidden function to initialize parameters for training and prediction phase
        Parameters
        ----------
        X: <numpy array> tweet text
        transform: <boolean> transform input using USE encoder. Only applicable if self.model_type == "snn"

        Returns:
        --------
        X: <numpy array> transformed (if applicable) X  input array
        vocabulary: <Vocabulary object> vocabulary from the lstm model if applicable
        sentence_length: <int> fixed length of sentences for the lstm model if applicable
        """
        try:
            vocabulary, sentence_length = self.model.vocabulary, self.sentence_length
        except Exception:
            vocabulary, sentence_length = None, None
            if transform:
                X = self.model.transform(X)
        return X, vocabulary, sentence_length
