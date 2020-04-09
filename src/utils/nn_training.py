import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.model_config import train_config_params

torch.manual_seed(0)


def _init_train_config(model, lr=0.01, weight_decay=0.015,
                       scheduler_factor=0.1, scheduler_patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_config = {"criterion": nn.BCELoss(),
                    "optimizer": optimizer,
                    "lr_scheduler": ReduceLROnPlateau(optimizer, 'max', factor=scheduler_factor,
                                                      patience=scheduler_patience, verbose=True),
                    "best_accuracy": 0}
    return train_config


def _save_model(model, train_config, path):
    train_state = {
        "state_dict": model.state_dict(),
        "optimizer": train_config["optimizer"].state_dict(),
        "scheduler": train_config["lr_scheduler"].state_dict()
    }
    torch.save(train_state, path)


def get_accuracy(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    for test_X, test_y in test_dl:
        outputs = model.predict(test_X).round()
        total += test_y.shape[0]
        correct += outputs.squeeze().eq(test_y).sum().item()
    return correct / total


def _update_model(model, cur_accuracy, train_config, model_path, verbose):
    train_config["best_accuracy"] = cur_accuracy
    if verbose:
        print("New best accuracy: %f" % (train_config["best_accuracy"]))
    if model_path:
        try:
            if verbose:
                print("Saving current model!!!")
            _save_model(model, train_config, model_path)
        except Exception as e:
            print("!!! Model could not be saved.")
            print("=================={}==================".format(e))
            model_path = None
    return model, train_config, model_path


def train(model, model_type, train_dl, test_dl, epoch, path, verbose):
    """ function to train the model
    Parameters:
    -----------
    model: <SNNClassifier object> or <LSTMClassifier> model class
    model_type: "lstm" or "snn"
    train_dl: <torch DataLoader object> dataloader to iterate training set
    test_dl: <torch DataLoader object> dataloader to iterate testing set
    epoch: <int> training duration
    path: <str> path to save model during traning
    verbose: <boolean> if True print statistics during training

    Returns:
    --------
    best_model: <SNNClassifier object> or <LSTMClassifier> trained model class
    """
    train_config = _init_train_config(model, **train_config_params[model_type])
    model_path = os.path.join(path, "{}.pt".format(model_type)) if path else None
    best_model = model
    print("Model started training.")
    for epoch in tqdm(range(epoch)):
        model.train()
        running_loss = 0.0
        for i, (batch_X, batch_y) in enumerate(train_dl, 0):
            # zero the parameter gradients
            train_config["optimizer"].zero_grad()

            # forward + backward + optimize
            outputs = model(batch_X)
            loss = train_config["criterion"](outputs, batch_y.float())
            loss.backward()
            train_config["optimizer"].step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        cur_accuracy = get_accuracy(model, test_dl)
        if verbose:
            print('Accuracy on EPOCH %d on test images: %d %%' % (epoch + 1, 100 * cur_accuracy))
        train_config["lr_scheduler"].step(cur_accuracy)
        if cur_accuracy > train_config["best_accuracy"]:
            best_model, train_config, model_path = _update_model(model, cur_accuracy, train_config, model_path, verbose)

    print('Model finished training.')
    print("Model's accuracy:", train_config["best_accuracy"])
    if not path:
        print("Warning, model is not saved.")
    return best_model
