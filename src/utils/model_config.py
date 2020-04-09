model_hp = {
    "snn": {"input_size": 512,
            "fc1_out": 512,
            "fc2_out": 512,
            "fc3_out": 32,
            "dropout1": 0.6,
            "dropout2": 0.2,
            "module_path": "./use"},
    "lstm": {"emb_dim": 200,
             "lstm_layers": 1,
             "hidden_size": 200,
             "fc1_out": 100,
             "dropout1": 0.6,
             "oov_id": -1}
}

train_config_params = {
    "snn": {"lr": 0.001,
            "weight_decay": 0.007,
            "scheduler_factor": 0.1,
            "scheduler_patience": 5},
    "lstm": {"lr": 0.01,
             "weight_decay": 0.011,
             "scheduler_factor": 0.1,
             "scheduler_patience": 10}
}
