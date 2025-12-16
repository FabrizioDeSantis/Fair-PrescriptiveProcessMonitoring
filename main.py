import ltn
from model.lstm import LSTMModel
import pandas as pd
import numpy as np
import torch
from data.dataset import NeSyDataset, ModelConfig
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="bpi12", help="Dataset to use")
    parser.add_argument("--backbone", type=str, default="lstm", help="Backbone model type: lstm or transformer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")

    return parser.parse_args()

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='output.log', level=logging.INFO)

    args = get_args()

    print("INFO: Reading dataset...")
    data = pd.read_pickle('data/bpi12.pkl')
    print("INFO: Dataset read successfully!")
    print("INFO: Preprocessing dataset...")
    if args.dataset == "bpi12":
        from data import preprocess_bpi12
        (X_train, y_train, X_val, y_val, X_test, y_test, seq_len, feature_names), vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(data, args.seed)
    print("INFO: Preprocessing completed!")
    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_epochs = args.num_epochs,
        dataset = "bpi12",
        sequence_length = seq_len,
        seed = args.seed
    )
    train_dataset = NeSyDataset(X_train, y_train)
    val_dataset = NeSyDataset(X_val, y_val)
    test_dataset = NeSyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Knowledge Theory
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    SatAgg = ltn.fuzzy_ops.SatAgg()

    # Define LTN model
    lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
    # P = ltn.Predicate(lstm)
    # params = list(P.parameters())
    # optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.train_vanilla:
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(args.num_epochs):
            lstm.train()
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = lstm(x)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"INFO: Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")
        print("INFO: Training completed!")
        print("INFO: Starting evaluation on test set...")
        lstm.eval()
        maes = []
        rmses = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = lstm(x)
                y_pred = scalers['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                y_true = scalers['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                maes.append(mean_absolute_error(y_true, y_pred))
                rmses.append(root_mean_squared_error(y_true, y_pred))
        mae = sum(maes) / len(maes)
        rmse = sum(rmses) / len(rmses)
        print("INFO: Evaluation completed!")
        print(f"INFO: Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")