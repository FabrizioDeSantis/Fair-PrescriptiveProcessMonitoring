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
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="bpi12", help="Dataset to use")
    parser.add_argument("--backbone", type=str, default="lstm", help="Backbone model type: lstm or transformer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--case_id", type=str, default="case:concept:name", help="Case ID column name in the dataset")

    return parser.parse_args()

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='output.log', level=logging.INFO)

    args = get_args()

    print("INFO: Reading dataset...")
    data = pd.read_pickle('data/bpi12.pkl')
    # data = pd.read_csv('data/bpi12.csv')
    print("INFO: Dataset read successfully!")
    print("INFO: Preprocessing dataset...")
    if args.dataset == "bpi12" or args.dataset == "bpi12_kappel":
        from data import preprocess_bpi12
        (X_train, y_train, X_val, y_val, X_test, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes = preprocess_bpi12.preprocess_eventlog(data, args.dataset, args.case_id)
    print("INFO: Preprocessing completed!")
    print("INFO: Feature names: ", feature_names)
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
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    if args.train_vanilla:
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(0, args.num_epochs):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = lstm(x)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            lstm.eval()
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)
                    outputs = lstm(x)
                    val_loss = criterion(outputs.squeeze(), y)
                    total_validation_loss += val_loss.item()
            avg_val_loss = total_validation_loss / len(val_loader)
            print(f"INFO: Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            # Early stopping check
            if epoch == 0:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(lstm.state_dict(), 'best_model.pt')
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(lstm.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"INFO: Early stopping triggered at epoch {epoch+1}")
                        lstm.load_state_dict(torch.load('best_model.pt'))
                        break
        print("INFO: Training completed!")
        print("INFO: Starting evaluation on test set...")
        # lstm.load_state_dict(torch.load('best_model.pt'))
        lstm.eval()
        maes = []
        rmses = []
        predictions = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = lstm(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
        print("INFO: Evaluation completed!")
        print(f"INFO: Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, Test RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}")

    if args.train_nesy:
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        SatAgg = ltn.fuzzy_ops.SatAgg()
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        P = ltn.Predicate(lstm)
        params = list(P.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        print("INFO: Starting training LTN model...")
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
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--train_vanilla", action="store_true", help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", action="store_true", help="Train LTN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="bpi12", help="Dataset to use")
    parser.add_argument("--backbone", type=str, default="lstm", help="Backbone model type: lstm or transformer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--case_id", type=str, default="case:concept:name", help="Case ID column name in the dataset")

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
    if args.dataset == "bpi12" or args.dataset == "bpi12_kappel":
        from data import preprocess_bpi12
        (X_train, X_train_cf, y_train, X_val, X_val_cf, y_val, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes = preprocess_bpi12.preprocess_eventlog(data, args.dataset, args.case_id, "gender", ["case:AMOUNT_REQ"])
    print("INFO: Preprocessing completed!")
    print("INFO: Feature names: ", feature_names)
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
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    if args.train_vanilla:
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(0, args.num_epochs):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = lstm(x)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            lstm.eval()
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)
                    outputs = lstm(x)
                    val_loss = criterion(outputs.squeeze(), y)
                    total_validation_loss += val_loss.item()
            avg_val_loss = total_validation_loss / len(val_loader)
            print(f"INFO: Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if epoch == 0:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(lstm.state_dict(), 'best_model.pt')
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(lstm.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"INFO: Early stopping triggered at epoch {epoch+1}")
                        lstm.load_state_dict(torch.load('best_model.pt'))
                        break
        print("INFO: Training of vanilla baseline completed!")
        print("INFO: Starting evaluation on test set...")
        lstm.load_state_dict(torch.load('best_model.pt'))
        lstm.eval()
        maes = []
        rmses = []
        predictions = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = lstm(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
        print("INFO: Evaluation completed!")
        print(f"INFO: Baseline Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}")

    if args.train_nesy:
        # LTN Knowledge Base
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        # Equality Predicate - not trainable
        alpha = 0.05
        Eq = ltn.Predicate(func = lambda u, v: torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1))))
        SatAgg = ltn.fuzzy_ops.SatAgg()
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        f = ltn.Function(lstm)
        params = list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        def compute_sat_level(loader):
            mean_sat = 0
            for x_data, y_data in loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                mean_sat += Forall(ltn.diag(x, y), Eq(f(x), y)).value
            mean_sat /= len(loader)
            return mean_sat
        def compute_mae_rmse(loader):
            mean_mae = 0
            mean_rmse = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                predictions = f.model(x).cpu().detach().numpy()
                mean_mae += mean_absolute_error(y.cpu().numpy(), predictions)
                mean_rmse += root_mean_squared_error(y.cpu().numpy(), predictions)
            mean_mae /= len(loader)
            mean_rmse /= len(loader)
            return mean_mae, mean_rmse
        print("INFO: Starting training LTN model...")
        for epoch in range(0, args.num_epochs_nesy):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x_data, y_data) in enumerate(train_loader):
                x, y = x_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                sat_agg = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if epoch % 5 == 0:
                mae, rmse = compute_mae_rmse(test_loader)
                print(f"INFO: Epoch [{epoch+1}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Sat: {compute_sat_level(train_loader):.4f}, Test Sat: {compute_sat_level(val_loader):.4f}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")