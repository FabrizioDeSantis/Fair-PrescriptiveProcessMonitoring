import ltn
from model.lstm import LSTMModel
import pandas as pd
import numpy as np
import torch
from data.dataset import CounterfactualDataset, ModelConfig
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from data.utils import compute_cf_mse
from utils import recommend_act_loader, compute_cf_fairness
from data.utils import compute_average_execution_time

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
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--abstraction_mode", type=str, default="sequence", help="Abstraction mode for transition system")
    parser.add_argument("--k", type=int, default=1, help="Parameter k for transition system")

    return parser.parse_args()

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename='output.log', level=logging.INFO)

    args = get_args()

    print("INFO: Reading dataset...")
    data = pd.read_pickle('data/bpi12.pkl')

    if 'case_id' in data.columns:
        case_id_name = 'case_id'
    elif 'case:concept:name' in data.columns:
        case_id_name = 'case:concept:name'
    else:
        raise ValueError("Could not find case ID column. Available columns: " + str(list(data.columns)))

    if 'activity' in data.columns:
        activity_name = 'activity'
    elif 'concept:name' in data.columns:
        activity_name = 'concept:name'
    else:
        raise ValueError("Could not find activity column. Available columns: " + str(list(data.columns)))
    
    data[case_id_name] = data[case_id_name].astype(int)

    print("INFO: Dataset read successfully.")
    print("INFO: Preprocessing dataset...")
    if args.dataset == "bpi12" or args.dataset == "bpi12_kappel":
        from data import preprocess_bpi12
        (case_ids_train, X_train, X_train_cf, y_train, case_ids_val, X_val, X_val_cf, y_val, case_ids_test, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes, transition_system, activity_mapping = preprocess_bpi12.preprocess_eventlog(data, args.dataset, "gender", ["case:AMOUNT_REQ"], args.abstraction_mode)
    print("INFO: Preprocessing completed.")
    
    print("INFO: Feature names: ", feature_names)

    avg_exec_times = compute_average_execution_time(data, case_col=case_id_name, activity_col=activity_name, timestamp_col="time:timestamp")
    print("INFO: Average execution times computed.")

    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_epochs = args.num_epochs,
        dataset = "bpi12",
        sequence_length = seq_len,
        seed = args.seed
    )

    train_dataset = CounterfactualDataset(X_train, X_train_cf, y_train, case_ids_train)
    val_dataset = CounterfactualDataset(X_val, X_val_cf, y_val, case_ids_val)
    test_dataset = CounterfactualDataset(X_test, X_test_cf, y_test, case_ids_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if args.train_vanilla:
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(0, args.num_epochs):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x, x_cf, y, case_id) in enumerate(train_loader):
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
                for batch_idx, (x, x_cf, y, case_id) in enumerate(val_loader):
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
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                y_pred = lstm(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = lstm(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
        print("INFO: Evaluation completed!")
        print(f"INFO: Baseline Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader, transition_system, lstm, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LSTM", save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: Baseline Counterfactual Fairness: {cf_fairness:.4f}")
    
    if args.train_nesy:
        # LTN Knowledge Base
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        alpha = 0.05
        Eq = ltn.Predicate(func = lambda u, v: torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1) + 1e-10 )))
        SatAgg = ltn.fuzzy_ops.SatAgg()
        model = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        f = ltn.Function(model)
        params = list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
        def compute_regr_fair_sat_level(loader):
            mean_sat_regr = 0
            mean_sat_fair = 0
            for x_data, x_cf_data, y_data, case_id in loader:
                x_data, y_data, x_cf_data = x_data.to(device), y_data.to(device), x_cf_data.to(device)
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                x_cf = ltn.Variable("x_cf", x_cf_data)
                mean_sat_regr += Forall(ltn.diag(x, y), Eq(f(x), y)).value
                mean_sat_fair += Forall(ltn.diag(x, x_cf), Eq(f(x), f(x_cf))).value
            mean_sat_regr /= len(loader)
            mean_sat_fair /= len(loader)
            return mean_sat_regr, mean_sat_fair
        def compute_mae_rmse(loader):
            mean_mae = 0
            mean_rmse = 0
            for x, x_cf, y, case_id in loader:
                x, y = x.to(device), y.to(device)
                predictions = f.model(x).cpu().detach().numpy()
                predictions = scalers_test['remaining_time'].inverse_transform(predictions.reshape(-1, 1))
                predictions = np.maximum(predictions, 0)
                y = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                mean_mae += mean_absolute_error(y, predictions)
                mean_rmse += root_mean_squared_error(y, predictions)
            mean_mae /= len(loader)
            mean_rmse /= len(loader)
            return mean_mae, mean_rmse
        def compute_cf_mse_nesy(loader):
            total_mse = 0
            count = 0
            for x, x_cf, y, case_id in loader:
                x, x_cf = x.to(device), x_cf.to(device)
                predictions = f.model(x).cpu().detach().numpy()
                predictions = scalers_test['remaining_time'].inverse_transform(predictions.reshape(-1, 1))
                predictions = np.maximum(predictions, 0)
                predictions_cf = f.model(x_cf).cpu().detach().numpy()
                predictions_cf = scalers_test['remaining_time'].inverse_transform(predictions_cf.reshape(-1, 1))
                predictions_cf = np.maximum(predictions_cf, 0)
                mse = np.mean((predictions - predictions_cf) ** 2)
                total_mse += mse * x.size(0)
                count += x.size(0)
            return total_mse / count
        print("INFO: Starting training LTN model...")
        print("INFO: Training with regression axioms only...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x_data, x_cf_data, y_data, case_id) in enumerate(train_loader):
                x_data, x_cf_data, y_data = x_data.to(device), x_cf_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                sat_agg = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if epoch % 1 == 0:
                mae, rmse = compute_mae_rmse(test_loader)
                sat_regr_train, sat_fair_train = compute_regr_fair_sat_level(train_loader)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level(val_loader)
                sat_regr_test, sat_fair_test = compute_regr_fair_sat_level(test_loader)
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                y_pred = model(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = model(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
        print("INFO: Evaluation completed!")
        print(f"INFO: LTN Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN", save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN Counterfactual Fairness: {cf_fairness:.4f}")
        print("INFO: Training with regression and fairness axioms...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_data, x_cf_data, y_data, case_id) in enumerate(train_loader):
                x_data, x_cf_data, y_data = x_data.to(device), x_cf_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                x = ltn.Variable("x", x_data)
                x_cf = ltn.Variable("x_cf", x_cf_data)
                y = ltn.Variable("y", y_data)
                sat_regression = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                sat_fairness = Forall(ltn.diag(x, x_cf), Eq(f(x), f(x_cf))).value
                sat_agg = SatAgg(*[sat_regression, sat_fairness])
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if epoch % 1 == 0:
                mae, rmse = compute_mae_rmse(test_loader)
                sat_regr_train, sat_fair_train = compute_regr_fair_sat_level(train_loader)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level(val_loader)
                sat_regr_test, sat_fair_test = compute_regr_fair_sat_level(test_loader)
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                y_pred = model(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = model(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
        print("INFO: Evaluation completed!")
        print(f"INFO: LTN + Fairness Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN_Fairness", save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN + Fairness Counterfactual Fairness: {cf_fairness:.4f}")