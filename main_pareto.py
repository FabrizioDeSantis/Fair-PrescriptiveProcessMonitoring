import ltn
from model.lstm import LSTMModel
import pandas as pd
import numpy as np
import torch
from data.dataset import CounterfactualDataset, ModelConfig, StandardDataset
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from data.utils import compute_cf_mse
from utils import recommend_act_loader, compute_cf_fairness
from data.utils import compute_average_execution_time
import matplotlib.pyplot as plt
import seaborn as sns

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
    print('Args are', args)
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
        (case_ids_train, X_train, X_train_cf, y_train, case_ids_val, X_val, X_val_cf, y_val, case_ids_test, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes, transition_system, activity_mapping = preprocess_bpi12.preprocess_eventlog(data, args.dataset, ["gender"], args.abstraction_mode)
        y_train_og = y_train[0]
        y_train_cf = y_train[1]
        y_val_og = y_val[0]
        y_val_cf = y_val[1]
        y_test_og = y_test[0]
        y_test_cf = y_test[1]
        case_ids_train_og = case_ids_train[0]
        case_ids_train_cf = case_ids_train[1]
        case_ids_val_og = case_ids_val[0]
        case_ids_val_cf = case_ids_val[1]
        case_ids_test_og = case_ids_test[0]
        case_ids_test_cf = case_ids_test[1]
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

    train_dataset_og = StandardDataset(X_train, y_train_og, case_ids_train_og)
    val_dataset_og = StandardDataset(X_val, y_val_og, case_ids_val_og)
    test_dataset_og = StandardDataset(X_test, y_test_og, case_ids_test_og)

    train_dataset_cf = CounterfactualDataset(X_train_cf, y_train_og, case_ids_train_cf)
    val_dataset_cf = CounterfactualDataset(X_val_cf, y_val_og, case_ids_val_cf)
    test_dataset_cf = CounterfactualDataset(X_test_cf, y_test_og, case_ids_test_cf)

    train_loader = DataLoader(train_dataset_og, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset_og, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset_og, batch_size=32, shuffle=False)

    train_loader_cf = DataLoader(train_dataset_cf, batch_size=32, shuffle=True)
    val_loader_cf = DataLoader(val_dataset_cf, batch_size=32, shuffle=False)
    test_loader_cf = DataLoader(test_dataset_cf, batch_size=32, shuffle=False)

    if args.train_vanilla:
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(0, args.num_epochs):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x, y, case_id) in enumerate(train_loader):
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
                for batch_idx, (x, y, case_id) in enumerate(val_loader):
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
        predictions_og = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
                y_pred = lstm(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions_og.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = lstm(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
        
        print("INFO: Evaluation completed!")
        print(f"INFO: Baseline Test MAE: {mean_absolute_error(gt_labels, predictions_og):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions_og):.4f}, CF MSE: {compute_cf_mse(predictions_og, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, lstm, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LSTM", save_to_file=True)
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
            with torch.no_grad():
                for x_data, x_cf_data, y_data, _ in loader:
                    x_data.to(device), x_cf_data.to(device), y_data.to(device)
                    num_counterfactual = x_cf_data.shape[1]
                    x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                    x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                    x = ltn.Variable("x", x_data)
                    x_cf = ltn.Variable("x_cf", x_cf_data)
                    x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                    y = ltn.Variable("y", y_data)
                    mean_sat_regr += Forall(ltn.diag(x, y), Eq(f(x), y)).value
                    mean_sat_fair += Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
            mean_sat_regr /= len(loader)
            mean_sat_fair /= len(loader)
            return mean_sat_regr, mean_sat_fair
        print("INFO: Starting training LTN model...")
        print("INFO: Training with regression axioms only...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x_data, y_data, case_id) in enumerate(train_loader):
                x_data, y_data = x_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                sat_agg = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            model.eval()
            with torch.no_grad():
                for x_data, y_data, _ in val_loader:
                    x_data.to(device), y_data.to(device)
                    x = ltn.Variable("x", x_data)
                    y = ltn.Variable("y", y_data)
                    val_loss = 1.0 - Forall(ltn.diag(x, y), Eq(f(x), y)).value
                    total_validation_loss += val_loss.item()
            avg_val_loss = total_validation_loss / len(val_loader)
            if epoch % 1 == 0:
                sat_regr_train, sat_fair_train = compute_regr_fair_sat_level(train_loader_cf)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level(val_loader_cf)
                sat_regr_test, sat_fair_test = compute_regr_fair_sat_level(test_loader_cf)
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
            if epoch == 1:
                best_val_loss_nesy = avg_val_loss
                patience_counter_nesy = 0
                torch.save(model.state_dict(), 'best_model_nesy_regr.pt')
            else:
                if avg_val_loss < best_val_loss_nesy:
                    best_val_loss_nesy = avg_val_loss
                    patience_counter_nesy = 0
                    torch.save(model.state_dict(), 'best_model_nesy_regr.pt')
                else:
                    patience_counter_nesy += 1
                    if patience_counter_nesy >= 10:
                        print(f"INFO: Early stopping triggered at epoch {epoch}")
                        model.load_state_dict(torch.load('best_model_nesy_regr.pt'))
                        break
        model.load_state_dict(torch.load('best_model_nesy_regr.pt'))
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
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
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN", save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN Counterfactual Fairness: {cf_fairness:.4f}")
        pareto_archive_val = []
        model_checkpoints = []
        val_history = []
        patience = 10
        last_val_improvement = 0
        print("INFO: Training with regression and fairness axioms using Pareto front monitoring...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_data, x_cf_data, y_data, case_id) in enumerate(train_loader_cf):
                x_data, x_cf_data, y_data = x_data.to(device), x_cf_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                num_counterfactual = x_cf_data.shape[1]
                x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                x = ltn.Variable("x", x_data)
                x_cf = ltn.Variable("x_cf", x_cf_data)
                x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                y = ltn.Variable("y", y_data)
                sat_regression = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                sat_fairness = Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
                sat_agg = SatAgg(*[sat_regression, sat_fairness])
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_cf)
            if epoch % 1 == 0:
                sat_regr_train, sat_fair_train = compute_regr_fair_sat_level(train_loader_cf)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level(val_loader_cf)
                sat_regr_test, sat_fair_test = compute_regr_fair_sat_level(test_loader_cf)
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
                current_val = [sat_regr_val.item(), sat_fair_val.item()]
                val_history.append(current_val)
                dominated = False
                for i, p in enumerate(pareto_archive_val):
                    if all(current_val[j] >= p[j] for j in range(2)):
                        pareto_archive_val.pop(i)
                    elif all(p[j] >= current_val[j] for j in range(2)):
                        dominated = True
                        break
                if not dominated:
                    pareto_archive_val.append(current_val)
                    model_path = f'best_pareto_model_e{epoch}_sats{current_val}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_sats': current_val,
                        'pareto_archive': pareto_archive_val.copy()
                    }, model_path)
                    model_checkpoints.append(model_path)
                    last_val_improvement = epoch
                    print(f"INFO: New Pareto front found at epoch {epoch}, saved model to {model_path}")
                if epoch - last_val_improvement >= patience:
                    print(f"INFO: Early stopping triggered at epoch {epoch+1}. Val Pareto stable for {patience}")
                    break
        pareto_models = []
        for model_path in model_checkpoints:
            checkpoint = torch.load(model_path)
            sats = checkpoint['val_sats']
            pareto_models.append((sats, model_path))
        best_idx = np.argmax([min(sats) for sats, _ in pareto_models])
        best_model_path = pareto_models[best_idx][1]
        print(f"INFO: Loading best Pareto model from {best_model_path} for final evaluation.")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
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
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN_Fairness", save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN + Fairness Counterfactual Fairness: {cf_fairness:.4f}")
        val_data = np.array(val_history)
        pareto_data = np.array(pareto_archive_val)
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.figure(figsize=(10, 8), dpi=300)
        df_val = pd.DataFrame(val_data, columns=['Regression Satisfaction', 'Fairness Satisfaction'])
        sns.scatterplot(data=df_val, x='Regression Satisfaction', y='Fairness Satisfaction', s=80, color='lightblue', alpha=0.7, edgecolor='navy', linewidth=0.8, label='Validation Epochs')
        df_pareto = pd.DataFrame(pareto_data, columns=['Regression Satisfaction', 'Fairness Satisfaction'])
        sort_idx = np.argsort(df_pareto['Regression Satisfaction'])[::-1]
        pareto_sorted = df_pareto.iloc[sort_idx].reset_index(drop=True)
        sns.lineplot(data=pareto_sorted, x='Regression Satisfaction', y='Fairness Satisfaction',
                color='red', linewidth=5, marker='o', markersize=12,
                markeredgecolor='darkred', markeredgewidth=2,
                label='Validation Pareto front')
        plt.xlabel('Regression Axioms Satisfaction', fontsize=14, fontweight='bold')
        plt.ylabel('Fairness Axioms Satisfaction', fontsize=14, fontweight='bold')
        plt.title('LTN Training: Pareto Front Monitoring (Validation)', 
                fontsize=16, fontweight='bold', pad=20)

        # Legend styling
        plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)

        # Set axis limits and grid
        if len(val_data) > 0 and len(pareto_data) > 0:
            all_x = np.concatenate([val_data[:,0], pareto_data[:,0]])
            all_y = np.concatenate([val_data[:,1], pareto_data[:,1]])
        else:
            all_x = val_data[:,0] if len(val_data) > 0 else pareto_data[:,0]
            all_y = val_data[:,1] if len(val_data) > 0 else pareto_data[:,1]

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        padding = 0.05  # 5% padding

        plt.xlim(x_min - (x_max - x_min)*padding, x_max + (x_max - x_min)*padding)
        plt.ylim(y_min - (y_max - y_min)*padding, y_max + (y_max - y_min)*padding)
        
        sns.despine()  # Clean border

        # Tight layout for perfect spacing
        plt.tight_layout()

        # Save high-quality PDF (vector graphics, publication ready)
        plt.savefig('ltn_pareto_front.pdf', format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', backend='pdf')