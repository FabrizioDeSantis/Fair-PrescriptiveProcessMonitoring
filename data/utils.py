import pandas as pd
import numpy as np
import datetime
import torch
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def remaining_time_helper_func(df, case_id_column):
    case_id = case_id_column
    event_time = "time:timestamp"
    unique_cases = df[case_id].unique()
    remaining_times = []
    df[event_time] = df[event_time].astype(str)
    for _, case in enumerate(unique_cases):
        time = df[df[case_id] == case][event_time].str[:19].to_list()
        for i in range(0, len(time)):
            time_stamp = time[i]
            tcc = datetime.datetime.strptime(time[-1], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            # tcc = tcc.total_seconds() / 60
            tcc = tcc.total_seconds() / 86400
            remaining_times.append(tcc)
    df["remaining_time"] = remaining_times
    return df

def create_temporal_features(df):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format="mixed")
    df['day_of_week'] = df['time:timestamp'].dt.dayofweek
    df['time_of_day'] = (
        (df['time:timestamp'].dt.hour * 3600) + 
        (df['time:timestamp'].dt.minute * 60) + 
        (df['time:timestamp'].dt.second)
    ) / 86400
    df["time_delta"] = df.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds().fillna(0)
    return df

def compute_average_execution_time(event_log, case_col, activity_col, timestamp_col):
    
    df = event_log.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed")
    
    df = df.sort_values([case_col, timestamp_col]).reset_index(drop=True)
    
    df['prev_timestamp'] = df.groupby(case_col)[timestamp_col].shift(1)
    df['execution_time'] = df[timestamp_col] - df['prev_timestamp']
    
    df['execution_time_seconds'] = df['execution_time'].dt.total_seconds()
    
    df_with_exec_time = df[df['execution_time_seconds'].notna()].copy()
    
    avg_times = df_with_exec_time.groupby(activity_col).agg(
        avg_execution_time=('execution_time_seconds', 'mean'),
        count=('execution_time_seconds', 'count')
    ).reset_index()
    
    avg_times = avg_times.sort_values('avg_execution_time', ascending=False)
    
    result_dict = dict(zip(avg_times[activity_col], avg_times['avg_execution_time']))
    
    return result_dict

def compute_cf_mse(predictions, predictions_cf):
    predictions = np.array(predictions)
    predictions_cf = np.array(predictions_cf)
    mse = np.mean((predictions - predictions_cf) ** 2)
    return mse

def create_temporal_split(data, case_id_column, val_ratio=0.2, test_ratio=0.2):
    data = data.groupby(case_id_column).first().reset_index()
    graph_ids = data[case_id_column].unique()
    num_graphs = len(graph_ids)
    test_size = int(num_graphs * test_ratio)
    val_size = int((num_graphs - test_size) * val_ratio)
    train_size = num_graphs - test_size - val_size
    print(f"INFO: Total traces: {num_graphs}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    train_ids = graph_ids[:train_size]
    val_ids = graph_ids[train_size:train_size + val_size]
    test_ids = graph_ids[train_size + val_size:]
    return train_ids, val_ids, test_ids

def create_prefixes(data, data_counterfactual, case_id_column, train_ids, val_ids, test_ids, numerical_features):

    prefixes_test = []
    prefixes_val = []
    prefixes_training = []

    prefixes_counterfactual_test = []
    prefixes_counterfactual_val = []
    prefixes_counterfactual_training = []

    labels_training = []
    labels_training_counterfactual = []
    labels_val = []
    labels_val_counterfactual = []
    labels_test = []
    labels_test_counterfactual = []

    scalers_training = {}
    scalers_val = {}
    scalers_test = {}

    case_ids_train = []
    case_ids_val = []
    case_ids_test = []

    case_ids_train_counterfactual = []
    case_ids_val_counterfactual = []
    case_ids_test_counterfactual = []

    max_seq_len = data.groupby(case_id_column).size().max()
    print("INFO: Maximum sequence length: ", max_seq_len)
    training_data = data[data[case_id_column].isin(train_ids)]
    training_data_counterfactual = data_counterfactual[data_counterfactual[case_id_column].isin(train_ids)]
    val_data = data[data[case_id_column].isin(val_ids)]
    val_data_counterfactual = data_counterfactual[data_counterfactual[case_id_column].isin(val_ids)]
    test_data = data[data[case_id_column].isin(test_ids)]
    test_data_counterfactual = data_counterfactual[data_counterfactual[case_id_column].isin(test_ids)]

    for feature in numerical_features:
        scaler = MinMaxScaler()
        training_data[feature] = training_data[feature].ffill()
        training_data[feature] = scaler.fit_transform(training_data[[feature]])
        training_data_counterfactual[feature] = training_data_counterfactual[feature].ffill()
        training_data_counterfactual[feature] = scaler.transform(training_data_counterfactual[[feature]])
        scalers_training[feature] = scaler
        val_data[feature] = val_data[feature].ffill()
        val_data[feature] = scaler.transform(val_data[[feature]])
        val_data_counterfactual[feature] = val_data_counterfactual[feature].ffill()
        val_data_counterfactual[feature] = scaler.transform(val_data_counterfactual[[feature]])
        scalers_val[feature] = scaler
        test_data[feature] = test_data[feature].ffill()
        test_data[feature] = scaler.transform(test_data[[feature]])
        test_data_counterfactual[feature] = test_data_counterfactual[feature].ffill()
        test_data_counterfactual[feature] = scaler.transform(test_data_counterfactual[[feature]])
        scalers_test[feature] = scaler

    scaler = StandardScaler()
    training_data["remaining_time"] = scaler.fit_transform(training_data[["remaining_time"]])
    scalers_training["remaining_time"] = scaler
    val_data["remaining_time"] = scaler.transform(val_data[["remaining_time"]])
    scalers_val["remaining_time"] = scaler
    test_data["remaining_time"] = scaler.transform(test_data[["remaining_time"]])
    scalers_test["remaining_time"] = scaler
    
    for id_value, group in training_data.groupby(case_id_column):

        group = group.reset_index(drop=True)
        group = group.drop(columns=[case_id_column, "time:timestamp"])

        counterfactual_group = training_data_counterfactual[training_data_counterfactual[case_id_column] == id_value]
        counterfactual_group = counterfactual_group.reset_index(drop=True)
        counterfactual_group = counterfactual_group.drop(columns=[case_id_column, "time:timestamp"])

        counterfactual_groups = counterfactual_group.groupby("counterfactual_index")

        for n in range(1, len(group)-1, 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            ngram_df = ngram_df.drop(columns=["remaining_time"])
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
            prefixes_training.append(cols)
            case_ids_train.append(id_value)
            
            counterfactuals = []
            for cf_idx, cf_group in counterfactual_groups:
                cf_group = cf_group.reset_index(drop=True)
                cf_ngram_df = cf_group.iloc[:n]
                cf_ngram_df = cf_ngram_df.drop(columns=["counterfactual_index", "remaining_time"])
                cf_list_of_lists = cf_ngram_df.values.tolist()
                cf_cols = [list(col) for col in zip(*cf_list_of_lists)]
                cf_cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cf_cols]
                counterfactuals.append(cf_cols)

            prefixes_counterfactual_training.append((cols, counterfactuals))
            case_ids_train_counterfactual.append(id_value)
            labels_training_counterfactual.append(label)
    
    print(cf_ngram_df.columns.tolist())

    for id_value, group in val_data.groupby(case_id_column):

        group = group.reset_index(drop=True)
        group = group.drop(columns=[case_id_column, "time:timestamp"])        

        counterfactual_group = val_data_counterfactual[val_data_counterfactual[case_id_column] == id_value]
        counterfactual_group = counterfactual_group.reset_index(drop=True)
        counterfactual_group = counterfactual_group.drop(columns=[case_id_column, "time:timestamp"])

        counterfactual_groups = counterfactual_group.groupby("counterfactual_index")

        for n in range(1, len(group)-1, 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_val.append(label)
            ngram_df = group.iloc[:n]
            ngram_df = ngram_df.drop(columns=["remaining_time"])
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
            prefixes_val.append(cols)
            case_ids_val.append(id_value)

            counterfactuals = []
            for cf_idx, cf_group in counterfactual_groups:
                cf_group = cf_group.reset_index(drop=True)
                cf_ngram_df = cf_group.iloc[:n]
                cf_ngram_df = cf_ngram_df.drop(columns=["counterfactual_index", "remaining_time"])
                cf_list_of_lists = cf_ngram_df.values.tolist()
                cf_cols = [list(col) for col in zip(*cf_list_of_lists)]
                cf_cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cf_cols]
                counterfactuals.append(cf_cols)

            prefixes_counterfactual_val.append((cols, counterfactuals))
            case_ids_val_counterfactual.append(id_value)
            labels_val_counterfactual.append(label)

    for id_value, group in test_data.groupby(case_id_column):

        # take the corresponding group in test_data_counterfactual with the same id_value
        counterfactual_group = test_data_counterfactual[test_data_counterfactual[case_id_column] == id_value]

        group = group.reset_index(drop=True)
        group = group.drop(columns=[case_id_column, "time:timestamp"])
        counterfactual_group = counterfactual_group.reset_index(drop=True)
        counterfactual_group = counterfactual_group.drop(columns=[case_id_column, "time:timestamp", "counterfactual_index"])

        feature_names = group.columns.tolist()

        # Choose a random counterfactual group by counterfactual_index

        random_n = random.randint(2, len(group)-1)

        labels_test.append(group.iloc[random_n-1]["remaining_time"])
        labels_test_counterfactual.append(counterfactual_group.iloc[random_n-1]["remaining_time"])

        ngram_df = group.iloc[:random_n].drop(columns=["remaining_time"])
        list_of_lists = ngram_df.values.tolist()
        cols = [list(col) for col in zip(*list_of_lists)]
        cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
        prefixes_test.append(cols)

        cf_ngram_df = counterfactual_group.iloc[:random_n].drop(columns=["remaining_time"])
        cf_list_of_lists = cf_ngram_df.values.tolist()
        cf_cols = [list(col) for col in zip(*cf_list_of_lists)]
        cf_cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cf_cols]
        
        prefixes_counterfactual_test.append((cols, [cf_cols]))
        case_ids_test.append(id_value)
        case_ids_test_counterfactual.append(id_value)

    feature_names = ngram_df.columns.tolist()
    print(feature_names)
    
    return (case_ids_train, case_ids_train_counterfactual), prefixes_training, prefixes_counterfactual_training, (labels_training, labels_training_counterfactual), (case_ids_val, case_ids_val_counterfactual), prefixes_val, prefixes_counterfactual_val, (labels_val, labels_val_counterfactual), (case_ids_test, case_ids_test_counterfactual), prefixes_test, prefixes_counterfactual_test, (labels_test, labels_test_counterfactual), max_seq_len, feature_names, scalers_training, scalers_val, scalers_test

if __name__ == "__main__":
    data = pd.read_pickle("bpi12.pkl")
    avg_exec_times = compute_average_execution_time(data, case_col="case_id", activity_col="activity", timestamp_col="time:timestamp")
    for activity, avg_time in avg_exec_times.items():
        print(f"Activity: {activity}, Average Execution Time (seconds): {avg_time:.2f}")