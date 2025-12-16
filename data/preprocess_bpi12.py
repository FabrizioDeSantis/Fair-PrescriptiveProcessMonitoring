from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from data.utils import remaining_time_helper_func

def create_temporal_split(data, val_ratio=0.2, test_ratio=0.2):
    data = data.sort_values(by=['case_id', 'time:timestamp'])
    graph_ids = data['case_id'].unique()
    num_graphs = len(graph_ids)
    test_size = int(num_graphs * test_ratio)
    val_size = int((num_graphs - test_size) * val_ratio)
    train_size = num_graphs - test_size - val_size
    print(f"INFO: Total traces: {num_graphs}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    train_ids = graph_ids[:train_size]
    val_ids = graph_ids[train_size:train_size + val_size]
    test_ids = graph_ids[train_size + val_size:]
    return train_ids, val_ids, test_ids

def create_ngrams(data, train_ids, val_ids, test_ids, window_size=40):

    ngrams_test = []
    ngrams_val = []
    ngrams_training = []
    labels_training = []
    labels_val = []
    labels_test = []

    max_seq_len = data.groupby('case_id').size().max()
    window_size = max_seq_len

    training_data = data[data["case_id"].isin(train_ids)]
    val_data = data[data["case_id"].isin(val_ids)]
    test_data = data[data["case_id"].isin(test_ids)]

    for id_value, group in training_data.groupby('case_id'):

        group = group.reset_index(drop=True)

        group = group.drop(columns=["case_id", "time:timestamp"])

        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    for id_value, group in val_data.groupby('case_id'):

        group = group.reset_index(drop=True)

        group = group.drop(columns=["case_id", "time:timestamp"])        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_val.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_val.append(cols)

    for id_value, group in test_data.groupby('case_id'):

        group = group.reset_index(drop=True)

        group = group.drop(columns=["case_id", "time:timestamp"])        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    return ngrams_training, labels_training, ngrams_val, labels_val, ngrams_test, labels_test, max_seq_len,feature_names

def preprocess_eventlog(data, seed):

    vocab_sizes = {}
    scalers = {}

    data = remaining_time_helper_func(data)

    train_ids, val_ids, test_ids = create_temporal_split(data, val_ratio=0.2, test_ratio=0.2)
    
    data = data.drop(columns=["case:REG_DATE", "lifecycle:transition"])

    categorical_features = ["activity", "org:resource", "gender", "day_of_week"]
    numerical_features = ["case:AMOUNT_REQ", "time_delta", "time_of_day"]

    for feature in categorical_features:
        data[feature] = pd.Categorical(data[feature])
        data[feature] = data[feature].cat.codes + 1
        vocab_sizes[feature] = data[feature].max()

    for feature in numerical_features:
        scaler = MinMaxScaler()
        data[feature] = data[feature].ffill()
        data[feature] = scaler.fit_transform(data[[feature]])
        scalers[feature] = scaler

    scaler = StandardScaler()
    data["remaining_time"] = scaler.fit_transform(data[["remaining_time"]])
    scalers["remaining_time"] = scaler

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers