import pandas as pd
import numpy as np
import datetime

def remaining_time_helper_func(df):
    case_id = "case_id"
    event_time = "time:timestamp"
    unique_cases = df[case_id].unique()
    remaining_times = []
    df[event_time] = df[event_time].astype(str)
    for _, case in enumerate(unique_cases):
        time = df[df[case_id] == case][event_time].str[:19].to_list()
        for i in range(0, len(time)):
            time_stamp = time[i]
            tcc = datetime.datetime.strptime(time[-1], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            tcc = tcc.total_seconds() / 86400
            remaining_times.append(tcc)
    df["remaining_time"] = remaining_times
    df.to_csv("data/bpi12_with_remaining_time.csv", index=False)
    return df