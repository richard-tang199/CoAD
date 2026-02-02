# -*-coding:utf-8-*-
import os
from dataclasses import asdict

import numpy as np
import json
import pandas as pd

data_dir = "mask_results//UCR//freq_tf_v2_random_seed_47//all//data"
data_group = os.listdir(data_dir)
data_group.sort()
# data_group = data_group[:180]
result_list = []

for data in data_group:
    try:
        # path = os.path.join("new_results", data, "result.json")
        result = json.load(open(os.path.join(data_dir, data), "r"))
        result_list.append(result)
    except FileNotFoundError:
        print(f"{data} not found")
        continue


duration_list = [result["duration"] for result in result_list]
params_list = [result["params"] for result in result_list]

all_time = np.sum(duration_list)
# mean_params = np.mean(params_list)
mean_params = None

efficiency_dict = {
    "all_time": all_time,
    "mean_params": mean_params,
}

data_dir = data_dir.split('//')

if "baseline" in data_dir[0]:
    save_dir = os.path.join(data_dir[0], data_dir[1], data_dir[2])
else:
    save_dir = os.path.join(data_dir[0], data_dir[1], data_dir[2], data_dir[3])

json.dump(efficiency_dict, open(f"{save_dir}//efficiency_result.json", "w"), indent=4)

# all_result_df.to_csv(f"{save_dir}//all_result.csv")
