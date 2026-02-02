# -*-coding:utf-8-*-
import os
import numpy as np
import json
import pandas as pd


data_dir = "mask_results/UCR/freq_tf_v2_08-35-59/all/data"
data_group = os.listdir(data_dir)
data_group.sort()
# data_group = data_group[:180]
result_list = []

for data in data_group:
    try:
        # path = os.path.join("new_results", data, "result.json")
        result = json.load(open(os.path.join(data_dir,data), "r"))
        result_list.append(result)
    except FileNotFoundError:
        print(f"{data} not found")
        continue

f1_score_all = [result["point_wise"]["f1_score"] for result in result_list]
auc_score_all = [result["point_wise"]["auc_prc"] for result in result_list]
r_prc_all = [result["vus"]["R_AUC_PR"] for result in result_list]
v_prc_all = [result["vus"]["VUS_PR"] for result in result_list]

all_result_df = pd.DataFrame({
    "f1_score": f1_score_all,
    "auc_score": auc_score_all,
    "r_prc": r_prc_all,
    "v_prc": v_prc_all
}, index=data_group)

f1_score = np.mean([result["point_wise"]["f1_score"] for result in result_list])
auc_score = np.mean([result["point_wise"]["auc_prc"] for result in result_list])
r_prc = np.mean([result["vus"]["R_AUC_PR"] for result in result_list])
v_prc = np.mean([result["vus"]["VUS_PR"] for result in result_list])

ucr_1 = np.mean([result["ucr"]["ucr_01"] for result in result_list if result["ucr"] is not None])
ucr_3 = np.mean([result["ucr"]["ucr_03"] for result in result_list if result["ucr"] is not None])
ucr_5 = np.mean([result["ucr"]["ucr_05"] for result in result_list if result["ucr"] is not None])

print(f"F1-score: {f1_score:.4f}")
print(f"AUC-PRC: {auc_score:.4f}")
print(f"R-AUC-PR: {r_prc:.4f}")
print(f"VUS-PR: {v_prc:.4f}")
print(f"UCR@0.1: {ucr_1:.4f}")
print(f"UCR@0.3: {ucr_3:.4f}")
print(f"UCR@0.5: {ucr_5:.4f}")

result_dict = {
    "F1-score": f1_score,
    "AUC-PRC": auc_score,
    "R-AUC-PR": r_prc,
    "VUS-PR": v_prc,
    "UCR@0.1": ucr_1,
    "UCR@0.3": ucr_3,
    "UCR@0.5": ucr_5
}

data_dir = data_dir.split('/')
save_dir = os.path.join(*data_dir[:-1])
json.dump(result_dict, open(f"{save_dir}//aggregate_result.json", "w"), indent=4)
all_result_df.to_csv(f"{save_dir}//all_result.csv")