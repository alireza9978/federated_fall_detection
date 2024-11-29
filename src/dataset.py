import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tsfel
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_path = "datasets/raw"

def read_meta():
    f = open('./datasets/cleaned_meta.json')
    data = json.load(f)
    f.close()
    return data


def save_meta(meta_json):
    f = open('./datasets/cleaned_meta.json', "w")
    json.dump(meta_json, f, ensure_ascii=False, indent=4)
    f.close()


def read_labels():
    df = pd.read_csv("./datasets/activity_map.csv")
    return df

def clean_up_fall(with_gyro):
    df = pd.read_csv(f"{base_path}/Up-Fall/CompleteDataSet.csv")
    belt_columns = df.columns[0:1].tolist() + df.columns[15:21].tolist() + df.columns[-4:].tolist()
    df = df[belt_columns].drop(0)
    df.columns = ['TimeStamps', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Subject', 'Activity', 'Trial',
                  'Tag']
    df["TimeStamps"] = pd.to_datetime(df["TimeStamps"])
    new_dtype = {'acc_x': "float16", 'acc_y': "float16", 'acc_z': "float16", 'gyro_x': "float16", 'gyro_y': "float16",
                 'gyro_z': "float16", 'Subject': "int8", 'Activity': "int8", 'Trial': "int8", 'Tag': "int8"}
    df = df.astype(new_dtype)
    dataset_name = "UpFall"
    meta_json = read_meta()
    this_meta = []

    for i in range(len(meta_json["files"]) - 1, -1, -1):
        file = meta_json["files"][i]
        if file['dataset'] == dataset_name:
            del meta_json["files"][i]
    for f in glob.glob(f"./datasets/cleaned/{dataset_name}_*.csv"):
        os.remove(f)

    labels = read_labels()

    def save_single_file(temp_df: pd.DataFrame):
        activity = temp_df["Activity"].values[0]
        subject = temp_df["Subject"].values[0]
        trial = temp_df["Trial"].values[0]
        temp_df["TimeStamps"] = temp_df["TimeStamps"] - temp_df["TimeStamps"].min()
        temp_df["TimeStamps"] = temp_df["TimeStamps"].dt.total_seconds() * 1000
        temp_new_dtype = {"TimeStamps": "int32"}
        temp_df = temp_df.astype(temp_new_dtype)
        if with_gyro:
            temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z", 'gyro_x', 'gyro_y', 'gyro_z']]
            temp_df.columns = ["TimeStamps", "X", "Y", "Z", 'GyrX', 'GyrY', 'GyrZ']
        else:
            temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z"]]
            temp_df.columns = ["TimeStamps", "X", "Y", "Z"]
        label_name = labels[labels["UP FALL"] == activity]["ACTIVITY"].values[0]
        is_fall = labels[labels["UP FALL"] == activity]["IS FALL"].values[0]
        label = labels[labels["UP FALL"] == activity]["LABEL"].values[0]
        file_name = f"{dataset_name}_{subject}_{label_name}_{trial}.csv"

        this_meta.append({
            "file_name": str(file_name),
            "activity": "Fall" if is_fall else "ADL",
            "subject": int(subject),
            "trial": int(trial),
            "dataset_label": int(activity),
            "label": int(label),
            "label_name": str(labels.iloc[label, 0]),
            "dataset": dataset_name
        })
        temp_df.to_csv(f"datasets/cleaned/{file_name}", index=False)
        print(file_name)

    df.groupby(["Activity", "Trial", "Subject"]).apply(save_single_file)
    meta_json['files'] += this_meta
    save_meta(meta_json)

def clean_sis_fall(with_gyro):
    path = f"{base_path}/SisFall/SisFall_dataset/"
    folders = os.listdir(path)
    acc1_coe = ((2 * 16) / (2 ** 13))
    acc2_coe = ((2 * 8) / (2 ** 14))
    gyro_coe = ((2 * 2000) / (2 ** 16))

    dataset_name = "SiSFall"
    labels = read_labels()
    meta_json = read_meta()

    for i in range(len(meta_json["files"]) - 1, -1, -1):
        file = meta_json["files"][i]
        if file['dataset'] == dataset_name:
            del meta_json["files"][i]
    for f in glob.glob(f"./datasets/cleaned/{dataset_name}_*.csv"):
        os.remove(f)

    this_meta = []
    for folder in folders:
        if os.path.isdir(f"{path}/{folder}"):
            files = os.listdir(f"{path}/{folder}")
            for file in files:
                if file.endswith(".txt"):
                    parts = file[:-4].split("_")
                    temp_df = pd.read_csv(f"{path}/{folder}/{file}", header=None)
                    temp_df.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc2_x', 'acc2_y',
                                       'acc2_z']
                    temp_df['acc2_z'] = temp_df['acc2_z'].apply(lambda x: int(str(x)[:-1]))
                    temp_df[['acc_x', 'acc_y', 'acc_z']] = temp_df[['acc_x', 'acc_y', 'acc_z']] * acc1_coe
                    temp_df[['acc2_x', 'acc2_y', 'acc2_z']] = temp_df[['acc2_x', 'acc2_y', 'acc2_z']] * acc2_coe
                    temp_df[['gyro_x', 'gyro_y', 'gyro_z']] = temp_df[['gyro_x', 'gyro_y', 'gyro_z']] * gyro_coe
                    temp_df["TimeStamps"] = pd.date_range(start="2017-01-01", periods=temp_df.shape[0], freq="5ms")
                    temp_df["TimeStamps"] = temp_df["TimeStamps"] - temp_df["TimeStamps"].min()
                    temp_df["TimeStamps"] = temp_df["TimeStamps"].dt.total_seconds() * 1000
                    temp_new_dtype = {"TimeStamps": "int16"}
                    temp_df = temp_df.astype(temp_new_dtype)
                    if with_gyro:
                        temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z", 'gyro_x', 'gyro_y', 'gyro_z']]
                        temp_df.columns = ["TimeStamps", "X", "Y", "Z", 'GyrX', 'GyrY', 'GyrZ']
                    else:
                        temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z"]]
                        temp_df.columns = ["TimeStamps", "X", "Y", "Z"]

                    label_name = int(parts[0][1:])
                    if parts[0].startswith("D"):
                        label_name += 15

                    subject = int(parts[1][2:])
                    if parts[1].startswith("SE"):
                        subject += 23
                    is_fall = labels[labels["SIS FALL"] == label_name]["IS FALL"].values[0]
                    label = labels[labels["SIS FALL"] == label_name]["LABEL"].values[0]
                    activity = label_name
                    label_name = labels[labels["SIS FALL"] == label_name]["ACTIVITY"].values[0]
                    trial = int(parts[2][1:])

                    # saving
                    file_name = f"{dataset_name}_{subject}_{label_name}_{trial}.csv"
                    this_meta.append({
                        "file_name": str(file_name),
                        "activity": "Fall" if is_fall else "ADL",
                        "subject": subject,
                        "trial": trial,
                        "dataset_label": int(activity),
                        "label": int(label),
                        "label_name": str(label_name),
                        "dataset": dataset_name
                    })

                    temp_df.to_csv(f"datasets/cleaned/{file_name}", index=False)
                    print(file_name)

    meta_json['files'] += this_meta
    save_meta(meta_json)

def clean_datasets(with_gyro):
    clean_up_fall(with_gyro)
    clean_sis_fall(with_gyro)

def main():
    clean_datasets(True)

if __name__ == '__main__':
    main()