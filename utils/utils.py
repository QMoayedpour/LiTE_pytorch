import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import pandas as pd


def create_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def load_data(file_name, folder_path="./data/"):
    if folder_path:
        path = folder_path
    else:
        path = ""
    path += file_name + "/"

    train_path = path + file_name + "_TRAIN.txt"
    test_path = path + file_name + "_TEST.txt"

    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float64)
    test = np.loadtxt(test_path, delimiter=",", dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest


def znormalisation(x):
    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):
    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def get_mean_col(df):
    moyennes = df.select_dtypes(include="float").mean()
    df.loc["Moyenne"] = moyennes
    df["model"].iloc[-1] = df["model"].iloc[-2]
    return df


def concat_df_on_mean(dataframes):
    """
    Extrait les lignes 'Moyenne' de chaque DataFrame dans une liste et les combine en un seul DataFrame.

    Parameters:
    dataframes (list of DataFrame): Liste de DataFrames

    Returns:
    pd.DataFrame: DataFrame composé des lignes 'Moyenne' de chaque DataFrame
    """
    moyenne_rows = []

    for df in dataframes:
        if "Moyenne" in df.index:
            moyenne_row = df.loc["Moyenne"]
            moyenne_rows.append(moyenne_row)

    combined_df = pd.DataFrame(moyenne_rows)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


class TimeDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.x = x
        self.y = y
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = torch.tensor(self.x[idx], dtype=torch.float32).to(self.device)
        label = torch.tensor(self.y[idx], dtype=torch.float32).to(self.device)
        return sample, label


class SimpleDataset(Dataset):
    def __init__(self, X, y, seq_len, device="cpu", stride=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        if stride > 1:
            self.seq_len = 0
        self.device = device

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float).to(self.device),
            torch.tensor(self.y[idx], dtype=torch.long).to(self.device),
        )
