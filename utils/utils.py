import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd


def create_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def plot_from_dict(dictionnary, mode="windows"):
    keys_to_exclude = {"n_anomaly", "n_data", "model", "Accuracy"}

    datasets = list(dictionnary.keys())
    metrics = [k for k in dictionnary[datasets[0]].keys() if k not in keys_to_exclude]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        metric_values = [dictionnary[ds][metric] for ds in datasets]
        ax.plot(
            [i] * len(datasets), metric_values, "o", label=metric, c="red", alpha=0.5
        )

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    model_name = dictionnary[datasets[0]]["model"]
    plt.title(f"Metrics for anomaly per {mode}, model pre-trained : {model_name}")
    plt.tight_layout()
    plt.show()


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


class ETTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        mode="train",
        univariate=True,
        scale=True,
        seq_len=336,
        target_window=96,
        split_ratios=[0.6, 0.2, 0.2],
    ):
        super().__init__()
        assert sum(split_ratios) == 1, (
            "Les proportions doivent avoir une somme égale à 1"
        )

        # if univariate:
        #    x_y = df.iloc[:, 1]
        # else:
        #    x_y = df.iloc[:, 1:]
        # time_stamp = df.iloc[:, 0]

        assert mode in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[mode]

        self.seq_len = seq_len
        self.pred_len = target_window

        n = x.shape[0]
        train_size = int(n * split_ratios[0])
        val_size = int(n * split_ratios[1])
        # test_size = n - train_size - val_size

        border1s = [0, train_size, train_size + val_size]
        border2s = [train_size, train_size + val_size, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if scale:
            train_x = x[border1s[0] : border2s[0]]
            self.ss = StandardScaler()
            if univariate:
                train_x = train_x.reshape(-1, 1)

            self.ss.fit(train_x)
            x_y = self.ss.transform(train_x)
        else:
            if univariate:
                x_y = x_y.reshape(-1, 1)
            else:
                x_y = train_x
        # time_stamp = time_stamp.to_numpy()
        self.data_x = x_y[0 : border2 - border1, :].astype(np.float32)
        self.data_y = x_y[0 : border2 - border1, -1].astype(np.float32)
        # self.data_stamp = time_stamp[border1: border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.ss.inverse_transform(data)


class MultiDataset(Dataset):
    def __init__(
        self,
        series_list,
        mode="train",
        univariate=True,
        scale=True,
        seq_len=336,
        target_window=96,
        split_ratios=[0.6, 0.2, 0.2],
    ):
        super().__init__()
        assert sum(split_ratios) == 1, (
            "Les proportions doivent avoir une somme égale à 1"
        )
        assert mode in ["train", "test", "val"], (
            "Mode doit être 'train', 'test' ou 'val'"
        )

        self.seq_len = seq_len
        self.pred_len = target_window
        self.mode = mode
        self.univariate = univariate
        self.scale = scale

        self.data_x = []
        self.data_y = []

        for series in series_list:
            self.process_series(series, split_ratios)

        self.data_x = np.concatenate(self.data_x, axis=0).astype(np.float32)
        self.data_y = np.concatenate(self.data_y, axis=0).astype(np.float32)

    def process_series(self, x, split_ratios):
        n = x.shape[0]
        train_size = int(n * split_ratios[0])
        val_size = int(n * split_ratios[1])
        test_size = n - train_size - val_size

        if self.mode == "train":
            data_x = x[:train_size]
        elif self.mode == "val":
            data_x = x[train_size : train_size + val_size]
        else:  # self.mode == 'test'
            data_x = x[train_size + val_size :]

        if self.scale:
            ss = StandardScaler()
            if self.univariate:
                data_x = data_x.reshape(-1, 1)
            data_x = ss.fit_transform(data_x)

        if self.univariate and not self.scale:
            data_x = data_x.reshape(-1, 1)

        self.data_x.append(data_x)
        self.data_y.append(data_x[:, -1])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.ss.inverse_transform(data)
