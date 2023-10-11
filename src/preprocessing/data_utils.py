import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def clean_data(data):
    data['Type 2'].replace('-', pd.NA, inplace=True)
    data.drop_duplicates(subset=['Dex No', 'Name'], keep='first', inplace=True)
    return data


def preprocess_data(data):
    numerical_cols = ['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed', 'BST']
    categorical_cols = ['Type 1', 'Type 2']

    # Normalize the numerical columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Fill NA values in 'Type 2' with a placeholder string
    data['Type 2'].fillna('None', inplace=True)

    # One-hot encode the categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(data[categorical_cols])
    type1_labels = encoder.categories_[0][1:]
    type2_labels = encoder.categories_[1][1:]
    encoded_df = pd.DataFrame(encoded_features, columns=np.concatenate((type1_labels, type2_labels)))

    # Concatenate the processed data
    preprocessed_data = pd.concat([data[numerical_cols], encoded_df], axis=1)

    # Convert to PyTorch tensor
    preprocessed_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float32)

    return preprocessed_tensor, type1_labels, type2_labels, scaler


class PokemonDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
