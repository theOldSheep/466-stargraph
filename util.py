"""
utils.py
Centralized helper utilities for training, evaluation, and checkpoint management.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy

PATH_CKPT = os.path.join(os.path.dirname(__file__), "ckpt")
PATH_DATA = os.path.join(os.path.dirname(__file__), "data")

def is_subdir(parent, child):
    # convert to abs path
    parent_path = os.path.realpath(parent)
    child_path = os.path.realpath(child)
    return os.path.commonpath([parent_path, child_path]) == parent_path

def get_abs_filepath(base_path, filepath):
    # for rel path, join with base
    if not is_subdir(base_path, filepath):
        filepath = os.path.join(base_path, filepath)
    return filepath



def save_checkpoint(state_dict, filepath):
    abs_path = get_abs_filepath(PATH_CKPT, filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    torch.save(state_dict, abs_path)

def load_checkpoint(filepath, model):
    abs_path = get_abs_filepath(PATH_CKPT, filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Ckpt not found: {filepath} (abs path: {abs_path})")
    
    checkpoint = torch.load(abs_path)
    model.load_state_dict(checkpoint)


class StarGraphDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def load_dataset(filepath, random_seed=42):
    abs_path = get_abs_filepath(PATH_DATA, filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset not found: {filepath} (abs path: {abs_path})")

    with open(abs_path, "r") as data_file:
        data = data_file.read().split('\n')
    
    numpy.random.default_rng(random_seed)
    numpy.random.shuffle(data)
    data_size = len(data)
    train_idx, eval_idx = int(data_size * 0.8), int(data_size * 0.9)
    train_arr = data[:train_idx]
    eval_arr = data[train_idx:eval_idx]
    test_arr = data[eval_idx:]
    return StarGraphDataset(train_arr), StarGraphDataset(eval_arr), StarGraphDataset(test_arr)