import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, train_window_size, predict_size, is_inference=False):
        self.data = data.values # convert DataFrame to numpy array
        self.train_window_size = train_window_size
        self.predict_size = predict_size
        self.window_size = self.train_window_size + self.predict_size
        self.is_inference = is_inference

    def __len__(self):
        if self.is_inference:
            return len(self.data)
        else:
            return self.data.shape[0] * (self.data.shape[1] - self.window_size - 3)

    def __getitem__(self, idx):
        if self.is_inference:
            # 추론 시
            encode_info = self.data[idx, :4]
            window = self.data[idx, -self.train_window_size:]
            input_data = np.column_stack((np.tile(encode_info, (self.train_window_size, 1)), window))
            return input_data
        else:
            # 학습 시
            row = idx // (self.data.shape[1] - self.window_size - 3)
            col = idx % (self.data.shape[1] - self.window_size - 3)
            encode_info = self.data[row, :4]
            sales_data = self.data[row, 4:]
            window = sales_data[col : col + self.window_size]
            input_data = np.column_stack((np.tile(encode_info, (self.train_window_size, 1)), window[:self.train_window_size]))
            target_data = window[self.train_window_size:]
            return input_data, target_data