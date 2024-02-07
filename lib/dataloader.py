# -*- coding: utf-8 -*-
# @Time    : 27/11/2023 15:50
# @Author  : mmai
# @FileName: dataloader
# @Software: PyCharm
import random
from torch.utils.data import Dataset, DataLoader
import torch
from collections import deque
class FlowDataset(Dataset):
    def __init__(self, xs, ys, batch_size):
        self.x_data = torch.tensor(xs, dtype=torch.float32)
        self.y_data = torch.tensor(ys, dtype=torch.float32)
        self.batch_size = batch_size

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

    def __len__(self):
        return len(self.x_data)

class replay_buffer():
    def __init__(self, capacity):
        self.buffer = deque()
        self.capacity = capacity
        self.count = 0

    def add(self, observation, label):  # done: whether the final state, TD error would be different.
        experience = (observation, label)
        if self.count < self.capacity:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self, batch_size):  # return a tuple
        batch = random.sample(self.buffer, batch_size)  # a list [(s,a,r,s), ...]
        return zip(*batch)

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def __len__(self):
        # return len(self.buffer)
        return self.count

if __name__ == '__main__':
    from lib.utils import gen_data_dict, process_sensor_data, generate_insample_dataset

    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = '../sc_sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)
    up = data_dict['../sc_sensor/crossroad1'][:,0,0].reshape(-1,1) # shape (ts, num_nodes)
    down = data_dict['../sc_sensor/crossroad1'][:,1,1].reshape(-1,1) # shape (ts, 1)

    x_train, y_train, x_val, y_val, x_test, y_test = generate_insample_dataset(up, down)
    train_dataset = FlowDataset(x_train, y_train, batch_size=2)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    for i, (x, y) in enumerate(train_dataloader):
        print(x)
        print(y)
        print('*************')