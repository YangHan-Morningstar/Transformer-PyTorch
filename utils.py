from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import os


def collate_fn(batch_data):
    src_list, tar_inp_list, tar_real_list = [], [], []
    for i in range(len(batch_data)):
        src, tar_inp, tar_real = batch_data[i][0], batch_data[i][1], batch_data[i][2]
        src_list.append(torch.tensor(src))
        tar_inp_list.append(torch.tensor(tar_inp))
        tar_real_list.append(torch.tensor(tar_real))

    src_padding = pad_sequence(src_list, batch_first=True, padding_value=0)
    tar_inp_padding = pad_sequence(tar_inp_list, batch_first=True, padding_value=0)
    tar_real_padding = pad_sequence(tar_real_list, batch_first=True, padding_value=0)

    return src_padding, tar_inp_padding, tar_real_padding


class SelfDataset(Dataset):

    def __init__(self, source_filepath):
        self.source_filepath = source_filepath
        self.data_list = os.listdir(source_filepath)

    def __getitem__(self, item):
        data = np.load("{}/{}".format(self.source_filepath, self.data_list[item]))
        src, tar = data["src"], data["tar"]
        tar_inp, tar_real = tar[:-1], tar[1:]
        return [src, tar_inp, tar_real]

    def __len__(self):
        return len(self.data_list)
