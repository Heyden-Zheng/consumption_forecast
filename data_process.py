import numpy as np
import torch
import torch.utils.data as Data


class MyDataset(Data.Dataset):
    # 构造函数
    def __init__(self, train_features, train_label, test_features, test_label):
        super(MyDataset, self).__init__()

        self.train_features = train_features
        self.train_label = train_label
        # self.x_train = self.x_train.astype(float)
        self.test_features = test_features
        self.test_label = test_label

    # 按索引取出对应元素
    def __getitem__(self, index):
        return self.train_features[index], self.train_label[index]

    def __len__(self):
        return self.train_features.shape[0]
