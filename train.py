import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from model import CNN
from data_process import MyDataset

# 定义超参数
EPOCH = 10
BATCH_SIZE = 30
LR = 0.001

# 导入数据
train_data = pd.read_csv('data/train_data.csv', encoding='gb18030',
                         usecols=['way', 'consumption_type', 'amounts_of_money', 'site', 'deal_type',
                                  'consumption_detail']).to_numpy()
test_data = pd.read_csv('data/test_data.csv', encoding='gb18030',
                        usecols=['way', 'consumption_type', 'amounts_of_money', 'site', 'deal_type',
                                 'consumption_detail']).to_numpy()

train_features = torch.tensor(train_data[:, 0:5].astype('float32')).unsqueeze(-1)  # 升维，扩充的那一维代表词向量的维度是1
train_label = torch.tensor(train_data[:, -1].astype('float32')).long()
test_features = torch.tensor(test_data[:, 0:5].astype('float32')).unsqueeze(-1)
test_label = torch.tensor(test_data[:, -1].astype('float32')).long()

train_loader = Data.DataLoader(dataset=MyDataset(train_features, train_label, test_features, test_label),
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=MyDataset(train_features, train_label, test_features, test_label),
                              batch_size=BATCH_SIZE,
                              shuffle=False)

cnn = CNN()
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)


# 训练模型
def train():
    for epoch in range(EPOCH):
        for (train_features, train_label) in train_loader:
            features = Variable(train_features)
            label = Variable(train_label)
            output = cnn(features)
            loss = loss_function(output, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            test_output = cnn(test_features)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_label) / test_label.size(0)
            print('Epoch:', epoch + 1, '|train loss:%.4f' % loss.item(),
                  'test accuracy:%.4f' % accuracy)


def test():
    test_output = cnn(test_features)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_label.numpy(), 'real number')


if __name__ == '__main__':
    train()
    test()
