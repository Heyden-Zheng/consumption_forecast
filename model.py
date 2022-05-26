import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 共有10个卷积核，每个卷积核大小为1*1；特征图大小为batch_size*5(30*5*1)
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=1),  #  in_channels特征的维度  out_channels卷积核的数量  kernel_size卷积核的大小
            nn.ReLU(),  # 10*[(1-1/1)+1)] 即10*1
            nn.MaxPool1d(kernel_size=1)  # 10*1
        )
        self.conv2 = nn.Sequential(  # 10*1
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=1),  # 20*[(1-1)/1+1] 即 2*5
            nn.ReLU(),  # 20*[(1-1)/1+1] 即20*1
            nn.MaxPool1d(kernel_size=1)  # 20*1
        )
        self.out = nn.Linear(20*1, 7)  # 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展开成一维
        output = self.out(x)
        return output
