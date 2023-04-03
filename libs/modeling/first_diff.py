import torch
import torch.nn as nn

class TimeSequence(nn.Module):
    def __init__(self):
        super(TimeSequence, self).__init__()

        self.conv = nn.Sequential(
            # (32,1,12) => (32,32,10), batch-size 保持不变，维度增加到32，长度由于卷积而缩小 (12-k)/s + 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.LeakyReLU(inplace=True),

            # (32,32,10) => (32,64,8), batch-size 保持不变，维度增加到64，长度由于卷积而缩小 (12-k)/s + 1
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.LeakyReLU(inplace=True),

            # (32,64,8) => (32,64,5), batch-size和维度保持不变，自适应池化层处理后长度指定为 5，可将长度不一致的数据统一处理
            nn.AdaptiveMaxPool1d(output_size=5),
        )

        self.flatten = nn.Sequential(
            nn.Linear(in_features=5*64, out_features=10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*5)    # 将维度64和长度5转换到一起，再用全连接层处理
        x = self.flatten(x)
        return x

net = TimeSequence()

# 测试数据，
# 32是batchsize的数量，1 是代表数据维度（比如一场降雨量是一维数据），12代表数据长度（比如这场降雨有12个数据）
a = torch.randn(32,1,12)
b = torch.randn(32,1,9)
c = torch.randn(32,1,6)

# 网络中用到自适应池化层，统一输出长度为5，所以此处的最终输出一致
print('测试输出：')
print(net(a).shape)
print(net(b).shape)
print(net(c).shape)

'''
测试输出：
torch.Size([32, 10])
torch.Size([32, 10])
torch.Size([32, 10])
'''
