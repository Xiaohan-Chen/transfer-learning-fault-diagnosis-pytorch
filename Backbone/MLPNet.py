import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, num_in = 1024, num_out = 10):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_in,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
            )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.fc5 = nn.Linear(64,num_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x