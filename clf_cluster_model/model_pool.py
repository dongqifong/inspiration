import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, 1)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, 3, 1, 1)
        self.conv3 = nn.Conv1d(mid_channels, in_channels, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.relu(x_out)

        x_out = self.conv2(x_out)
        x_out = self.relu(x_out)

        x_out = self.conv3(x_out)

        x_out = x_out + x
        return x_out


class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        return x


class ClfCluster(nn.Module):
    # autoencoder
    # loss = crossentropy + mse
    def __init__(self, n_features, input_size, n_class) -> None:
        super().__init__()
        self.n_features = n_features
        self.input_size = input_size
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = PointWiseConv(n_features, n_features)
        self.bn1 = nn.BatchNorm1d(self.n_features)

        self.conv2 = ResBlock(n_features, n_features)
        # self.bn2 = nn.BatchNorm1d(self.n_features)

        self.conv3 = nn.Conv1d(n_features, n_features//2, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(n_features//2)

        self.conv4 = ResBlock(n_features//2, n_features//2)
        # self.bn4 = nn.BatchNorm1d(n_features//2)

        self.conv5 = nn.Conv1d(n_features//2, n_features//4, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(n_features//4)

        self.l = self._get_conv_output_size()
        # print("l", self.l)

        self.conv6 = nn.Conv1d(n_features//4, 30, self.l)
        self.fc1 = nn.Linear(15, 15)
        self.fc2 = nn.Linear(15, 8)
        self.fc3 = nn.Linear(8, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.MaxPool1d(2)(x)
        # print("1:", x.shape)

        x = self.conv2(x)
        x = nn.MaxPool1d(2)(x)
        # print("2:", x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = nn.MaxPool1d(2)(x)
        # print("3:", x.shape)

        x = self.conv4(x)
        x = nn.MaxPool1d(2)(x)
        # print("4:", x.shape)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu(x)
        x = nn.MaxPool1d(2)(x)
        # print("5:", x.shape)

        x = self.conv6(x)
        x = self.leakyrelu(x)
        # print("6:", x.shape)

        x = x.view(-1, x.shape[1]*x.shape[2])  # -1,15

        code = x[:, 15:]
        x = self.dropout(x[:, :15])

        x = self.fc1(x)
        x = self.leakyrelu(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)

        x = self.fc3(x)

        return code, x

    def _get_conv_output_size(self):
        x = torch.randn((2, self.n_features, self.input_size))
        x = self.conv1(x)
        x = nn.MaxPool1d(2)(x)

        x = self.conv2(x)
        x = nn.MaxPool1d(2)(x)

        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = nn.MaxPool1d(2)(x)

        x = self.conv4(x)
        x = nn.MaxPool1d(2)(x)

        x = self.conv5(x)
        x = self.leakyrelu(x)
        x = nn.MaxPool1d(2)(x)
        return x.shape[-1]


if __name__ == "__main__":
    x = torch.randn((2, 60, 1500))
    m = ClfCluster(60, 1500, 5)
    latent, y = m(x)
    print(latent.shape)
