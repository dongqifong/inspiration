from turtle import forward
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, x_in_size: int) -> None:
        super().__init__()
        self.letent_size = x_in_size // 2
        if self.letent_size % 2 != 0:
            self.letent_size = self.letent_size + 1
        self.fc1 = nn.Linear(x_in_size, x_in_size*2)
        self.bn1 = nn.BatchNorm1d(x_in_size*2)
        self.fc2 = nn.Linear(x_in_size*2, x_in_size)
        self.bn2 = nn.BatchNorm1d(x_in_size)
        self.fc3 = nn.Linear(x_in_size, self.letent_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(x)

        x = self.fc3(x)

        y_star = x[:self.letent_size//2]  # (-1,d/4)
        x_star = x[self.letent_size//2:]  # (-1,d/4)
        return x_star, y_star


class Generator(nn.Module):
    def __init__(self, x_star_size: int) -> None:
        super().__init__()
        self.x_star_size = x_star_size
        self.conv1 = nn.Conv1d(2, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, x_star_size*x_star_size, x_star_size, 1, 0)

    def forward(self, x_cat):
        x = self.conv1(x_cat)
        x = self.bn1(x)
        x = nn.LeakyReLU(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(x)

        x = self.conv4(x)
        a_matrix = x.view(-1, self.x_star_size,
                          self.x_star_size)  # [-1,d/4,d/4]
        # [-1,(d/4)*(d/4)]
        code = x.view(-1, self.x_star_size*self.x_star_size)
        return code, a_matrix


class ProjectModel(nn.Module):
    def __init__(self, a_matrix: nn.Module) -> None:
        super().__init__()
        self.a_matrix = a_matrix

    def forward(self, x_star):
        y_star_est = torch.matmul(self.a_matrix, x_star.T)
        return y_star_est  # (-1,d/4)


class Decoder(nn.Module):
    def __init__(self, letent_size, x_ori_size) -> None:
        super().__init__()
        self.x_ori_size = x_ori_size
        self.conv1 = nn.Conv1d(2, 128, 7, 1, 3)
        self.conv2 = nn.Conv1d(128, 64, 7, 1, 3)
        self.conv3 = nn.Conv1d(64, x_ori_size, letent_size//2, 1, 0)

    def forward(self, x_cat):
        x = self.conv1(x_cat)
        x = nn.LeakyReLU(x)

        x = self.conv2(x)
        x = nn.LeakyReLU(x)

        x = self.conv3(x)
        x_reconst = x.view(-1, self.x_ori_size)
        return x_reconst


class LossFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_star_est, y_star, code, x_reconst, x):
        esti_loss = nn.MSELoss()(y_star_est, y_star)
        code_loss = torch.sum(torch.std(code, axis=0))
        reconst_loss = nn.MSELoss()(x_reconst, x)
        return esti_loss + code_loss + reconst_loss
