import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

import pandas as pd
import numpy as np

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')
if not os.path.exists('./filters'):
    os.mkdir('./filters')
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def to_img(x):
    x = x.view(x.size(0), 1, 100, 50)
    return x

num_epochs = 20
batch_size = 128
learning_rate = 1e-3


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def plot_sample_img(img, name):
    img = img.view(1, 100, 50)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)

class SimulatedDataset(Dataset):
    '''
    每一个 Item 是 (5000, ) 的向量
    '''
    def __init__(self, csv_data_path):
        self.csv_data_path = csv_data_path
        self.csv_data = pd.read_csv(csv_data_path)

    def __len__(self):
        return len(self.csv_data.columns) - 1

    def __getitem__(self, index):
        a_column_of_data = self.csv_data.iloc[:, index+1]
        return np.asarray(a_column_of_data)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])
dataset = SimulatedDataset("./data/counts_simulated_dataset1_dropout0.05.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 5000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = img.view(img.size(0), -1)
        noisy_img = add_noise(img.float())
        noisy_img = Variable(noisy_img).float().to(device)
        img = Variable(img).float().to(device)
        # ===================forward=====================
        print(noisy_img.size())
        output = model(noisy_img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss.data,
        MSE_loss.data))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        x_noisy = to_img(noisy_img.cpu().data)
        weights = to_img(model.encoder[0].weight.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        save_image(weights, './filters/epoch_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')