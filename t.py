import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from scipy.stats.stats import pearsonr
from Progbar import Progbar

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
batch_size = 1
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
    return torch.round(tensor.float())

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1))
   # transforms.Lambda(lambda tensor:tensor_round(tensor))
])


class SimulatedDataset(Dataset):
    '''
    每一个 Item 是 (5000, ) 的向量
    '''

    def __init__(self, simulated_csv_data_path, true_csv_data_path, transform=None):
        self.simulated_csv_data = pd.read_csv(simulated_csv_data_path)
        self.true_csv_data_path = pd.read_csv(true_csv_data_path)
        self.transform = transform

    def __len__(self):
        return len(self.simulated_csv_data.columns) - 1

    def __getitem__(self, index):
        a_column_of_simulated_data = self.simulated_csv_data.iloc[:, index+1]
        a_column_of_true_data = self.true_csv_data_path.iloc[:, index+1]
        a_column_of_simulated_data = np.asarray(a_column_of_simulated_data).reshape(1,-1)
        a_column_of_true_data = np.asarray(a_column_of_true_data).reshape(1,-1)
        if self.transform is not None:
            a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
            a_column_of_true_data = self.transform(a_column_of_true_data)
        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)
        return simulated_true_pack

dataset = SimulatedDataset(simulated_csv_data_path="./data/counts_simulated_dataset1_dropout0.05.csv",
                           true_csv_data_path="./data/true_counts_simulated_dataset1_dropout0.05.csv",
                           transform=img_transform)
# dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 5000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

#if os.path.exists('./sim_autoencoder.pth'):
    #model.load_state_dict(torch.load('./sim_autoencoder.pth'))

for epoch in range(num_epochs):
    prog = Progbar(len(dataloader))
    for i, data in enumerate(dataloader):
        (noisy_data, true_data) = data
        noisy_data = Variable(noisy_data).float().to(device)
        true_data = Variable(true_data).float().to(device)
        # ===================forward=====================
        # print(true_data.size())
        output = model(noisy_data)
        loss = criterion(output, true_data)
        MSE_loss = nn.MSELoss()(output, true_data)
        np1 = output.cpu().detach().numpy().reshape(-1)
        np2 = true_data.cpu().detach().numpy().reshape(-1)
        PCC, p_value = pearsonr(np1, np2)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # =====================log=======================
        prog.update(i + 1, [("loss", loss.item()), ("MSE_loss", MSE_loss.data), ("PCC", PCC), ("p-value", p_value)])
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}, PCC:{:.4f}, p-value:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data, MSE_loss.data, PCC, p_value))
    if epoch % 10 == 0:
        x = to_img(true_data.cpu().data)
        x_hat = to_img(output.cpu().data)
        x_noisy = to_img(noisy_data.cpu().data)
        weights = to_img(model.encoder[0].weight.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        save_image(weights, './filters/epoch_{}.png'.format(epoch))

torch.save(model.state_dict(), './512BCE_sim_autoencoder.pth')
