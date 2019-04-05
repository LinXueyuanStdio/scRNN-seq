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

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def to_img(x):
    x = x.view(x.size(0), 1, 100, 50)
    return x


def norm(x, reverse=False):
    if reverse:
        y = np.power(10, x) - 1.01
        y = np.around(y).astype(np.int32)
        return y
    else:
        return np.log10(x + 1.01)


def get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path):
    a = pd.read_csv(simulated_csv_data_path)
    b = pd.read_csv(true_csv_data_path)
    for i in range(2000):
        a.iloc[:, i+1] = np.asarray(np.floor(output_data[i][0] * (np.max(a.iloc[:, i+1]))), dtype="int")
    return a, b


def calculate_pcc(arr1, arr2):
    PCC, _ = pearsonr(
        np.asarray(arr1).reshape(2000*5000),
        np.asarray(arr2).reshape(2000*5000))
    return PCC


num_epochs = 10
batch_size = 50
learning_rate = 1e-3
prefix = "BCE_conv2d"
debug = True


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
        a_column_of_simulated_data = np.asarray(a_column_of_simulated_data).reshape(1, 100, 50)
        a_column_of_true_data = np.asarray(a_column_of_true_data).reshape(1, 100, 50)

        a_column_of_simulated_data = a_column_of_simulated_data / np.max(a_column_of_simulated_data)
        a_column_of_true_data = a_column_of_true_data / np.max(a_column_of_true_data)

#         if self.transform is not None:
#             a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
#             a_column_of_true_data = self.transform(a_column_of_true_data)
        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)
        return simulated_true_pack


# Encoder
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),   # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(5, 5)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 7 x 7
            nn.ReLU()
        )

    def forward(self, x):
        global debug
        if debug:
            print(x.size())
        out = self.layer1(x)
        if debug:
            print(out.size())
        out = self.layer2(out)
        if debug:
            print(out.size())
        out = out.view(batch_size, -1)
        if debug:
            print(out.size())
        return out


# Decoder
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),   # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),     # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),     # batch x 32 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),     # batch x 32 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 6, 5, 1, 1),    # batch x 1 x 28 x 28
            nn.ReLU()
        )

    def forward(self, x):
        global debug
        if debug:
            print(x.size())
        out = x.view(batch_size, 256, 10, 5)
        if debug:
            print(out.size())
        out = self.layer1(out)
        if debug:
            print(out.size())
        out = self.layer2(out)
        if debug:
            print(out.size())
        return out


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 5000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def predict(simulated_csv_data_path="./data/counts_simulated_dataset1_dropout0.05.csv",
            true_csv_data_path="./data/true_counts_simulated_dataset1_dropout0.05.csv",
            save_model_filename="./model_dropout0.05.pth", num_epochs=20):
    dataset = SimulatedDataset(simulated_csv_data_path, true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    if os.path.exists(save_model_filename):
        loaded_model = torch.load(save_model_filename, "cpu")
        encoder.load_state_dict(loaded_model)
        decoder.load_state_dict(loaded_model)
    else:
        encoder.train()
        decoder.train()
        for epoch in range(num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, num_epochs))
            prog = Progbar(len(dataloader))
            for i, data in enumerate(dataloader):
                (noisy_data, true_data) = data
                noisy_data = Variable(noisy_data).float().to(device)
                true_data = Variable(true_data).float().to(device)
                # ===================forward=====================
                output = encoder(noisy_data)
                output = decoder(output)
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
                global debug
                debug=False

        torch.save([encoder, decoder], save_model_filename)

    encoder.eval()
    decoder.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, true_data) = data
        noisy_data = Variable(noisy_data).float().to(device)
        true_data = Variable(true_data).float().to(device)
        # ===================forward=====================
        output = encoder(noisy_data)
        output = decoder(output)
        loss = criterion(output, true_data)
        mse = MSE_loss(output, true_data).data
        output_data = output.data.numpy()

        predict_df, true_df = get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path)
        pcc = calculate_pcc(predict_df.iloc[:, 1:], true_df.iloc[:, 1:])

        print("predict PCC:{:.4f} MSE:{:.8f}".format(pcc, mse))

        filepath = "./data/"+prefix+"_predict_PCC_{:.4f}_MSE_{:.8f}_".format(pcc, mse)+simulated_csv_data_path[7:]
        predict_df.to_csv(filepath, index=0)
        break  # 只有一个 batch, 一次全拿出来了，不会有第二个


predict(
    "./data/counts_simulated_dataset1_dropout0.05.csv",
    "./data/true_counts_simulated_dataset1_dropout0.05.csv",
    "./"+prefix+"_model_dropout0.05.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.10.csv",
    "./data/true_counts_simulated_dataset1_dropout0.10.csv",
    "./"+prefix+"_model_dropout0.10.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.15.csv",
    "./data/true_counts_simulated_dataset1_dropout0.15.csv",
    "./"+prefix+"_model_dropout0.15.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.20.csv",
    "./data/true_counts_simulated_dataset1_dropout0.20.csv",
    "./"+prefix+"_model_dropout0.20.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.25.csv",
    "./data/true_counts_simulated_dataset1_dropout0.25.csv",
    "./"+prefix+"_model_dropout0.25.pth"
)
