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
from util import to_img, norm, minmax_0_to_1, normalization
from util import get_predict_and_true, calculate_pcc

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


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

        a_column_of_simulated_data = a_column_of_simulated_data / np.max(a_column_of_simulated_data) # 根据最大值来归一化
        a_column_of_true_data = a_column_of_true_data / np.max(a_column_of_true_data)

#         if self.transform is not None:
#             a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
#             a_column_of_true_data = self.transform(a_column_of_true_data)
        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)
        return simulated_true_pack


class Encoder(nn.Module):
    """
    Encoder:
        torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                        stride=1, padding=0, dilation=1,
                        groups=1, bias=True)
        batch x 1 x 100 x 50 -> batch x 12800
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # batch x 32 x 100 x 50
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),   # batch x 32 x 100 x 50
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 100 x 50
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 100 x 50
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(5, 5)   # batch x 64 x 20 x 10
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 10 x 5
            nn.ReLU()
        )

    def forward(self, x):
        global debug
        if debug:
            print(x.size())  # [50, 1, 100, 50]
        out = self.layer1(x)
        if debug:
            print(out.size())  # [50, 64, 20, 10]
        out = self.layer2(out)
        if debug:
            print(out.size())  # [50, 256, 10, 5]
        out = out.view(batch_size, -1)
        if debug:
            print(out.size())  # [50, 12800]
        return out


class Decoder(nn.Module):
    """
    Decoder
        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                stride=1, padding=0, output_padding=0,
                                groups=1, bias=True)
        output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
        batch x 12800 -> batch x 1 x 100 x 50
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # batch x 128 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),   # batch x 128 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # batch x 64 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),     # batch x 64 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),     # batch x 32 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),     # batch x 32 x 20 x 10
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 6, 5, 1, 1),    # batch x 1 x 100 x 50
            nn.ReLU()
        )

    def forward(self, x):
        global debug
        if debug:
            print(x.size())  # [50, 12800]
        out = x.view(batch_size, 256, 10, 5)
        if debug:
            print(out.size())  # [50, 256, 10, 5]
        out = self.layer1(out)
        if debug:
            print(out.size())  # [50, 64, 20, 10]
        out = self.layer2(out)
        if debug:
            print(out.size())  # [50, 1, 100, 50]
        return out


class AutoEncoder(nn.Module):
    """
    encoder:
        torch.Size([50, 1, 100, 50])
        torch.Size([50, 64, 20, 10])
        torch.Size([50, 256, 10, 5])
        torch.Size([50, 12800])
    decoder:
        torch.Size([50, 12800])
        torch.Size([50, 256, 10, 5])
        torch.Size([50, 64, 20, 10])
        torch.Size([50, 1, 100, 50])
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def predict(simulated_csv_data_path="./data/counts_simulated_dataset1_dropout0.05.csv",
            true_csv_data_path="./data/true_counts_simulated_dataset1_dropout0.05.csv",
            save_model_filename="./model_dropout0.05.pth", num_epochs=20):
    dataset = SimulatedDataset(simulated_csv_data_path, true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if os.path.exists(save_model_filename):
        loaded_model = torch.load(save_model_filename, "cpu")
        model.load_state_dict(torch.load(save_model_filename, "cpu"))
    else:
        model.train()
        for epoch in range(num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, num_epochs))
            prog = Progbar(len(dataloader))
            for i, data in enumerate(dataloader):
                (noisy_data, _) = data # 下面只用到 noisy_data 来训练
                noisy_data = Variable(noisy_data).float().to(device)
                # ===================forward=====================
                output = model(noisy_data)
                loss = criterion(output, noisy_data)
                MSE_loss = nn.MSELoss()(output, noisy_data)
                np1 = output.cpu().detach().numpy().reshape(-1)
                np2 = noisy_data.cpu().detach().numpy().reshape(-1)
                PCC, p_value = pearsonr(np1, np2)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =====================log=======================
                prog.update(i + 1, [("loss", loss.item()), ("MSE_loss", MSE_loss.data), ("PCC", PCC), ("p-value", p_value)])
                global debug
                debug = False  # 只打印一次

        torch.save(model.state_dict(), save_model_filename)

    model.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, _) = data
        noisy_data = Variable(noisy_data).float().to(device)
        # ===================forward=====================
        output = model(noisy_data)
        loss = criterion(output, noisy_data)
        mse = MSE_loss(output, noisy_data).data
        output_data = output.data.numpy()

        predict_df, true_df = get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path)
        pcc = calculate_pcc(predict_df.iloc[:, 1:], true_df.iloc[:, 1:])

        print("predict PCC:{:.4f} MSE:{:.8f}".format(pcc, mse))

        filepath = "./data/"+prefix+"_predict_PCC_{:.4f}_MSE_{:.8f}_".format(pcc, mse)+simulated_csv_data_path[7:]
        predict_df.to_csv(filepath, index=0)
        break  # 只有一个 batch, batch_size=2000，一次全拿出来了，不会有第二个


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
