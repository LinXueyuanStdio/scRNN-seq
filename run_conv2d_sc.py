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
from util import norm
from util import calculate_pcc_mse, calculate_pcc
from util import Conv2d_100x50_Dataset
from util import predict_one_by_one
from util import OutputManager, save_output_data  # 保存文件


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


num_epochs = 10
batch_size = 50
learning_rate = 1e-3
output_path = "./output"
model_name = "Conv2dAutoEncoder"
prefix = "BCE_conv2d"
debug = True


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


class Conv2dAutoEncoder(nn.Module):
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
        super(Conv2dAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def predict(output_manager, device, num_epochs=10):
    dataset = Conv2d_100x50_Dataset(output_manager.simulated_csv_data_path, output_manager.true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    model = Conv2dAutoEncoder().to(device)
    criterion = nn.MSELoss()
    MSE_loss = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 训练
    if os.path.exists(output_manager.model_file_path()):
        model.load_state_dict(torch.load(output_manager.model_file_path(), "cpu"))
    else:
        model.train()
        for epoch in range(num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, num_epochs))
            prog = Progbar(len(dataloader))
            for i, data in enumerate(dataloader):
                (noisy_data, _) = data  # 下面只用到 noisy_data 来训练
                noisy_data = Variable(noisy_data).float().to(device)
                # ===================forward=====================
                output = model(noisy_data)
                loss = criterion(output, noisy_data)
                pcc, mse = calculate_pcc_mse(output, noisy_data, MSE_loss)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =====================log=======================
                prog.update(i + 1, [("loss", loss.item()), ("MSE", mse), ("PCC", pcc)])
                global debug
                debug = False  # 只打印一次
        torch.save(model.state_dict(), output_manager.model_file_path())

    model.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, _) = data
        noisy_data = Variable(noisy_data).float().to(device)
        # ===================forward=====================
        output = model(noisy_data)
        loss = criterion(output, noisy_data)
        # =====================log and save==============
        save_output_data(output, noisy_data, MSE_loss, output_manager)
        break  # 只有一个 batch, batch_size=2000，一次全拿出来了，不会有第二个


def predict_with_output_manager(simulated_csv_data_path, true_csv_data_path, model_filename, dropout):
    output_manager = OutputManager(simulated_csv_data_path=simulated_csv_data_path,
                                   true_csv_data_path=true_csv_data_path,
                                   model_filename=model_filename,
                                   output_path=output_path,
                                   model_name=model_name,
                                   dropout=dropout)
    predict(output_manager=output_manager,
            device=device,
            num_epochs=num_epochs)


predict_one_by_one(predict_with_output_manager)
