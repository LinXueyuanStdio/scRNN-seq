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
from util.data_process import to_img, norm, minmax_0_to_1, normalization
from util.evaluate import get_predict_and_true, calculate_pcc

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 50
learning_rate = 1e-3
prefix = "BCE_norm"


class SimulatedDataset(Dataset):
    '''
    每一个 Item 是 (1, 5000) 的向量，元素范围[0, ~6]，已 normalization
    '''

    def __init__(self, simulated_csv_data_path, true_csv_data_path, transform=None):
        self.simulated_csv_data = normalization(pd.read_csv(simulated_csv_data_path).iloc[:, 1:])
        self.true_csv_data_path = normalization(pd.read_csv(true_csv_data_path).iloc[:, 1:])
        self.transform = transform

    def __len__(self):
        return len(self.simulated_csv_data.columns)

    def __getitem__(self, index):
        a_column_of_simulated_data = self.simulated_csv_data.iloc[:, index]
        a_column_of_true_data = self.true_csv_data_path.iloc[:, index]

        a_column_of_simulated_data = np.asarray(a_column_of_simulated_data).reshape(1, -1)  # (1, 5000)
        a_column_of_true_data = np.asarray(a_column_of_true_data).reshape(1, -1)

        if self.transform is not None:
            a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
            a_column_of_true_data = self.transform(a_column_of_true_data)
        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)
        return simulated_true_pack


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            # nn.ReLU(True),
            # nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            # nn.Linear(64, 128),
            # nn.ReLU(True),
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
            save_model_filename="./model_dropout0.05.pth", num_epochs=10):
    dataset = SimulatedDataset(simulated_csv_data_path, true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    model = AutoEncoder().to(device)
    MSE_loss = nn.MSELoss()
    BCE_Loss = nn.BCELoss()
    criterion = MSE_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if os.path.exists(save_model_filename):
        model.load_state_dict(torch.load(save_model_filename, "cpu"))
    else:
        model.train()
        for epoch in range(num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, num_epochs))
            prog = Progbar(len(dataloader))
            for i, data in enumerate(dataloader):
                (noisy_data, _) = data
                noisy_data = minmax_0_to_1(noisy_data, False, torch.max(noisy_data))
                noisy_data = Variable(noisy_data).float().to(device)
                # ===================forward=====================
                output = model(noisy_data)
                loss = criterion(output, noisy_data)
                mse = MSE_loss(output, noisy_data).data
                np1 = output.cpu().detach().numpy().reshape(-1)
                np2 = noisy_data.cpu().detach().numpy().reshape(-1)
                PCC, p_value = pearsonr(np1, np2)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =====================log=======================
                prog.update(i + 1, [("loss", loss.item()), ("MSE_loss", mse), ("PCC", PCC), ("p-value", p_value)])
        torch.save(model.state_dict(), save_model_filename)

    model.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, _) = data
        noisy_data = Variable(noisy_data).float().to(device)
        noisy_data = minmax_0_to_1(noisy_data, False, torch.max(noisy_data))
        # ===================forward=====================
        output = model(noisy_data)
        mse = MSE_loss(output, noisy_data).data
        output_data = output.data.numpy()

        predict_df, true_df = get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path)
        pcc = calculate_pcc(predict_df.iloc[:, :], true_df.iloc[:, :])

        print("predict PCC:{:.4f} MSE:{:.8f}".format(pcc, mse))

        # filepath = "./data/"+prefix+"_predict_PCC_{:.4f}_MSE_{:.8f}_".format(pcc, mse)+simulated_csv_data_path[7:]
        # predict_df.to_csv(filepath, index=0)
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
