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

from model import LinearAutoEncoder  # 模型
from util import LinearPackDataset, norm, minmax_0_to_1  # 数据
from util import OutputManager, save_output_data  # 保存文件
from util import calculate_pcc_mse, minmax_noisy_data  # 计算
from util import predict_one_by_one

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 50
learning_rate = 1e-3
output_path = "./output"
model_name = "LinearAutoEncoder"


def predict(output_manager, device, num_epochs=10):
    dataset = LinearPackDataset(output_manager.simulated_csv_data_path, output_manager.true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    model = LinearAutoEncoder().to(device)
    MSE_loss = nn.MSELoss()
    BCE_Loss = nn.BCELoss()
    criterion = MSE_loss
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
                (noisy_data, _) = data
                noisy_data = minmax_noisy_data(noisy_data, device)
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
        torch.save(model.state_dict(), output_manager.model_file_path())

    # 预测、评价
    model.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, _) = data
        noisy_data = minmax_noisy_data(noisy_data, device)
        # ===================forward=====================
        output = model(noisy_data)
        loss = criterion(output, noisy_data)
        # =====================log and save==============
        save_output_data(output, noisy_data, MSE_loss, output_manager)
        break  # 只有一个 batch, 一次全拿出来了，不会有第二个


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
