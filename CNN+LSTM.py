#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import time

DAYS_FOR_TRAIN = 10


def create_dataset(data, days_for_train=5):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers,
                 cnn_out_channels, cnn_kernel_size, output_size=1):
        super(LSTM_CNN_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.conv1d = nn.Conv1d(in_channels=lstm_hidden_size,
                                out_channels=cnn_out_channels,
                                kernel_size=cnn_kernel_size)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cnn_out_channels, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        cnn_input = lstm_out.permute(0, 2, 1)  # [batch, hidden, seq_len]
        conv_out = self.relu(self.conv1d(cnn_input))  # [batch, cnn_out, new_seq]
        pooled = self.global_pool(conv_out).squeeze(-1)  # [batch, cnn_out]
        out = self.fc(pooled)  # [batch, output]
        return out


if __name__ == '__main__':
    t0 = time.time()

    # 读取股价数据
    data_close = pd.read_csv('data/MSFT_price.csv')  # 确保只包含一列价格
    data_close = data_close.astype('float32').values
    plt.plot(data_close)
    plt.title("Raw MSFT Prices")
    plt.savefig('data.png', format='png', dpi=200)
    plt.close()

    # 标准化价格
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    # 构造训练数据
    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)
    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    train_x = train_x.reshape(-1, DAYS_FOR_TRAIN, 1)
    train_y = train_y.reshape(-1, 1)

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()

    # 初始化模型
    model = LSTM_CNN_Model(
        input_size=1,
        lstm_hidden_size=64,
        lstm_layers=2,
        cnn_out_channels=32,
        cnn_kernel_size=3,
        output_size=1
    )

    print("Total Parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # 损失函数与优化器
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    train_loss = []
    for epoch in range(200):
        model.train()
        pred = model(train_x)
        loss = loss_function(pred, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")
        with open('log.txt', 'a+') as f:
            f.write(f"{epoch+1} - {loss.item()}\n")

    plt.figure()
    plt.plot(train_loss)
    plt.title("LSTM-CNN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("MSFT_loss_lstmcnn.png")
    plt.close()

    t1 = time.time()
    print(f"Training time: {(t1 - t0) / 60:.2f} mins")

    # 模型预测（全量数据）
    model.eval()
    dataset_x_all = dataset_x.reshape(-1, DAYS_FOR_TRAIN, 1)
    dataset_x_all = torch.from_numpy(dataset_x_all).float()
    pred_test = model(dataset_x_all).detach().numpy().flatten()
    pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_close, 'b', label='real')
    plt.plot((train_size, train_size), (0, 1), 'g--')
    plt.legend()
    plt.title("LSTM-CNN Prediction vs Real")
    plt.savefig('MSFT_result_lstmcnn.png')
    plt.close()
