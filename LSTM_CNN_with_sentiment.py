#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# === 配置参数 ===
DAYS_FOR_TRAIN = 10
EPOCHS = 200
LR = 1e-3
EPSILON = 1e-8


# === 构造多变量时间序列数据集 ===
def create_multivariate_dataset(data, days):
    data_x, data_y = [], []
    for i in range(len(data) - days):
        x = data[i:i+days]
        y = data[i+days][0]  # 预测Close
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


# === LSTM-CNN 模型定义 ===
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
        lstm_out, _ = self.lstm(x)                      # [batch, seq_len, hidden]
        cnn_input = lstm_out.permute(0, 2, 1)           # [batch, hidden, seq_len]
        conv_out = self.relu(self.conv1d(cnn_input))    # [batch, cnn_out, new_seq_len]
        pooled = self.global_pool(conv_out).squeeze(-1) # [batch, cnn_out]
        out = self.fc(pooled)                           # [batch, output]
        return out


def main():
    # === 加载数据 ===
    df = pd.read_csv("MSFT_sentiment.csv")
    df = df[['Close', 'Sentiment_Score']].iloc[::-1].reset_index(drop=True)

    # === 缺失值填补 ===
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(method='ffill').fillna(0)

    # === 标准化 ===
    close_max, close_min = df['Close'].max(), df['Close'].min()
    sent_max, sent_min = df['Sentiment_Score'].max(), df['Sentiment_Score'].min()

    df['Close'] = (df['Close'] - close_min) / (close_max - close_min + EPSILON)
    df['Sentiment_Score'] = (df['Sentiment_Score'] - sent_min) / (sent_max - sent_min + EPSILON)
    data = df[['Close', 'Sentiment_Score']].values

    # === 构造样本 ===
    dataset_x, dataset_y = create_multivariate_dataset(data, DAYS_FOR_TRAIN)
    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float().view(-1, 1)

    # === 检查是否存在 NaN ===
    if torch.isnan(train_x).any() or torch.isnan(train_y).any():
        raise ValueError("输入或标签中含有 NaN，请检查数据预处理。")

    # === 初始化模型 ===
    model = LSTM_CNN_Model(
        input_size=2,
        lstm_hidden_size=64,
        lstm_layers=2,
        cnn_out_channels=32,
        cnn_kernel_size=3,
        output_size=1
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === 模型训练 ===
    train_loss = []
    for epoch in range(EPOCHS):
        model.train()
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.5f}")

    # === 保存损失图 ===
    plt.plot(train_loss)
    plt.title("LSTM-CNN with Sentiment - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("lstmcnn_sentiment_loss.png")
    plt.close()

    # === 全量预测 ===
    model.eval()
    all_x = torch.from_numpy(dataset_x).float()
    pred_y = model(all_x).detach().numpy().flatten()
    pred_y = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_y))

    real_close = df['Close'].values * (close_max - close_min + EPSILON) + close_min
    pred_close = pred_y * (close_max - close_min + EPSILON) + close_min

    # === 保存结果图 ===
    plt.plot(real_close, label='Real Close')
    plt.plot(pred_close, label='Predicted Close')
    plt.axvline(x=train_size + DAYS_FOR_TRAIN, color='g', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title("LSTM-CNN with Sentiment - Prediction vs Real")
    plt.tight_layout()
    plt.savefig("lstmcnn_sentiment_result.png")
    plt.close()

    print("✅ Finished! 图像保存在 'lstmcnn_sentiment_loss.png' 和 'lstmcnn_sentiment_result.png'")


if __name__ == "__main__":
    main()
