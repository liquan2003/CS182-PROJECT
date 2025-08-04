#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os

# === 配置参数 ===
DAYS_FOR_TRAIN = 10
EPOCHS = 200
LR = 1e-3
EPSILON = 1e-2


# === 构造多变量时间序列数据集 ===
def create_multivariate_dataset(data, days):
    data_x, data_y = [], []
    for i in range(len(data) - days):
        x = data[i:i+days]
        y = data[i+days][0]  # 预测Close
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


# === LSTM 模型定义 ===
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步
        out = self.fc(out)
        return out


def main():
    # === 加载数据 ===
    df = pd.read_csv("MSFT_sentiment.csv")
    df = df[['Close', 'Sentiment_Score']].iloc[::-1].reset_index(drop=True)

    # === 缺失值填补 ===
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(method='ffill').fillna(0)

    # === 标准化处理，避免除以0 ===
    close_max, close_min = df['Close'].max(), df['Close'].min()
    sent_max, sent_min = df['Sentiment_Score'].max(), df['Sentiment_Score'].min()

    df['Close'] = (df['Close'] - close_min) / (close_max - close_min + EPSILON)
    df['Sentiment_Score'] = (df['Sentiment_Score'] - sent_min) / (sent_max - sent_min + EPSILON)
    data = df[['Close', 'Sentiment_Score']].values

    # === 构造训练数据 ===
    dataset_x, dataset_y = create_multivariate_dataset(data, DAYS_FOR_TRAIN)
    train_size = int(len(dataset_x) * 0.7)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float().view(-1, 1)

    # === NaN 检查 ===
    if torch.isnan(train_x).any() or torch.isnan(train_y).any():
        raise ValueError("输入或标签中含有 NaN，请检查数据预处理。")

    # === 初始化模型 ===
    model = LSTM_Regression(input_size=2, hidden_size=8, output_size=1, num_layers=2)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === 模型训练 ===
    train_loss = []
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.5f}")

    # === 保存损失曲线图 ===
    plt.plot(train_loss)
    plt.title("LSTM with Sentiment - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("lstm_sentiment_loss.png")
    plt.close()

    # === 用全数据预测 ===
    model.eval()
    all_x = torch.from_numpy(dataset_x).float()
    pred_y = model(all_x).detach().numpy().flatten()
    pred_y = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_y))

    real_close = df['Close'].values * (close_max - close_min + EPSILON) + close_min
    pred_close = pred_y * (close_max - close_min + EPSILON) + close_min

    # === 保存预测图 ===
    plt.plot(real_close, label='Real Close')
    plt.plot(pred_close, label='Predicted Close')
    plt.axvline(x=train_size + DAYS_FOR_TRAIN, color='g', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title("LSTM with Sentiment - Prediction vs Real")
    plt.tight_layout()
    plt.savefig("lstm_sentiment_result.png")
    plt.close()

    print("✅ Finished! 图像保存在 'lstm_sentiment_loss.png' 和 'lstm_sentiment_result.png'")


if __name__ == "__main__":
    main()
