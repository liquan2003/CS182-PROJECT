#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# === 配置参数 ===
DAYS_FOR_TRAIN = 10
EPSILON = 1e-8


# === 构造多变量滑动窗口数据集（确保输出为2维）===
def create_multivariate_dataset(data, days):
    data_x, data_y = [], []
    for i in range(len(data) - days):
        x = data[i:i+days].flatten()  # shape: (days * features,)
        y = data[i+days][0]           # 只预测 Close
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


def main():
    print("📥 读取数据...")
    df = pd.read_csv("MSFT_sentiment.csv")
    df = df[['Close', 'Sentiment_Score']].iloc[::-1].reset_index(drop=True)
    print(f"原始数据行数: {len(df)}")

    print("🧹 处理缺失值...")
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(method='ffill').fillna(0)
    print(f"仍有缺失值: \n{df.isnull().sum()}")

    print("📊 标准化 Close 与 Sentiment...")
    close_max, close_min = df['Close'].max(), df['Close'].min()
    sent_max, sent_min = df['Sentiment_Score'].max(), df['Sentiment_Score'].min()
    df['Close'] = (df['Close'] - close_min) / (close_max - close_min + EPSILON)
    df['Sentiment_Score'] = (df['Sentiment_Score'] - sent_min) / (sent_max - sent_min + EPSILON)

    print("🧱 构造滑动窗口样本...")
    data = df[['Close', 'Sentiment_Score']].values
    dataset_x, dataset_y = create_multivariate_dataset(data, DAYS_FOR_TRAIN)
    print(f"输入维度: {dataset_x.shape}，输出维度: {dataset_y.shape}")

    print("🧪 划分训练集与测试集...")
    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    test_x = dataset_x[train_size:]
    test_y = dataset_y[train_size:]
    print(f"训练样本数: {len(train_x)}，测试样本数: {len(test_x)}")

    print("📦 转换为 DMatrix...")
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    print("⚙️ 设置 XGBoost 参数并开始训练...")
    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    num_round = 200
    model = xgb.train(params, dtrain, num_round)
    print("✅ 模型训练完成。")

    print("🔍 执行预测...")
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    pred_all = np.concatenate([pred_train, pred_test])
    pred_all = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_all))

    print("📈 反标准化并绘图...")
    real_close = df['Close'].values * (close_max - close_min + EPSILON) + close_min
    pred_close = pred_all * (close_max - close_min + EPSILON) + close_min

    plt.plot(real_close, label='Real Close')
    plt.plot(pred_close, label='Predicted Close')
    plt.axvline(x=train_size + DAYS_FOR_TRAIN, color='g', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title("XGBoost with Sentiment - Prediction vs Real")
    plt.tight_layout()
    plt.savefig("xgboost_sentiment_result.png")
    plt.close()
    print("✅ 完成！结果图已保存至 'xgboost_sentiment_result.png'")


if __name__ == "__main__":
    main()
