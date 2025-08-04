#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import time

DAYS_FOR_TRAIN = 10  # 用过去10天预测第11天


def create_dataset(data, days_for_train=10):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


if __name__ == '__main__':
    t0 = time.time()

    print("📥 Step 1: 读取数据...")
    data_close = pd.read_csv('MSFT_price.csv')
    data_close = data_close.astype('float32').values.flatten()  # 保证是一维向量
    print(f"✅ 数据长度: {len(data_close)}")

    print("📊 Step 2: 绘制原始价格图...")
    plt.plot(data_close)
    plt.title("Raw MSFT Prices")
    plt.savefig('xgb_data.png')
    plt.close()

    print("🧮 Step 3: 标准化数据...")
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close_scaled = (data_close - min_value) / (max_value - min_value + 1e-8)

    print("🧱 Step 4: 构造数据集...")
    dataset_x, dataset_y = create_dataset(data_close_scaled, DAYS_FOR_TRAIN)
    dataset_x = dataset_x.reshape(dataset_x.shape[0], -1)  # ✅ 转为二维
    print(f"✅ 特征维度: {dataset_x.shape}, 标签维度: {dataset_y.shape}")

    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    test_x = dataset_x[train_size:]
    test_y = dataset_y[train_size:]
    print(f"✅ 训练样本数: {len(train_x)}, 测试样本数: {len(test_x)}")

    print("📦 Step 5: 转换为 DMatrix 格式...")
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    print("⚙️ Step 6: 设置 XGBoost 参数并训练模型...")
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

    print("🔍 Step 7: 模型预测...")
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    pred_all = np.concatenate([pred_train, pred_test])
    pred_all = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_all))

    assert len(pred_all) == len(data_close_scaled)

    print("📈 Step 8: 绘图展示结果...")
    plt.figure(figsize=(10, 5))
    plt.plot(pred_all, 'r', label='Prediction')
    plt.plot(data_close_scaled, 'b', label='Real')
    plt.axvline(x=train_size + DAYS_FOR_TRAIN, color='g', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title("XGBoost (No Sentiment) - Prediction vs Real")
    plt.xlabel("Days")
    plt.ylabel("Normalized Close Price")
    plt.tight_layout()
    plt.savefig('MSFT_result_xgb.png')
    plt.close()

    t1 = time.time()
    print("⏱️ Step 9: Done! Training time: %.2f mins" % ((t1 - t0) / 60))
