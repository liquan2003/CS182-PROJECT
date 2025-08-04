**环境准备**
- pandas
- numpy
- torch
- matplotlib
- xgboost
- yfinance
- transformers
- sklearn

**代码运行**

1.数据准备
- code/扒数据和标题.ipynb 得到股价和对应公司数据，但后期这部分数据被下一部分数据替代
- code/计算标题情感参数.ipynb 得到data/AAPL_stock_and_news_enhanced_sentiment.csv also for GOOGL & MSFT
然后我们手动截取了股价数据得到了data/AAPL_PRICE.csv also for GOOGL & MSFT 这部分内容是在excel中手动对data/AAPL_stock_and_news_enhanced_sentiment.csv also for GOOGL & MSFT进行，没有代码。

2.股价预测

所有的py文件均可独立运行

对于无情感因子的部分

[XGBoost.py](code/without_sentiment/XGBoost.py)
[LSTM.py](code/without_sentiment/LSTM.py)
[CNN+LSTM.py](code/without_sentiment/CNN%2BLSTM.py)

对应算法中运行对应公司的纯股价数据data/GOOGL__price.csv also for APPL & MSFT 可以得到对应的loss.csv 和预测图
[GOOGLE_loss.csv](result/without_sentiment/lstm/GOOGL/GOOGLE_loss.csv)
![google_loss.png](result/without_sentiment/lstm/GOOGL/google_loss.png)
![google_result.png](result/without_sentiment/lstm/GOOGL/google_result.png)

对于有情感因子的部分



[LSTM_CNN_with_sentiment.py](code/with_sentiment/LSTM_CNN_with_sentiment.py)
[LSTM_with_sentiment.py](code/with_sentiment/LSTM_with_sentiment.py)
[XGboost_with_sentiment.py](code/with_sentiment/XGboost_with_sentiment.py)

对应算法中运行对应公司的股价+情感因子数据data/GOOGL_stock_and_news_enhanced_sentiment.csv also for APPL & MSFT 可以得到对应的result图和loss图
![lstmcnn_sentiment_result (GOOGL).png](result/with_sentiment/cnn_lstm/lstmcnn_sentiment_result%20%28GOOGL%29.png)
![lstmcnn_sentiment_loss (GOOGL).png](result/with_sentiment/cnn_lstm/lstmcnn_sentiment_loss%20%28GOOGL%29.png)

由于前期代码设计问题，我们的loss数据并未全部写入 loss.csv 部分在log中，部分单独写入loss文件中，最终结果经过手动excel整理

**结果计算**

result/MAE_RMSE.csv 这个结果是我们利用后40位的loss数据计算得出，这部分工作是利用excel完成，没有代码体现。

**结果分析**

result/analysis.md

**report**
[38_report.pdf](result/38_report.pdf)

**contribution**
- [Presented] Yang Jiao ,Yuechen Wu ,Quan Li
- [Data]Yuechen Wu
- [Developed] Yuechen Wu ,Quan Li
- [result analysis ]Yang Jiao ,Quan Li
