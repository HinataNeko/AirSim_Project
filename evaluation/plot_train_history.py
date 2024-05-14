import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cnn_rnn_train_loss = np.load('./results/train_history/CNN_RNN_model_train_loss_history.npy')
cnn_rnn_eval_loss = np.load('./results/train_history/CNN_RNN_model_eval_loss_history.npy')
cnn_lstm_train_loss = np.load('./results/train_history/CNN_LSTM_model_train_loss_history.npy')
cnn_lstm_eval_loss = np.load('./results/train_history/CNN_LSTM_model_eval_loss_history.npy')
cnn_gru_train_loss = np.load('./results/train_history/CNN_GRU_model_train_loss_history.npy')
cnn_gru_eval_loss = np.load('./results/train_history/CNN_GRU_model_eval_loss_history.npy')
cnn_srnn_train_loss = np.load('./results/train_history/CNN_SRNN_T8H128_model_train_loss_history.npy')
cnn_srnn_eval_loss = np.load('./results/train_history/CNN_SRNN_T8H128_model_eval_loss_history.npy')

# np.save('./results/CNN_LSTM_model_train_loss_history.npy', cnn_lstm_train_loss)

plt.figure(figsize=(10, 5))
fontsize=14
colors = ['blue', 'green', 'red', 'purple']

plt.subplot(1, 2, 1)
plt.plot(cnn_rnn_train_loss, label='Naive RNN', color=colors[0], linewidth=1.5)
plt.plot(cnn_lstm_train_loss, label='LSTM', color=colors[1], linewidth=1.5)
plt.plot(cnn_gru_train_loss, label='GRU', color=colors[2], linewidth=1.5)
plt.plot(cnn_srnn_train_loss, label='SRNN', color=colors[3], linewidth=1.5)
plt.title('Training Loss', fontsize=fontsize)
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize)
plt.xlim(0, 100)
plt.ylim(0, 0.8)
plt.legend(fontsize=fontsize, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.75, alpha=0.75, axis='y')

# 隐藏上方和右边的坐标轴
ax = plt.gca()  # 获取当前轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(1, 2, 2)
plt.plot(cnn_rnn_eval_loss, label='Naive RNN', color=colors[0], linewidth=1.5)
plt.plot(cnn_lstm_eval_loss, label='LSTM', color=colors[1], linewidth=1.5)
plt.plot(cnn_gru_eval_loss, label='GRU', color=colors[2], linewidth=1.5)
plt.plot(cnn_srnn_eval_loss, label='SRNN', color=colors[3], linewidth=1.5)
plt.title('Validation Loss', fontsize=fontsize)
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize)
plt.xlim(0, 100)
plt.ylim(0, 0.8)
plt.legend(fontsize=fontsize, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.75, alpha=0.75, axis='y')

# 隐藏上方和右边的坐标轴
ax = plt.gca()  # 获取当前轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()  # 确保子图之间有足够的空间
plt.show()
