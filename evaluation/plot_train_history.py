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

plt.figure(figsize=(12, 7))  # 使用高分辨率

# 定义一个清晰的颜色方案
colors = ['blue', 'green', 'red', 'purple']

plt.subplot(1, 2, 1)
plt.plot(cnn_rnn_train_loss, label='Naive RNN', color=colors[0], linewidth=2, linestyle=':')
plt.plot(cnn_lstm_train_loss, label='LSTM', color=colors[1], linewidth=2, linestyle='-.')
plt.plot(cnn_gru_train_loss, label='GRU', color=colors[2], linewidth=2, linestyle='--')
plt.plot(cnn_srnn_train_loss, label='Spiking RNN', color=colors[3], linewidth=2, linestyle='-')
plt.title('Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.ylim(0, 0.8)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(cnn_rnn_eval_loss, label='Naive RNN', color=colors[0], linewidth=2, linestyle=':')
plt.plot(cnn_lstm_eval_loss, label='LSTM', color=colors[1], linewidth=2, linestyle='-.')
plt.plot(cnn_gru_eval_loss, label='GRU', color=colors[2], linewidth=2, linestyle='--')
plt.plot(cnn_srnn_eval_loss, label='Spiking RNN', color=colors[3], linewidth=2, linestyle='-')
plt.title('Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.ylim(0, 0.8)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()  # 确保子图之间有足够的空间
plt.show()
