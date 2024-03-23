import numpy as np
import matplotlib.pyplot as plt

cnn_rnn_train_loss = np.load('./results/train_history/CNN_RNN_model_train_loss_history.npy')
cnn_rnn_eval_loss = np.load('./results/train_history/CNN_RNN_model_eval_loss_history.npy')
cnn_lstm_train_loss = np.load('./results/train_history/CNN_LSTM_model_train_loss_history.npy')
cnn_lstm_eval_loss = np.load('./results/train_history/CNN_LSTM_model_eval_loss_history.npy')
cnn_gru_train_loss = np.load('./results/train_history/CNN_GRU_model_train_loss_history.npy')
cnn_gru_eval_loss = np.load('./results/train_history/CNN_GRU_model_eval_loss_history.npy')
# cnn_cfc_train_loss = np.load('./results/train_history/CNN_CfC_model_train_loss_history.npy')
# cnn_cfc_eval_loss = np.load('./results/train_history/CNN_CfC_model_eval_loss_history.npy')
cnn_srnn_train_loss = np.load('./results/train_history/CNN_SRNN_T8H128_model_train_loss_history.npy')
cnn_srnn_eval_loss = np.load('./results/train_history/CNN_SRNN_T8H128_model_eval_loss_history.npy')

plt.figure(figsize=(16, 9))

# np.save('./results/CNN_CfC_model_eval_loss_history.npy', cnn_cfc_eval_loss)
# np.save('./results/CNN_LSTM_model_eval_loss_history.npy', cnn_lstm_eval_loss)

plt.subplot(1, 2, 1)
plt.plot(cnn_rnn_train_loss, label='Naive RNN')
plt.plot(cnn_lstm_train_loss, label='LSTM')
plt.plot(cnn_gru_train_loss, label='GRU')
plt.plot(cnn_srnn_train_loss, label='Spiking RNN')
plt.title('训练Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(cnn_rnn_eval_loss, label='Naive RNN')
plt.plot(cnn_lstm_eval_loss, label='LSTM')
plt.plot(cnn_gru_eval_loss, label='GRU')
plt.plot(cnn_srnn_eval_loss, label='Spiking RNN')
plt.title('验证Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.show()
