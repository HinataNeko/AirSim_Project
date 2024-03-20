import numpy as np
import matplotlib.pyplot as plt

cnn_lstm_train_loss = np.load('./results/CNN_LSTM_model_train_loss_history.npy')
cnn_lstm_eval_loss = np.load('./results/CNN_LSTM_model_eval_loss_history.npy')
cnn_cfc_train_loss = np.load('./results/CNN_CfC_model_train_loss_history.npy')
cnn_cfc_eval_loss = np.load('./results/CNN_CfC_model_eval_loss_history.npy')
cnn_srnn_train_loss = np.load('./results/CNN_SRNN_model_train_loss_history.npy')
cnn_srnn_eval_loss = np.load('./results/CNN_SRNN_model_eval_loss_history.npy')

plt.figure(figsize=(16, 6))  # 宽度为10英寸，高度为5英寸

# np.save('./results/CNN_CfC_model_eval_loss_history.npy', cnn_cfc_eval_loss)
# np.save('./results/CNN_LSTM_model_eval_loss_history.npy', cnn_lstm_eval_loss)

plt.subplot(1, 3, 1)
plt.plot(cnn_lstm_train_loss, label='train')
plt.plot(cnn_lstm_eval_loss, label='val')
plt.title('CNN-LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(cnn_cfc_train_loss, label='train')
plt.plot(cnn_cfc_eval_loss, label='val')
plt.title('CNN-CfC Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(cnn_srnn_train_loss, label='train')
plt.plot(cnn_srnn_eval_loss, label='val')
plt.title('CNN-SpikingRNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.show()
