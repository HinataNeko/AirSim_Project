import numpy as np
import matplotlib.pyplot as plt

cnn_srnn_train_loss = np.load('./results/CNN_SRNN_model_train_loss_history.npy')
cnn_srnn_eval_loss = np.load('./results/CNN_SRNN_model_eval_loss_history.npy')

plt.figure(figsize=(5, 5))  # 宽度为10英寸，高度为5英寸

plt.plot(cnn_srnn_train_loss, label='train')
plt.plot(cnn_srnn_eval_loss, label='val')
plt.title('CNN-SpikingRNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.8)
plt.legend()
plt.grid()

plt.show()
