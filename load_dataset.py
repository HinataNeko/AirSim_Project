import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_all_datasets(main_folder_path):
    images_list = []
    labels_list = []

    for file_name in os.listdir(main_folder_path):
        npz_file_path = os.path.join(main_folder_path, file_name)
        data = np.load(npz_file_path)

        images = data['images']
        labels = data['controls'] / 1.

        # Find the first frame where any feature is not zero
        first_non_zero_frame = np.where(labels.any(axis=1))[0][0]
        images = images[first_non_zero_frame:]
        labels = labels[first_non_zero_frame:]

        # 使用卷积函数进行滑动平均
        # window_size = 3
        # smoothed_labels = np.zeros_like(labels)
        # for i in range(labels.shape[1]):
        #     smoothed_labels[:, i] = np.convolve(labels[:, i], np.ones(window_size) / window_size, 'same')

        images_list.append(images)
        labels_list.append(labels)

    return images_list, labels_list


# roll, pitch, thrust, yaw
def plot_labels(data):
    plt.figure(figsize=(16, 8))
    titles = ["Roll", "Pitch", "Thrust", "Yaw"]
    for i in range(4):
        ax = plt.subplot(4, 1, i + 1)
        plt.plot(data[:, i])

        plt.xlim(0, len(data[:, i]))
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Time")
        plt.axhline(0, color='green', linewidth=1.2)  # Add a horizontal line at y=0

        ax.text(-0.05, 0.5, titles[i], va='center', ha='right', transform=ax.transAxes, fontsize=14)

    plt.show()


def play_images_as_video(images_array_rgb, frame_rate=30):
    """
    Play an array of images as a video.

    Args:
    images_array (ndarray): An array of images, uint8 and RGB format.
    frame_rate (int): The frame rate at which the video should be played.
    """
    # Get the shape of the first image to define the window size
    height, width, channels = images_array_rgb[0].shape

    # Create a window to display the video
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', width, height)

    # Play the video
    for img in images_array_rgb:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', img_bgr)
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()


class CustomDataset(Dataset):
    """
    A custom dataset class for handling video frame data with consistent augmentation across all frames in a video.
    """

    def __init__(self, main_folder_path, enhance=True, random_clip=False):
        self.main_folder_path = main_folder_path
        self.enhance = enhance
        self.random_clip = random_clip
        self.all_images, self.all_labels = load_all_datasets(self.main_folder_path)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        """
        Retrieves a video and its corresponding label from the dataset, applying consistent augmentation to all
        frames of the video.

        :param index: Index of the video in the dataset.
        :return: Tuple (x, y), where x is the tensor of augmented frames, and y is the label tensor.
        """
        # 为当前视频生成随机增强参数
        brightness_factor = random.uniform(0.5, 1.2)  # 调整范围
        contrast_factor = random.uniform(0.6, 1.4)  # 调整范围
        saturation_factor = random.uniform(0.6, 1.4)  # 调整范围

        video = self.all_images[index]
        enhanced_frames = []

        # 对每一帧应用相同的随机增强
        if self.enhance:
            for frame in video:
                frame = cv2.resize(frame, (320, 240))
                pil_img = Image.fromarray(frame.astype(np.uint8))

                # 调整亮度
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(brightness_factor)

                # 调整对比度
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(contrast_factor)

                # 调整饱和度
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(saturation_factor)

                # 将PIL图像转换回numpy数组
                enhanced_frame = np.array(pil_img)
                enhanced_frames.append(enhanced_frame)

            # 将增强后的帧堆叠
            x = np.stack(enhanced_frames)
        else:
            x = video
        # play_images_as_video(x)
        x = x.astype(np.float32) / 255.
        x = x.transpose((0, 3, 1, 2))
        y = self.all_labels[index].astype(np.float32)

        if self.random_clip:
            # 随机切片操作
            time_seq = x.shape[0]
            n = 32  # 切片长度最小值

            # 随机选择切片的起始位置
            start = np.random.randint(0, time_seq - n + 1)

            # 进行切片
            x = x[start:start + n, ...]
            y = y[start:start + n, ...]

        return x, y


if __name__ == "__main__":
    all_images, all_labels = load_all_datasets('./datasets/uav_recording')
    # print(np.mean(np.concatenate(all_labels, axis=0), axis=0))
    plot_labels(all_labels[0])
    play_images_as_video(all_images[0])
