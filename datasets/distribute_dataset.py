import os
import shutil
import random

# 定义文件夹路径
base_dir = "./yolo"
all_images_dir = os.path.join(base_dir, "all", "images")
all_labels_dir = os.path.join(base_dir, "all", "labels")

# 分配比例
train_ratio = 0.7
val_ratio = 0.2
# 剩余的将被分配到测试集
test_ratio = 1 - train_ratio - val_ratio

# 获取所有图片文件名
all_images = os.listdir(all_images_dir)

# 随机打乱图片顺序
random.shuffle(all_images)

# 计算每个集合的大小
total_images = len(all_images)
train_size = int(total_images * train_ratio)
val_size = int(total_images * val_ratio)
# 剩余的分配给测试集
test_size = total_images - train_size - val_size

# 分配图片到训练集、验证集和测试集
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]


# 定义复制文件的函数
def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)


# 生成并复制图片和标签文件
def distribute_files(image_files, src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir):
    # 生成标签文件名列表
    label_files = [os.path.splitext(image)[0] + '.txt' for image in image_files]

    # 复制图片和标签文件
    copy_files(image_files, src_images_dir, dest_images_dir)
    copy_files(label_files, src_labels_dir, dest_labels_dir)


# 生成文件路径并写入相应的文本文件
def generate_and_write_file_paths(image_files, dataset_type):
    # 将文件路径写入文本文件
    def write_paths_to_file(file_paths, file_name):
        with open(file_name, 'w') as file:
            for path in file_paths:
                file.write(path + '\n')

    file_paths = [f"./images/{dataset_type}/{image}" for image in image_files]
    file_name = os.path.join(base_dir, f"{dataset_type}.txt")
    write_paths_to_file(file_paths, file_name)


def clear_target_directory():
    def _clear_directory(dir_path):
        # 检查目录是否存在
        if os.path.exists(dir_path):
            # 获取目录中的所有文件和子目录
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    # 如果是文件夹，则递归删除
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    # 如果是文件，则直接删除
                    else:
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    # 定义需要清空的目录
    directories_to_clear = [
        "./yolo/images/train",
        "./yolo/images/val",
        "./yolo/images/test",
        "./yolo/labels/train",
        "./yolo/labels/val",
        "./yolo/labels/test"
    ]

    # 清空每个目录
    for dir_path in directories_to_clear:
        _clear_directory(dir_path)

    print("所有指定目录已清空。")


clear_target_directory()

# 调用函数以分配文件和生成路径文本文件
distribute_files(train_images, all_images_dir, all_labels_dir, os.path.join(base_dir, "images", "train"), os.path.join(base_dir, "labels", "train"))
generate_and_write_file_paths(train_images, "train")

distribute_files(val_images, all_images_dir, all_labels_dir, os.path.join(base_dir, "images", "val"), os.path.join(base_dir, "labels", "val"))
generate_and_write_file_paths(val_images, "val")

distribute_files(test_images, all_images_dir, all_labels_dir, os.path.join(base_dir, "images", "test"), os.path.join(base_dir, "labels", "test"))
generate_and_write_file_paths(test_images, "test")

print("文件复制和路径写入完成。")
