import os
import random
import shutil

# 定义源文件夹和目标文件夹
source_dir = './uav_recording'
test_dir = './test'
train_dir = './train'

# 获取源文件夹下所有的.npy文件
files = os.listdir(source_dir)

m = 5  # 测试数据

# 随机选取m个文件作为测试集
test_files = random.sample(files, m)

# 将选中的文件复制到测试文件夹
for file in test_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

# 将未选中的文件复制到训练文件夹
for file in files:
    if file not in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
