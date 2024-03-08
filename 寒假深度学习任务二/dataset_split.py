import os
import shutil
from random import seed, sample

# 设定种子确保可复现性
seed(1)

# 划分比例（训练:验证）
train_ratio = 0.8

# 路径设置
dataset_path = 'datasets/full_field/'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# 划分后文件夹路径
train_images_path = os.path.join(dataset_path, 'images/train')
val_images_path = os.path.join(dataset_path, 'images/val')
train_labels_path = os.path.join(dataset_path, 'labels/train')
val_labels_path = os.path.join(dataset_path, 'labels/val')


# 创建文件夹函数
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 创建文件夹
create_dir(train_images_path)
create_dir(val_images_path)
create_dir(train_labels_path)
create_dir(val_labels_path)

# 获取所有图片文件名（不包括扩展名）
image_files = [os.path.splitext(file)[0] for file in os.listdir(images_path) if
               os.path.isfile(os.path.join(images_path, file))]

# 计算训练集的数量
train_size = int(len(image_files) * train_ratio)

# 随机选择训练图片和标签
train_images = set(sample(image_files, train_size))

# 划分数据集
for image_file in image_files:
    # 构造原始和目标路径
    src_image_file = os.path.join(images_path, image_file + '.jpg')
    src_label_file = os.path.join(labels_path, image_file + '.txt')
    if image_file in train_images:
        dest_image_file = os.path.join(train_images_path, image_file + '.jpg')
        dest_label_file = os.path.join(train_labels_path, image_file + '.txt')
    else:
        dest_image_file = os.path.join(val_images_path, image_file + '.jpg')
        dest_label_file = os.path.join(val_labels_path, image_file + '.txt')

    # 复制文件
    shutil.copy(src_image_file, dest_image_file)
    if os.path.exists(src_label_file):
        shutil.copy(src_label_file, dest_label_file)