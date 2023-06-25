import os
import random
import shutil


def copyFile(fileDir):
    image_list = os.listdir(fileDir)  # 获取图片的原始路径
    image_number = len(image_list)

    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取比例的图像.
    val_sample = list(set(image_list) - set(train_sample))
    # val_sample = random.sample(sx_sample, int(train_number*0.3))
    sample = [train_sample, val_sample]

    # 复制图像到目标文件夹
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k]):
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k] + '/', name))
        else:
            os.makedirs(save_dir[k])
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k] + '/', name))


if __name__ == '__main__':
    # 原始数据集路径
    origion_path = 'G:/shiyanmoxing/data/new/data/train/'

    # 保存路径
    save_train_dir = 'G:/shiyanmoxing/data/new/data/train/train/'
    save_val_dir = 'G:/shiyanmoxing/data/new/data/train/val/'
    save_dir = [save_train_dir, save_val_dir]

    # 训练集比例
    train_rate = 0.7

    copyFile(origion_path)
    print('划分完毕！')
