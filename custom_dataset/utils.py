import os
import json
import pickle
import random
import shutil
import collections
import math

import matplotlib.pyplot as plt

Debug = False

def read_txt_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    #   分割成图片名 + label
    tokens = [l.strip().split(' ') for l in lines]
    #   返回字典
    return dict(((name, label) for name, label in tokens))

def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将训练集分割成 训练集 + 测试集"""
    print("开始分割数据集...")
    #   Cifar-10的标签
    classes = {'0':'plane', '1':'car', '2':'bird', '3':'cat', '4':'deer', '5':'dog', '6':'frog', '7':'horse', '8':'ship', '9':'truck'}
    #   支持的文件后缀类型
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    #   存储训练集的所有图片路径
    train_images_path = []
    #   存储训练集图片对应索引信息
    train_images_label = []
    #   存储验证集的所有图片路径
    val_images_path = []
    #   存储验证集图片对应索引信息
    val_images_label = []

    #   训练数据集中示例最少的类别中的示例数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    #   验证集中每个类别的示例数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))

    #   创建字典
    label_count = {}
    counter = 0
    #   遍历图片，分割
    for train_file in os.listdir(data_dir):
        #   判断是不是支持的文件格式
        if os.path.splitext(train_file)[-1] in supported:
            #   获取label，对应字典value
            label = labels[train_file]
            #   获取图片路径
            fname = os.path.join(data_dir, train_file)
            #   将同类型图片放在同一文件夹下
            #   train_valid文件夹用于全部数据的分类整理
            if Debug:
                copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', classes[label]))
            #   放入测试集
            if label not in label_count or label_count[label] < n_valid_per_label:
                if Debug:
                    copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', classes[label]))
                #   获取key对应的value并加一, 如果key值出错, 用0代替
                #   label越界中括号方法会抛出异常，get方法返回None
                label_count[label] = label_count.get(label, 0) + 1
                #   添加验证集图片路径
                val_images_path.append(fname)
                #   添加验证集图片标签
                val_images_label.append(int(label))
            #   放入训练集
            else:
                if Debug:
                    copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', classes[label]))
                #   添加训练集图片路径
                train_images_path.append(fname)
                #   添加训练集图片标签
                train_images_label.append(int(label))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data(root: str, val_rate: float = 0.2):
    #   保证随机结果可复现, 种子号都是0
    #   随机抽样未实现
    random.seed(0)
    #   文件夹不存在则报错
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # # 遍历文件夹，一个文件夹对应一个类别
    # cifar10_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # # 排序，保证顺序一致
    # flower_class.sort()

    #   生成类别名称以及对应的数字索引
    #   返回的字典格式为：{'0.jpg':'0', '1.jpg':'0', ...}
    #   注意拼接前不要加\
    class_indices = read_txt_labels(os.path.join(root, r"clean_label.txt"))
    #   将val和key反向
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    #   存储训练集的所有图片路径
    train_images_path = []
    #   存储训练集图片对应索引信息
    train_images_label = []
    #   存储验证集的所有图片路径
    val_images_path = []
    #   存储验证集图片对应索引信息
    val_images_label = []
    #   存储每个类别的样本总数
    every_class_num = []
    #   支持的文件后缀类型
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    #   分割图像到各文件夹
    train_images_path, train_images_label, val_images_path, val_images_label = reorg_train_valid(os.path.join(root), class_indices, val_rate)
    # # 遍历’train_valid‘文件夹下的文件，包含了训练集和验证集所有图片
    cifar10_class = [cla for cla in os.listdir(os.path.join(root, r'train_valid_test\train_valid')) if os.path.isdir(os.path.join(root, r'train_valid_test\train_valid', cla))]
    for cla in cifar10_class:
        cla_path = os.path.join(root, r'train_valid_test\train_valid', cla)
        #   遍历获取supported支持的所有文件路径
        images = [os.path.join(root, r'train_valid_test\train_valid', cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        #   记录该类别的样本数量
        every_class_num.append(len(images))

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    #   是否开启数据统计
    plot_image = False
    if plot_image:
        #   绘制每种类别个数柱状图
        plt.bar(range(len(cifar10_class)), every_class_num, align='center')
        #   将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(cifar10_class)), cifar10_class)
        #   在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        #   设置x坐标
        plt.xlabel('image class')
        #   设置y坐标
        plt.ylabel('number of images')
        #   设置柱状图的标题
        plt.title('class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list