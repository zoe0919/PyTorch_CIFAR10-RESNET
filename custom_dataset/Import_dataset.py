import os

import torch
from torchvision import transforms

from dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

#   数据集所在根目录
root = r"C:\Users\Tingkai_Bao\Desktop\PyTorch_CIFAR10-RESNET\images"



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {

        "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        #   验证集
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])}

    #   创建Dataset
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 128
    #   number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))
    #   打包成batch
    #   collate_fn:
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images, labels = data


if __name__ == '__main__':
    main()