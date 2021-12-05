import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from myDataset import MyDataSet
from ViT_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data
import matplotlib.pyplot as plt

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(r'../DataSet/flower_photos')

    data_transform = { # 图像增广
        "train": transforms.Compose([transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    # 实例化数据集
    train_dataset = MyDataSet(image_path = train_images_path,
                              image_class = train_images_label,
                              transform = data_transform["train"])
    val_dataset = MyDataSet(image_path = val_images_path,
                            image_class = val_images_label,
                            transform = data_transform["val"])

    # 转化为迭代器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 8,
                                               shuffle = True,
                                               pin_memory = True,
                                               num_workers = 0,
                                               collate_fn = train_dataset.collate_fn)
    train_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 8,
                                               shuffle = False,
                                               pin_memory = True,
                                               num_workers = 0,
                                               collate_fn = train_dataset.collate_fn)

    #model = create_model(num_classes = 5, has_logits = False).to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type = int, default = 5)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--batch-size', type = int, default = 8)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--lrf', type = float, default = 0.01)

    # 数据所在根目录
    parser.add_argument('--data-path', type = str, default = './DataSet/flower-photos')
    parser.add_argument('--model-name', default = '', help = 'create model name')

    # 预训练权重路径
    parser.add_argument('--weights', type = str, default = '')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type = bool, default = True)
    parser.add_argument('--device', default = 'cuda:0', help = 'device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)