# encoding: utf-8


import os.path as osp
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def do_mosaic(frame, neighbor=8):
    fw, fh = frame.size
    draw = ImageDraw.Draw(frame)
    neighbor = int(neighbor)
    for i in range(0, fh, int(neighbor)): # 关键点0 减去neightbour 防止溢出
        for j in range(0, fw, int(neighbor)):
            r, g, b =frame.getpixel((j, i))
            left_up = (j, i)
            width = min(neighbor, fw - j)
            height = min(neighbor, fh - i)
            right_down = (j + width - 1, i + height - 1) # 关键点2 减去一个像素
            draw.rectangle((left_up, right_down),fill=(r, g, b),outline=None)

class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """
 
    def __init__(self, mean, sigma):
 
        self.mean = mean
        self.sigma = sigma
 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 将图片灰度标准化
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        # 产生高斯 noise
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out*255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('RGB')

class ImageTrainDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, direction = 'AtoB', type='mosaic', radius=8, transform=None, size=[384, 128], interpolation=TF.InterpolationMode.BILINEAR):
        self.dataset = dataset
        self.transform = transform
        self.direction = direction
        self.radius = radius
        self.type = type
        self.size = size
        self.interpolation = interpolation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        
        img_A = read_image(img_path)
        img_A = TF.resize(img_A, self.size, self.interpolation)
        img_B = img_A.copy()
        if self.type == 'mosaic':
            do_mosaic(img_B, self.radius)
        elif self.type == 'blur':
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
        elif self.type == 'noise':
            add_noise = Gaussian_noise(0, self.radius)
            img_B = add_noise(img_B)

        if self.transform is not None:
            img_A, img_B = self.transform(img_A, img_B)

        if self.direction == 'AtoB':
            return img_A, img_B, pid, camid, img_path
        elif self.direction == 'BtoA':
            return img_B, img_A, pid, camid, img_path

class ImageTestDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, direction = 'AtoB', type='mosaic', radius=8, transform=None, size=[384, 128], interpolation=TF.InterpolationMode.BILINEAR):
        self.dataset = dataset
        self.transform = transform
        self.direction = direction
        self.radius = radius
        self.type = type
        self.size = size
        self.interpolation = interpolation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        img_A = read_image(img_path)
        img_A = TF.resize(img_A, self.size, self.interpolation)
        img_B = img_A.copy()

        if self.type == 'mosaic':
            do_mosaic(img_B, self.radius)
        elif self.type == 'blur':
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=self.radius))
        elif self.type == 'noise':
            add_noise = Gaussian_noise(0, self.radius)
            img_B = add_noise(img_B)


        if self.transform is not None:
            img_A, img_B = self.transform(img_A, img_B)

        if self.direction == 'AtoB':
            return img_A, img_B, pid, camid, img_path
        elif self.direction == 'BtoA':
            return img_B, img_A, pid, camid, img_path
