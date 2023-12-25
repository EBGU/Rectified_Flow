import math
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import random
from scipy.stats import truncnorm
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class Interpolation_Dataset(Dataset): 
    '''
    your folder should have following structrues:
    
    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32],
                 labels = "None"):

        self.root = root
        self.width, self.height = imgSize
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        if labels == "None":
            self.labels = None
            self.cat_num = 0
        else:
            self.labels = np.load(labels)
            self.cat_num = max(self.labels)-min(self.labels)+1 
    def __getitem__(self, index,t=None):
        #index = random.randint(0,9)
        filename = self.imgList[int(index)]
        if self.labels is None: 
            label = 0 #label always from 1
        else:
            label = int(filename.split('.')[0].split('_')[1])
            label = self.labels[label-1]            
        path = os.path.join(self.root,filename)
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height),antialias=False)
        img = img*2 -1 #[xmin, xmax] = [−1, 1]
        if t is None:
            t = np.random.rand()
        noise =torch.tensor(truncnorm.rvs(a=-1,b=1,scale=1,size=tuple(img.shape)))
        #noise = torch.normal(0,1,img.shape)
        noisy_img = t*img+(1-t)*noise
        return noisy_img,img,noise,t,label

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Interpolation_Dataset("/data/protein/OxfordFlowers/train",[64,64],'/home/jiayinjun/Rectified_Flow/OxfordFlowersLabels.npy')
    for i in range(0,2000,200):
        print(i)
        noisy_img,img,noise,t,label = dataset.__getitem__(0,i/2000)
        img = (img + 1)/2
        noisy_img = (noisy_img + 1)/2
        noise = (noise+1)/2
        img,noisy_img,noise = F.to_pil_image(img),F.to_pil_image(noisy_img),F.to_pil_image(noise)
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(img)
        ax2.imshow(noisy_img)
        ax3.imshow(noise)
        plt.savefig('/home/jiayinjun/Rectified_Flow/tmp/test_dataloader.png')