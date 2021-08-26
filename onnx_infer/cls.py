import torch
import torchvision
import models
from PIL import Image
import numpy as np
import os
import shutil


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cls(path=""):
    # -----model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    # load our pre-trained model on the koniq-10k dataset
    model_hyper.load_state_dict((torch.load('./pretrained/multilevel_0_1.pth')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    # ----images
    all_folders = os.listdir(path)
    for folder in all_folders:
        all_images = os.listdir(os.path.join(path, folder))
        for img_p in all_images:
            im_path = os.path.join(path, folder, img_p)

            # random crop 10 patches and calculate mean quality score
            pred_scores = []
            for i in range(1):
                img = pil_loader(im_path)
                img = transforms(img)
                img = torch.tensor(img.cuda()).unsqueeze(0)
                paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            # quality score ranges from 0-100, a higher score indicates a better quality
            # print('Predicted quality score: %.2f' % score)
            shutil.move(os.path.join(path,folder,img_p), os.path.join(path, folder, "{}.bmp".format(round(score, 4))))


if __name__ == "__main__":
    cls(path="/root/20210502")