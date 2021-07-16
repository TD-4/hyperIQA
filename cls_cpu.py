import torch
import torchvision
import models
from PIL import Image
import numpy as np
import os
import shutil
import time

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cls(path=""):
    # -----model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model_hyper.train(False)
    # load our pre-trained model on the koniq-10k dataset
    model_hyper.load_state_dict((torch.load('multilevel_r.pth', map_location=device)))
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((256, 256)),
        # torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    # ----images
    all_folders = os.listdir(path)
    for folder in all_folders:
        all_images = os.listdir(os.path.join(path, folder))
        if all_images == []:
            print("{} is empty ".format(folder))
            continue
        res_dict = {}
        for img_p in all_images:
            im_path = os.path.join(path, folder, img_p)
            start_time = time.time()
            # random crop 10 patches and calculate mean quality score
            pred_scores = []
            for i in range(1):
                img = pil_loader(im_path)
                img = transforms(img)
                img = torch.tensor(img.to(device=device)).unsqueeze(0)
                paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            print("predict this images use {} sec".format(time.time() - start_time))
            res_dict[score] = img_p

        select_max = sorted(res_dict.keys(), reverse=True)
        if res_dict[select_max[0]][0] not in set(('1','2','3','4','5','6','7','8','9','0')):
            os.rename(os.path.join(path, folder), os.path.join(path, folder + "----" + res_dict[select_max[1]][:10]))
            continue
        os.rename(os.path.join(path, folder), os.path.join(path, folder + "----" + res_dict[select_max[0]][:10]))

if __name__ == "__main__":

    cls(path=r"F:\复检扣小图-0501")