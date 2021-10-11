# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

import torch
import torchvision
from models import model as models
from PIL import Image
import numpy as np
import os
import shutil
import time
from torchcam.cams import SmoothGradCAMpp, CAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
import cv2


import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, conv_layer: str) -> None:
        self.submodule_dict = dict(model.named_modules())
        if conv_layer not in self.submodule_dict.keys():
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")

        self.model = model
        self.hook_handles = []
        self.hook_a, self.hook_g = None, None
        self.hook_handles.append(self.submodule_dict[conv_layer].register_forward_hook(self._hook_a))
        self.hook_handles.append(self.submodule_dict[conv_layer].register_backward_hook(self._hook_g))
        self._hooks_enabled = True

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def _hook_g(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_g = output[0].data

    def clear_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()

    def _backprop(self, scores: Tensor, class_idx: int) -> None:
        """Backpropagate the loss for a specific output class"""
        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""
        self.hook_g: Tensor
        self._backprop(scores, class_idx)
        return self.hook_g.squeeze(0).mean(dim=(1, 2))

    def __call__(self, class_idx: int, scores: Tensor = None, normalized: bool = True) -> Tensor:
        self._precheck(class_idx, scores)
        return self.compute_cams(class_idx, scores, normalized)

    def compute_cams(self, class_idx: int, scores: Tensor = None, normalized: bool = True) -> Tensor:
        weights = self._get_weights(class_idx, scores)
        batch_cams = (weights.view(*weights.shape, 1, 1) * self.hook_a.squeeze(0)).sum(0)
        batch_cams = F.relu(batch_cams, inplace=True)
        if normalized:
            batch_cams = self._normalize(batch_cams)
        return batch_cams

    def _precheck(self, class_idx: int, scores: Tensor = None) -> None:
        """Check for invalid computation cases"""
        if not isinstance(self.hook_a, Tensor):  # Check that forward has already occurred
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        if self.hook_a.shape[0] != 1:  # Check batch size
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")
        if class_idx < 0:  # Check class_idx value
            raise ValueError("Incorrect `class_idx` argument value")
        if not isinstance(scores, torch.Tensor):  # Check scores arg
            raise ValueError("model output scores is required to be passed to compute CAMs")

    @staticmethod
    def _normalize(cams: Tensor) -> Tensor:
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        return cams


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cls(path="", model_pth=""):
    # -----model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model_hyper.train(False)
    # load our pre-trained model on the koniq-10k dataset
    model_hyper.load_state_dict((torch.load(model_pth, map_location=device)))
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
        all_images = [img for img in os.listdir(os.path.join(path, folder)) if img[-4:]==".bmp"]
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

                # # CAM
                # cam_extractor = CAM(model_hyper, target_layer='res.layer4.2.conv2') #, target_layer='res.layer4.2.conv2'
                # restore_transform = torchvision.transforms.Compose([
                #                                         DeNormalize(mean=(0.485, 0.456, 0.406),
                #                                                     std=(0.229, 0.224, 0.225)),torchvision.transforms.ToPILImage()])
                # img_cam = restore_transform(img.squeeze(0))
                # # img_cam = pil_loader(im_path).resize((224, 224))
                # activation_map = cam_extractor(paras['target_in_vec'].cpu().squeeze(0).argmax().item(), paras['target_in_vec'].cpu())
                # result = overlay_mask(img_cam.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)
                # cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                #     os.path.join(im_path[:-4] + ".png"))
            score = np.mean(pred_scores)
            print("predict this images use {} sec".format(time.time() - start_time))
            res_dict[score] = img_p

        select_max = sorted(res_dict.keys(), reverse=True)
        if res_dict[select_max[0]][0] not in set(('1','2','3','4','5','6','7','8','9','0')):
            if res_dict[select_max[1]][0] not in set(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')):
                os.rename(os.path.join(path, folder),
                          os.path.join(path, folder + "----" + res_dict[select_max[2]][:])+ "--" + str(list(res_dict.keys())[2]))
            else:
                os.rename(os.path.join(path, folder), os.path.join(path, folder + "----" + res_dict[select_max[1]][:])+ "--" + str(list(res_dict.keys())[1]))
            continue
        os.rename(os.path.join(path, folder), os.path.join(path, folder + "----" + res_dict[select_max[0]][:])+ "--" + str(list(res_dict.keys())[0]))


if __name__ == "__main__":
    root_ = r"E:\202109反馈-预测\莱宝返回0922"
    for folder_p in os.listdir(root_):
        path = os.path.join(root_, folder_p)
        cls(path=path, model_pth=r"../pretrained/multilevel_0_11.pth")