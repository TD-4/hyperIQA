# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

import argparse
import json
import models
from PIL import Image
import numpy as np

import torch
import torchvision

import os
import shutil
import time

if __name__ == "__main__":

    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./multilevel_r.pth', help="set model checkpoint path")
    parser.add_argument('--model_out1', type=str, default='./hypernetwork_1.8_.onnx')
    parser.add_argument('--model_out2', type=str, default='./targetnetwork.onnx')
    parser.add_argument('--image', type=str, default="./test/D_03.jpg", help='input image to use')

    args = parser.parse_args()
    print(args)

    # ----------------------------set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # ----------------------------load the image
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((256, 256)),
        # torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()])
    if args.image:
        img = Image.open(args.image)
        input = transforms(img).view(1, -1, 224, 224).to(device)
    else:
        pixels = 224
        img = np.random.rand(pixels, pixels, 1)
        input = torch.zeros([1, 1, 224, 224], dtype=torch.float32).to(device)
    print("input size is..", input.shape)

    # ----------------------------load the model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
    model_hyper.train(False)
    # load our pre-trained model on the multilevel dataset
    model_hyper.load_state_dict((torch.load('multilevel_r.pth', map_location=device)))
    print('Loading model_hyper network and weight...')
    paras = model_hyper(input)
    # ----------------------------export the hyper network model
    input_names = ["input"]
    output_names = ["output"]
    # output_names = ["target_in_vec", "target_fc1w", "target_fc1b", "target_fc2w", "target_fc2b", "target_fc3w", "target_fc3b",
    #                 "target_fc4w", "target_fc4b", "target_fc5w", "target_fc5b"]
    print('exporting model_hyper to ONNX...')
    torch.onnx.export(model_hyper, input, args.model_out1, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)
    print('model exported to {:s}'.format(args.model_out1))

    # # ----------------------------load the model
    # paras = model_hyper(input)  # 'paras' contains the network weights conveyed to target network
    # # Building target network
    # model_target = models.TargetNet(paras).cuda()
    # print('Loading model_hyper network and weight...')
    #
    # # ----------------------------export the target network model
    # input_names = ["input_0"]
    # output_names = ["output_0"]
    # print('exporting model_target to ONNX...')
    # v_in = torch.tensor(paras["target_in_vec"], device=device)
    # v_in = v_in.detach()
    # torch.onnx.export(model_target, input_target, args.model_out2, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)
    # print('model exported to {:s}'.format(args.model_out2))
