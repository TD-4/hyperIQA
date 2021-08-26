# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

import numpy as np
import os
import time
import argparse
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort


def test_onnx(args):
    # ----------------------------set the device
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('running on device ' + str(device))

    # ----------------------------load the image
    # Dataset used for training the model
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    # cv2 2 PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_BITS2)
    # to tensor
    image = image / 255
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    # normal
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    image[0] = (image[0] - MEAN[0]) / STD[0]
    image[1] = (image[1] - MEAN[1]) / STD[1]
    image[2] = (image[2] - MEAN[2]) / STD[2]

    image = np.expand_dims(image, axis=0)
    # ----------------------------load the model
    sess = ort.InferenceSession(args.model)
    print('Loading onnx...')

    # 模型输入
    input_name = sess.get_inputs()[0].name
    # print("Input name  :", input_name)
    # input_shape = sess.get_inputs()[0].shape
    # print("Input shape :", input_shape)
    # input_type = sess.get_inputs()[0].type
    # print("Input type  :", input_type)

    # 模型输出
    output_name = sess.get_outputs()[0].name
    # print("Output name  :", output_name)
    # output_shape = sess.get_outputs()[0].shape
    # print("Output shape :", output_shape)
    # output_type = sess.get_outputs()[0].type
    # print("Output type  :", output_type)
    output_names = ["target_in_vec", "target_fc1w", "target_fc1b", "target_fc2w", "target_fc2b", "target_fc3w",
                    "target_fc3b",
                    "target_fc4w", "target_fc4b", "target_fc5w", "target_fc5b"]

    # ----------------------------get output
    t3 = time.time()
    result = []
    prediction = sess.run(output_names, {input_name: image})
    t4 = time.time()
    print("Inference time with the ONNX model: {}".format(t4 - t3))
    print("Inference result:", prediction)


if __name__ == "__main__":
    # exporter settings
    parser = argparse.ArgumentParser()
    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./hypernetwork.onnx', help="set model checkpoint path")
    parser.add_argument('--image', type=str, default="./test/D_03.jpg", help='input image to use')

    args = parser.parse_args()
    # print(args)

    # onnx 测试
    test_onnx(args)

