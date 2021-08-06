import os
import shutil
import cv2
import numpy as np
import torchvision
import re
from PIL import Image


def deal_one_group(path=""):
    all_images = os.listdir(path)
    ref_img = None
    img_score_ref = []

    # 寻找ref，并检查每张图片
    for img_p in all_images:
        img_p_split = img_p[:-4].split("-")
        if img_p[0] not in set(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")) or len(img_p_split) < 4:
            print("{} 文件夹下图片命名错误，请检查".format(path))
            return
        if len(img_p_split) >= 5:
            ref_img = img_p

    # ref不存在标注错误
    if ref_img is None:
        print("{} 文件夹下图片命名错误，请检查".format(path))
        return

    for img_p in all_images:
        img_p_split = img_p[:-4].split("-")
        str_ = img_p + "____" + str(float(img_p_split[3][1:])/100) + "____" + ref_img
        img_score_ref.append(str_)

    return img_score_ref


def deal_one_folder(path="", label_path=""):
    all_folder = os.listdir(path)
    label_path = os.path.join(label_path)

    with open(label_path, "a+") as file:
        for folder in all_folder:
            img_score_ref_s = deal_one_group(os.path.join(path, folder))
            for img_score_ref in img_score_ref_s:
                file.write(path.split("\\")[-1]+"____"+folder+"____"+img_score_ref + "\n")


if __name__ == "__main__":
    paths = [r"F:\Data\IQA\multilevel\train\20210423-10层46张",
             r"F:\Data\IQA\multilevel\train\20210423-10层170多张"
             ]
    label_path = r"F:\Data\IQA\multilevel\trainvallist.txt"

    for path in paths:
        deal_one_folder(path=path, label_path=label_path)
