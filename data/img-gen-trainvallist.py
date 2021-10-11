# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

import os


def deal_one_group(path=""):
    all_images = os.listdir(path)   # 获得一组/文件夹下所有图片
    ref_img = None  # 存放参考图片
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
    all_folder = os.listdir(path)   # 获得path目录下的所有文件夹/组
    label_path = os.path.join(label_path)   # trainvallist.txt保存路径
    count = 0
    with open(label_path, "a+") as file:
        for folder in all_folder:
            count += 1
            img_score_ref_s = deal_one_group(os.path.join(path, folder))    # 处理每一组图片, 获取一组图片中每张图片的“路径____分数____参考图片”
            for img_score_ref in img_score_ref_s:
                file.write(path.split("\\")[-1]+"____"+folder+"____"+img_score_ref + "\n")
    print("处理 {} 组图片\n".format(count))


if __name__ == "__main__":
    paths = [r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210423-10level",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210803-9level",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210803-BMDT",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210803-CF",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210803-TP",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210813-CF",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210819-9level",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210909-CF",
             r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\20210922-莱宝",
            ]
    label_path = r"D:\DataSets\iqa\hyperIQA\multilevel\trainval\trainvallist.txt"

    for path in paths:
        print("处理 {} 文件夹".format(path))
        deal_one_folder(path=path, label_path=label_path)
