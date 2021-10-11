import os


path_ = r"E:\202109反馈\莱宝返回0922"
all_folders = os.listdir(path_)
for folder in all_folders:
    all_images = [ img for img in  os.listdir(os.path.join(path_, folder)) if img[-4:]==".bmp"]
    for i, img_p in enumerate(all_images):
        os.rename(os.path.join(path_, folder,img_p),
                  os.path.join(path_,folder,str(i)+'-'+img_p))