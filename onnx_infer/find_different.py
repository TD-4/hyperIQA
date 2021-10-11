import os
import shutil


def del_same_folder(path=""):
    all_folders = os.listdir(path)
    count = 0
    for folder in all_folders:
        try:
            if folder.strip().split("---")[1] == folder.strip().split("---")[3][1]:
                print("del folder {}".format(os.path.join(path, folder)))
                count += 1
                shutil.rmtree(os.path.join(path, folder))
        except BaseException:
            pass
    print("total del {} folders\n\n".format(count))


if __name__ == "__main__":
    path = r"F:\test"
    for folder in os.listdir(path):
        del_same_folder(os.path.join(path, folder))