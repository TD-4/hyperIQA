# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.26
# @github:https://github.com/felixfu520

"""
图片评分工具
"""

from PyQt5 import QtWidgets,QtCore,QtGui
import sys,os


class ImgTag(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 文件夹全局变量
        self.dir_path = ""
        self.img_index_dict = dict()
        self.current_index = 0
        self.current_filename = ""
        self.current_filename_img_num = -1

        self.setWindowTitle("IQA图片标注")
        # 主控件和主控件布局
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # -------------------图像展示控件----------------
        self.imgs_widget = QtWidgets.QWidget()
        self.imgs_layout = QtWidgets.QGridLayout()
        self.imgs_widget.setLayout(self.imgs_layout)

        self.imgs_view = {}
        self.imgs_name = {}
        self.imgs_input = {}
        loc = [(0,0), (0,1), (0,2), (0,3),
               (3,0), (3,1), (3,2), (3,3),
               (6,0), (6,1), (6,2), (6,3)]
        for i in range(0, 12):
            self.imgs_view[str(i)] = QtWidgets.QLabel("图片{}".format(str(i)))  # 标签占位, 或图片view
            self.imgs_view[str(i)].setAlignment(QtCore.Qt.AlignCenter)
            self.imgs_name[str(i)] = QtWidgets.QLabel()  # 图像名称
            self.imgs_input[str(i)] = QtWidgets.QLineEdit()  # 图像标注控件，或文本框
            # self.imgs_input[str(i)].returnPressed.connect(self.next_img_click)  # 回车事件绑定

            self.imgs_layout.addWidget(self.imgs_view[str(i)], loc[i][0], loc[i][1])
            self.imgs_layout.addWidget(self.imgs_name[str(i)], loc[i][0] + 1, loc[i][1])
            self.imgs_layout.addWidget(self.imgs_input[str(i)], loc[i][0] + 2, loc[i][1])
        # --------------------------------------------------------

        # ---------------控制按钮控件-------------------------------
        self.opera_widget = QtWidgets.QWidget()
        self.opera_layout = QtWidgets.QVBoxLayout()
        self.opera_widget.setLayout(self.opera_layout)
        # 各个按钮
        self.select_img_btn = QtWidgets.QPushButton("选择目录")
        self.select_img_btn.clicked.connect(self.select_img_click)
        self.previous_img_btn = QtWidgets.QPushButton("上一张")
        self.previous_img_btn.setEnabled(False)
        self.previous_img_btn.setShortcut('Ctrl+f')
        self.previous_img_btn.clicked.connect(self.previous_img_click)
        self.next_img_btn = QtWidgets.QPushButton("下一张")
        self.next_img_btn.setEnabled(False)
        self.next_img_btn.setShortcut('Ctrl+d')
        self.next_img_btn.clicked.connect(self.next_img_click)
        self.save_img_btn = QtWidgets.QPushButton("保存")
        self.save_img_btn.setEnabled(False)
        self.save_img_btn.setShortcut('Ctrl+s')
        self.save_img_btn.clicked.connect(self.next_img_click)
        # 添加按钮到布局
        self.opera_layout.addWidget(self.select_img_btn)
        self.opera_layout.addWidget(self.previous_img_btn)
        self.opera_layout.addWidget(self.next_img_btn)
        self.opera_layout.addWidget(self.save_img_btn)
        # ----------------------------------------------------

        # ------------将控件添加到主控件布局层--------------------
        self.main_layout.addWidget(self.imgs_widget, 0, 0)
        self.main_layout.addWidget(self.opera_widget, 0, 12)
        # ---------------------------------------------------

        # --------------状态栏--------------------------------
        self.img_total_current_label = QtWidgets.QLabel()
        self.img_total_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.img_total_current_label)
        self.statusBar().addPermanentWidget(self.img_total_label, stretch=0)  # 在状态栏添加永久控件
        # ----------------------------------------------------

        # 设置UI界面核心控件
        self.setCentralWidget(self.main_widget)

    def _reset(self):
        for i in range(0, 12):
            self.imgs_view[str(i)].setText("图片{}".format(str(i)))
            self.imgs_name[str(i)].setText("")
            self.imgs_input[str(i)].setText("")

    def refresh(self):
        # 刷新图片显示部分——实例化12个图像
        all_images = os.listdir(self.current_filename)
        for img_p in all_images:
            image = QtGui.QPixmap(os.path.join(self.current_filename, img_p)).scaled(250, 250)
            # 1、显示图像
            self.imgs_view[img_p[0]].setPixmap(image)
            # 2、显示图片名称、并保存名称
            self.imgs_name[img_p[0]].setText(img_p)  # 显示文件名
            # 3、显示文本框，获取分数
            split_imgp = img_p[:-4].split('-')
            if len(split_imgp) > 3:  # 已经标注过的，获取分数
                score = split_imgp[3]
                if score[0] == 's': score = score[1:]
            else:  # 没有标注过得显示空
                score = ""
            self.imgs_input[img_p[0]].setText(score)
            self.imgs_input['0'].setFocus()  # 获取输入框焦点
            self.imgs_input['0'].selectAll()  # 全选文本

        # 设置状态栏 图片数量信息
        self.img_total_current_label.setText("{}".format(self.current_index + 1))
        self.img_total_label.setText("/{total}".format(total=len(os.listdir(self.dir_path))))

    def checkout_dir(self):
        # 修改文件夹中，1-Z脉冲1630.bmp，命名不正确的情况
        all_images = os.listdir(self.current_filename)
        for img_p in all_images:
            if img_p[0] in set(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")):
                if len(img_p[:-4].split('-')) > 2:
                    pass
                else:
                    img_p_split = img_p[:-4].split('-')
                    new_name = img_p_split[0] + "-" + img_p_split[1][:3] + "-" + img_p_split[1][3:] + ".bmp"
                    os.rename(os.path.join(self.current_filename, img_p),
                              os.path.join(self.current_filename, new_name))
            else:
                os.remove(os.path.join(self.current_filename, img_p))

        # 检查文件夹中所有文件是否满足命名要求
        all_images = os.listdir(self.current_filename)
        for img_p in all_images:
            if len(img_p[:-4].split('-')) > 2:
                pass
            else:
                str_ = self.current_filename + "文件夹中，有文件命名不正确"
                QtWidgets.QMessageBox.information(self, 'Warning', str_, QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                return

    def save_imgs(self):
        # 1、检查评分 & 修改最高分
        best_score_id = ""
        best_score = -1
        for img_p in os.listdir(self.current_filename):
            new_score = self.imgs_input[img_p[0]].text()
            if new_score is None or new_score == "":
                str_ = "评分不能为空"
                QtWidgets.QMessageBox.information(self, 'Error', str_, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
                return False
            try:    # 格式错误
                tmp = int(new_score)
            except Exception as e:
                print(repr(e))
                QtWidgets.QMessageBox.information(self, 'Error', "格式错误", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
                return False
            if int(new_score) < 0 or int(new_score) > 100:
                str_ = "评分应在[0,100]，此次不做修改"
                QtWidgets.QMessageBox.information(self, 'Error', str_, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
                return False
            if best_score < int(new_score): # 找到分数最高的一项
                best_score = int(new_score)
                best_score_id = img_p[0]

        # 2、重命名，保存评分。修改最高一项为sxx-ref
        for img_p in os.listdir(self.current_filename):
            current_img_name = img_p  # 0-Z脉冲-3260.bmp;1-Z脉冲-201590-s100-ref.bmp;0-Z脉冲-192610-s55.bmp
            new_score = self.imgs_input[img_p[0]].text()
            current_img_name_split = current_img_name[:-4].split('-')
            if len(current_img_name_split) <= 2:
                str_ = current_img_name + "\n " + img_p + " \n图片格式错误"
                QtWidgets.QMessageBox.information(self, 'Warning', str_, QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                continue
            if img_p[0] == best_score_id:
                new_img_name = current_img_name_split[0] + "-" + current_img_name_split[1] + "-" + current_img_name_split[
                    2] + "-s" + new_score + "-ref.bmp"
            else:
                new_img_name = current_img_name_split[0] + "-" + current_img_name_split[1] + "-" + \
                               current_img_name_split[2] + "-s" + new_score + ".bmp"
            os.rename(os.path.join(self.dir_path, self.current_filename, current_img_name),
                      os.path.join(self.dir_path, self.current_filename, new_img_name))  # 修改文件名

        return True

    # 选择目录按钮
    def select_img_click(self):
        self._reset()
        try:
            self.dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹')
            dir_list = os.listdir(self.dir_path)
            if len(dir_list) <= 0:
                QtWidgets.QMessageBox.information(self, '提示', '文件夹没有发现图片文件！', QtWidgets.QMessageBox.Ok)
                return

            # 建立“选择目录“下所有文件夹索引
            for i, d in enumerate(dir_list):
                self.img_index_dict[i] = d
            # 当前的文件夹索引
            self.current_index = 0
            # 当前文件夹路径
            self.current_filename = os.path.join(self.dir_path, self.img_index_dict[self.current_index])
            self.setWindowTitle(self.img_index_dict[self.current_index])    # 修改窗口的名称为文件夹

            # 检查当前文件夹中文件是否符合要求
            self.checkout_dir()
            self.current_filename_img_num = len(os.listdir(self.current_filename))
            # 刷新图片显示部分 & 状态栏
            self.refresh()

            # 启用其他按钮
            self.previous_img_btn.setEnabled(True)
            self.next_img_btn.setEnabled(True)
            self.save_img_btn.setEnabled(True)
        except Exception as e:
            print(repr(e))

    # 下一个文件夹
    def next_img_click(self):
        # 保存标注内容
        if self.save_imgs():
            # 判断是否越界
            if self.current_index == len(os.listdir(self.dir_path))-1:
                QtWidgets.QMessageBox.information(self, 'Warning', "已经是最后一张了", QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                return
            # 清空界面缓存
            self._reset()
            # 当前图像索引加1
            self.current_index += 1
            if self.current_index in self.img_index_dict.keys():
                # 当前图片文件路径
                self.current_filename = os.path.join(self.dir_path, self.img_index_dict[self.current_index])
                self.setWindowTitle(self.img_index_dict[self.current_index])
                # 检查当前文件夹中文件是否符合要求
                self.checkout_dir()
                self.current_filename_img_num = len(os.listdir(self.current_filename))
                # 刷新页面
                self.refresh()

    # 上一个文件夹
    def previous_img_click(self):
        # 重命名，保存评分
        if self.save_imgs():
            # 判断是否越界
            if self.current_index == 0:
                QtWidgets.QMessageBox.information(self, 'Warning', "已经是第一张了", QtWidgets.QMessageBox.Yes,
                                                  QtWidgets.QMessageBox.Cancel)
                return
            # 清空界面缓存
            self._reset()
            # 当前图像索引减1
            self.current_index -= 1
            if self.current_index in self.img_index_dict.keys():
                # 当前图片文件路径
                self.current_filename = os.path.join(self.dir_path, self.img_index_dict[self.current_index])
                self.setWindowTitle(self.img_index_dict[self.current_index])
                # 检查当前文件夹中文件是否符合要求
                self.checkout_dir()
                self.current_filename_img_num = len(os.listdir(self.current_filename))
                # 刷新页面
                self.refresh()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ImgTag()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
