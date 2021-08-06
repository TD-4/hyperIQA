# HyperIQA
⌚️: 2021年4月1日

📚参考
- [Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)
---
## 1. 环境/Dependencies
- Python 3.6+
- PyTorch 0.4+
- TorchVision
- scipy
(optional for loading specific IQA Datasets)
- csv (KonIQ-10k Dataset)
- openpyxl (BID Dataset)

## 2. 代码使用/Usages

###  2.1 文件介绍

- train_test_IQA.py: 训练的主函数，训练从此开始（定义了配置、数据路径等）
- HyperIQASolver.py: 训练的具体过程，train函数是训练的主函数（1、数据加载；2、模型定义；3、迁移学习；4、损失函数；5、优化器和部分冻结；6、训练和测试过程）
- data_loader.py: 数据加载部分，定义了transformer、Dataloader、dataset等内容
- folders.py:   数据集定义部分，定义了Dataset类，即数据的读取
- models.py:    模型定义部分，定义了IQA网络
- models2.py:   模型定义部分，定义了IQA网络，但与models.py的区别是此文件把targetnet融合到了hypernet当中，使得两个网络变成了一个整体
- demo.py:  推理演示部分，读取一张图片，输出iqa评价
- cls.py/cls_cpu.py:    项目需求，对整个文件夹中的图片进行推理。代码逻辑与demo.py一致
- export_onnx.py/onnx_run.py:   export_onnx.py将modes.py中模型导出, 并使用onnx_run.py运行
- export_onnx2.py/onnx_run2.py:   export_onnx2.py将modes2.py中模型导出, 并使用onnx_run2.py运行
- img-score-tools.py:为一个文件夹评分的工具
- img-gen-trainvallist.py:生成数据集的代码
### 2.2 效果展示/Testing a single image

Predicting image quality with our model trained on the Koniq-10k Dataset.

To run the demo, please download the pre-trained model at [Google drive](https://drive.google.com/file/d/1XBN_-fmUrDMm6nZ-Sf60BJGrDs735_s1/view?usp=sharing) or [Baidu cloud](https://pan.baidu.com/s/1yY3O8DbfTTtUwXn14Mtr8Q) (password: 1ty8), put it in 'pretrained' folder, then run:

```
python demo.py
```

You will get a quality score ranging from 0-100, and a higher value indicates better image quality.

### 2.3 训练/Training & Testing on IQA databases

Training and testing our model on the LIVE Challenge Dataset.

```
python train_test_IQA.py
```

Some available options:
* `--dataset`: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq | tid2013.
* `--train_patch_num`: Sampled image patch number per training image.
* `--test_patch_num`: Sampled image patch number per testing image.
* `--batch_size`: Batch size.

When training or testing on CSIQ dataset, please put 'csiq_label.txt' in your own CSIQ folder.
