# Transformer-PyTorch

## 一、简介

通过PyTorch框架从头实现《Attention Is All Your Need》

## 二、说明

### 2.1 编写测试环境

* Python-3.7.11、PyTorch-1.4.0

### 2.2 运行

* 因为Transformer本身及对应任务决定，当前仅支持通过```torch.distributed.launch```实现的单机多卡GPU训练与验证（多机多卡因为我的设备限制无法测试～）
* 运行train.py即可在模拟数据上进行训练与验证，同test.py（需要指定参数保存路径）
* train.py仅支持GPU，test.py、translating.py支持CPU，如需在多GPU上运行，参考train.py自行修改代码即可。

-----

* 进行训练与验证时，执行命令```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py```
* 上述命令的含义是：在当前主机的0号和3号GPU上进行训练与验证。
* ```CUDA_VISIBLE_DEVICES```决定可见的GPU编号（```nvidia-smi```中的编号），```nproc_per_node```代表对应GPU的数量，如果想要在一块GPU上训练，指定一块即可。

### 2.3 参考链接

* <https://www.tensorflow.org/tutorials/text/transformer>
* <https://pytorch.org/tutorials/beginner/translation_transformer.html>
* <https://github.com/jadore801120/attention-is-all-you-need-pytorch>
* <https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/train_multi_GPU>