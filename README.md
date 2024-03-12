# MindPet微调算法用户文档



## 一、MindPet简介

MindPet（Pet：Parameter-Efficient Tuning）是属于Mindspore领域的微调算法套件。随着计算算力不断增加，大模型无限的潜力也被挖掘出来。但随之在应用和训练上带来了巨大的花销，导致商业落地困难。因此，出现一种新的参数高效（parameter-efficient）算法，与标准的全参数微调相比，这些算法仅需要微调小部分参数，可以大大降低计算和存储成本，同时可媲美全参微调的性能。


## 二、环境准备

### 2.1 环境依赖

- Python 3.7至3.9版本
- MindSpore >= 1.8



### 2.2 软件安装

在代码仓根目录下运行以下命令，会生成dist文件夹以及whl包：

```shell
python set_up.py bdist_wheel
```

执行以下命令安装whl包：
```shell
pip install dist/mindpet-1.0.4-py3-none-any.whl
```


### 2.3 软件卸载

通过以下命令进行卸载：
```shell
pip uninstall mindpet
```



## 三、微调算法API

**目前MindPet已提供以下六种经典低参微调算法以及一种提升精度的微调算法的API接口，用户可快速适配原始大模型，提升下游任务微调性能和精度；**

| 微调算法           | 算法论文                                                    | 使用说明                                                            |
|----------------| ----------------------------------------------------------- |-----------------------------------------------------------------|
| LoRA           | LoRA: Low-Rank Adaptation of Large Language Models          | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第一章 |
| PrefixTuning   | Prefix-Tuning: Optimizing Continuous Prompts for Generation | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第二章 |
| Adapter        | Parameter-Efficient Transfer Learning for NLP               | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第三章 |
| LowRankAdapter | Compacter: Efficient low-rank hypercom plex adapter layers  | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第四章 |
| BitFit         | BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第五章 |
| R_Drop         | R-Drop: Regularized Dropout for Neural Networks | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第六章 |
| P-Tuning v2    | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks | [MindPet_DeltaAlgorithm_README](doc/MindPet_DeltaAlgorithm_README.md) 第七章 |



## 四、共性图操作API

### 4.1 冻结指定模块功能API

MindPet支持用户根据 微调算法 或 模块名 冻结网络中部分模块，提供调用接口和配置文件两种实现方式。

使用说明参考[MindPet_GraphOperation_README](doc/MindPet_GraphOperation_README.md) 第一章。



### 4.2 保存可训练参数功能API

MindPet支持用户单独保存训练中可更新的参数为ckpt文件，从而节省存储所用的物理资源。

使用说明参考[MindPet_GraphOperation_README](doc/MindPet_GraphOperation_README.md) 第二章。
