# 一、冻结指定模块功能

## 1.1 功能介绍

MindPet支持用户在训练下游任务时，根据 微调算法 或 模块名 冻结网络中部分模块，仅训练不冻结的模块参数。

主要提供以下三个API接口，包括传参和配置文件两种指定模块名的实现方式。

## 1.2 API接口

### freeze_modules

```python
freeze_modules(model, include, exclude)
```

根据指定模块列表冻结网络。_需在定义优化器之前调用_。

**参数：**

- **model**(nn.Cell) - 需要冻结的模型实例。
- **include**(Optional[List[str]]) - 需要冻结的模块名列表， 默认值为None。
    - 模糊匹配列表中所有模块名，挨个将匹配到的模块的`requires_grad`设置为`False`。
    - 列表项支持配置符号**\***，代表任意字符串，格式如` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
    - 如果不配置符号**\***，仅传字符串，表示精确匹配。
- **exclude**(Optional[List[str]]) - 不冻结的模块名列表， 默认值为None。
    - 模糊匹配列表中所有模块名，挨个将匹配到的模块的`requires_grad`设置为`True`。
    - 列表项支持配置符号**\***，代表任意字符串，格式如` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
    - 如果不配置符号**\***，仅传字符串，表示精确匹配。
    - 当`include`和`exclude`列表项冲突时，对该项匹配到的模块不做任何处理。

**异常：**

- **TypeError** - `model`参数类型不是`nn.Cell`。
- **ValueError** - `include`和`exclude`参数同时为空。
- **TypeError** - `include`或`exclude`参数不是非空列表。

**样例：**

```python
from mindpet.graph.freeze_utils import freeze_modules

# 初始化网络结构
model = Network()

# 根据指定模块列表冻结指定模块
freeze_modules(model,
               include=['*embedding*', 'transformer*', 'dense.weight'],
               exclude=['transformer.encoder.blocks.*.layernorm*'])
# 定义优化器
...
```

### freeze_delta

```python
freeze_delta(model, mode, include, exclude)
```

根据微调算法类型以及指定模块列表冻结网络。_需在定义优化器之前调用_。

<font color='orange'>目前已实现lora/prefixtuning/adapter/low_rank_adapter/bitfit等五种微调算法的冻结模式。</font>

**参数：**

- **model**(nn.Cell) - 需要冻结的模型实例。
- **mode**(str) - 微调算法类型。参数值必须是`['lora', 'prefixtuning', 'adapter', 'low_rank_adapter', 'bitfit']`中的一个字符串。
- **include**(Optional[List[str]]) - 需要冻结的模块名列表， 默认值为None。
    - 模糊匹配列表中所有模块名，逐个将匹配到的模块的`requires_grad`设置为`False`。
    - 列表项支持配置符号**\***，代表任意字符串，格式如 ` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
    - 如果不配置符号**\***，仅传字符串，表示精确匹配。
- **exclude**(Optional[List[str]]) - 不冻结的模块名列表， 默认值为None。
    - 模糊匹配列表中所有模块名，挨个将匹配到的模块的`requires_grad`设置为`True`。
    - 列表项支持配置符号**\***，代表任意字符串，格式如` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
    - 如果不配置符号**\***，仅传字符串，表示精确匹配。
    - 当`include`和`exclude`列表项冲突时，对该项匹配到的模块不做任何处理。

**异常：**

- **TypeError** - `model`参数类型不是`nn.Cell`。

- **ValueError** - `mode`参数值不是`['lora', 'prefixtuning', 'adapter', 'low_rank_adapter']`中一个字符串。

- **TypeError** - `include`或`exclude`参数是非空列表。

**样例：**

```python
from mindpet.graph.freeze_utils import freeze_delta

# 初始化网络结构
model = Network()

# 根据微调算法类型以及指定模块列表冻结指定模块
freeze_delta(model,
             mode='lora'
exclude = ['transformer.encoder.blocks.*.layernorm*'])
# 定义优化器
...
```

### freeze_from_config

```python
freeze_from_config(model, config_path)
```

根据配置文件中freeze关键词下的include和exclude列表冻结网络指定模块。_需在定义优化器之前调用_。

**参数：**

- **model**(nn.Cell) - 需要冻结的模型实例。
- **config_path**(str) - 配置文件路径。

  ```
  配置文件必须是yaml格式，内容示例：
  freeze:
    include: ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    exclude: ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
  ```
    - **include**(Optional[List[str]]) - 需要冻结的模块名列表， 默认值为None。
        - 模糊匹配列表中所有模块名，挨个将匹配到的模块的`requires_grad`设置为`False`。
        - 列表项支持配置符号**\***，代表任意字符串，格式如 ` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
        - 如果不配置符号**\***，仅传字符串，表示精确匹配。
    - **exclude**(Optional[List[str]]) - 不冻结的模块名列表， 默认值为None。
        - 模糊匹配列表中所有模块名，挨个将匹配到的模块的`requires_grad`设置为`True`。
        - 列表项支持配置符号**\***，代表任意字符串，格式如` ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']`。
        - 如果不配置符号**\***，仅传字符串，表示精确匹配。
        - 当`include`和`exclude`列表项冲突时，对该项匹配到的模块不做任何处理。

**异常：**

- **TypeError** - `model`参数类型不是`nn.Cell`。
- **TypeError** - `config_path`参数不是字符串。
- **ValueError** - `config_path`参数是空值。
- **ReadYamlFileError** - `config_path`参数对应文件不存在，或是软链接，或读取文件时报错。
- **ModelConfigFreezeInfoError** - 配置文件中没有内容，或没有`freeze`关键词，或`freeze`值不是字典，或`freeze`值中没有`include`或`exclude`关键词。
- **ValueError** - `include`和`exclude`参数同时为空。
- **TypeError** - `include`或`exclude`参数是非空列表。

**样例：**

```python
from mindpet.graph.freeze_utils import freeze_from_config

# 初始化网络结构
model = Network()

# 根据配置文件冻结指定模块
freeze_from_config(model, config_path='test_freeze_config_file.yaml')

# 定义优化器
...
```

# 二、保存可训练参数功能

## 2.1 功能介绍

此功能支持单独保存训练中可更新的参数为ckpt文件，以达到节省存储空间的目的。

## 2.2 API接口

### TrainableParamsCheckPoint

```python
TrainableParamsCheckPoint(directory, prefix, config)
```

继承自mindspore.ModelCheckpoint，实现在训练的过程中保存可训练的参数为ckpt文件。

**参数：**

- **directory**(str) - ckpt文件保存的路径，必须是绝对路径。
- **prefix**(str) - 保存的ckpt文件的前缀名，默认值为"DELTA_CKP"。
- **config**(mindspore.CheckpointConfig) - 保存ckpt的配置，默认值为None。

**异常:**

- **TypeError** - 输入的`config`不为`CheckpointConfig`类或其子类。
- **ValueError** - 输入的`directory`为None或者为空。
- **LinkPathError** - 输入的`directory`路径为软链接路径。
- **AbsolutePathError** - 输入的`directory`路径不是绝对路径。

**样例：**

- **在模型微调时**，从大模型微调工具包中引入`TrainableParamsCheckPoint`类，用法与MindSpore的`ModelCheckpoint`一致，实例化此`callback`后，加入训练时的`callback list`即可，例如：

```python
from mindpet.graph import TrainableParamsCheckPoint
from mindspore import CheckpointConfig

ckpt_config = CheckpointConfig()

## 实例化
params_check_point = TrainableParamsCheckPoint(prefix=prefix,
                                               directory=directory,
                                               config=ckpt_config)

# 加入callback list
callbacks.append(params_check_point)
model.train(callbacks=callbacks)
```

- **在模型评估时**，需要按照以下方案加载预训练的ckpt以及微调后生成的ckpt，分为单卡和多卡场景。

**单卡**

示例代码参见如下，其中checkpoint文件、模型实例需要用户根据实际情况进行替换。

```python
from mindspore import load_checkpoint, load_param_into_net

## 预训练checkpoint文件
pre_trained_ckpt_path = xxx
## 加载预训练参数
pre_trained_pramas = load_checkpoint(pre_trained_ckpt_path)
load_param_into_net(network=net, pre_trained_pramas)

## 微调后生成的checkpoint文件
trainable_ckpt_path = 'xxx'
## 加载微调更新的参数
trainable_pramas = load_checkpoint(trainable_ckpt_path)
load_param_into_net(network=net, trainable_pramas)

# 开始评估
model.eval()
```

**多卡**

示例代码参见如下，其中checkpoint文件列表、分布式策略文件路径、模型实例需要用户根据实际情况进行替换。

```python
from mindspore import load_distributed_checkpoint

## 预训练checkpoint文件列表
pre_trained_ckpt_list = [...]
## 预训练的策略文件
pre_trained_strategy_path = 'xxxxx'
load_distributed_checkpoint(network=net, checkpoint_filenames=pre_trained_ckpt_list,
                            train_strategy_filename=pre_trained_strategy_path)

## 微调后生成的checkpoint文件列表
trainable_ckpt_list = [...]
## 微调后生成的checkpoint文件保存的策略文件
trainable_strategy_path = 'xxxxx'
load_distributed_checkpoint(network=net, checkpoint_filenames=trainable_ckpt_list,
                            train_strategy_filename=trainable_strategy_path)

# 开始评估
model.eval()
```