# 排序模型

[TOC]

本例子综合了计算图, 样本解析器, 优化器等模块实现了若干个实用排序模型的训练和预测.

本例子希望你通过阅读文档和代码, 掌握如何开发一个完整的深度学习应用.

## trainer使用手册

trainer是单机版训练工具.

### 设置样本解析器

```shell
./trainer --instance_reader=name --instance_reader_config=config
```

参考[样本格式](../../doc/instance.md)和[样本解析器](../../doc/instance_reader.md).

某些模型和样本解析器有耦合, 请配套使用.

### 设置模型

```shell
./trainer --model=name --model_config=config
```

模型名称和配置参考["model\_zoo"](model_zoo).

### 设置优化器

```shell
./trainer --optimizer=name --optimizer_config=config
```

参考[优化器](../../doc/optimizer.md).

### 设置迭代轮数(epoch)

```shell
./trainer --epoch=n
```

n是正整数.

### 设置batch size

```shell
./trainer --batch=n
```

n是正整数.

### 设置训练线程数

```shell
./trainer --thread=n
```

n是正整数.

训练时, 线程以文件粒度调度, 多线程加速的前提是将训练数据切分成多个文件.

#### 选择线程数

线程数不宜超过文件数.

线程数不宜超过CPU核数的2倍.

可以结合具体的硬件条件, 平衡训练速度和训练效果, 找到合适的线程数.

### 设置训练数据

```shell
./trainer --in=in
```

in可以是文件, 也可以是目录. 如果是目录, 其下的所有文件都是训练文件.

文件可以是普通文件, 也可以是gz压缩文件.

文件可以存放在[hdfs](../../doc/hdfs.md).

### 设置对训练文件逆序

```shell
./trainer --reverse_in=1
```

训练前, 对训练文件逆序.

### 设置对训练文件洗牌

```shell
./trainer --shuffle_in=1
```

每轮迭代前, 对训练文件洗牌.

### 设置分片模式

```shell
./trainer --model_shard=n
```

n是0时, 使用无分片模式.

n是正整数时, 使用分片模式, 分片数是n.

| | 无分片模式 | 分片模式 |
| - | - | - |
| 要求 | 参数必须不包含稀疏张量 | 参数必须包含稀疏张量 |
| 参数分片 | 参数只有1份 | 参数被均匀分到n份 |

增量训练无法改变分片模式. 但在分片模式时, 增量训练可以改变分片数.

### 设置输入模型目录

```shell
./trainer --in_model=in
```

从in目录加载模型进行增量学习.

### 设置warmup模型目录

```shell
./trainer --warmup_model=in
```

从in目录加载模型对现有模型warmup.

### 设置删除全0参数

输出模型前, 删除模型参数(和优化器参数)稀疏张量中的全0行.

```shell
./trainer --out_model_remove_zeros=1
```

### 设置输出模型目录

```shell
./trainer --out_model=out
```

### 设置输出文本模型目录

```shell
./trainer --out_text_model=out
```

### 设置输出feature kv模型目录和输出feature kv协议版本

```shell
./trainer --out_feature_kv_model=out --out_feature_kv_protocol_version=n
```

n是2或3.

### 设置verbosity

```shell
./trainer --verbose=n
```

n是非负整数.

n越大, 输出的调试信息越多.

### 设置随机数种子

```shell
./trainer --seed=n
```

n是整数.

### 设置参数时间戳淘汰

每小时(3600秒)训练, 淘汰6小时(21600秒)内模型参数(和优化器参数)稀疏张量中没有更新的行.

```shell
./trainer --ts_enable=1 --ts_now=1600000000 --ts_expire_threshold=21600
./trainer --ts_enable=1 --ts_now=1600003600 --ts_expire_threshold=21600
./trainer --ts_enable=1 --ts_now=1600007200 --ts_expire_threshold=21600
...
```

该功能只在分片模式下生效.

### 设置特征频率过滤

```shell
./trainer --freq_filter_threshold=n
```

n是0时, 不过滤.

n是正整数时, 过滤频率低于n的特征.

该功能只在分片模式下生效.

## predictor使用手册

predictor是单机版预测工具.

### 和trainer相同的参数

以下参数的含义和trainer的完全相同.

```
--instance_reader
--instance_reader_config
--batch
--thread
--in
--in_model
--verbose
```

### 设置输出文件/目录

```shell
./predictor --out_predict=out
```

输出的格式将根据样本解析器的不同略有不同. 输出的典型格式是"标签 预测概率", 例如.

```
1 0.355917
0 0.406134
1 0.405263
0 0.102477
```

## 例子

参考["example"](example).

## dist\_trainer使用手册

dist\_trainer是分布式版训练/预测工具.

分布式训练时, 节点分2个集群: PS(param server)集群和WK(worker)集群.

### PS集群

PS集群中包含CS(coord server)和PS(param server).

CS负责整个训练任务的调度.

PS负责存储参数, 处理参数的拉(pull)/推(push)请求.

分布式训练时, 模型使用分片模式, 1个PS对应1个模型分片, PS数即分片数.

系统中只有1个CS, 第1个PS同时承担了CS的角色.

### WK集群

WK集群中只包含WK(worker).

WK负责处理训练数据, 系统启动时WK数量可以不确定, WK可以随时动态进入/退出系统.

### 和trainer相同的参数

以下参数的含义和trainer的完全相同.

```
--instance_reader
--instance_reader_config
--model
--model_config
--optimizer
--optimizer_config
--epoch
--batch
--in
--reverse_in
--shuffle_in
--in_model
--warmup_model
--out_model_remove_zeros
--out_model
--out_text_model
--out_feature_kv_model
--out_feature_kv_protocol_version
--verbose
--seed
--ts_enable
--ts_now
--ts_expire_threshold
--freq_filter_threshold
```

### 和predictor相同的参数

以下参数的含义和predictor的完全相同.

```
--out_predict
```

### 设置训练或预测

dist\_trainer同时提供了训练和预测的功能.

训练.

```shell
./dist_trainer --sub_command=train
```

预测.

```shell
./dist_trainer --sub_command=predict
```

### 设置节点角色和地址

#### 设置节点为CS

第1个PS(id=0)同时承担了CS的角色.

```shell
./dist_trainer --role=ps --ps_id=0
```

#### 设置节点为PS

```shell
./dist_trainer --role=ps --ps_id=n
```

n是从0开始, 依次递增的整数.

##### 设置PS工作线程数

```shell
./dist_trainer --role=ps --ps_id=n --ps_thread=m
```

m是正整数.

#### 设置节点为WK

```shell
./dist_trainer --role=wk
```

#### 设置PS集群地址

```shell
./dist_trainer \
--cs_addr="10.1.1.1:61000" \
--ps_addrs="10.1.1.1:60000;10.1.1.2:60000;10.1.1.3:60000;10.1.1.4:60000"
```

- 第1个PS监听10.1.1.1:60000和10.1.1.1:61000(做为CD)
- 第2个PS监听10.1.1.2:60000
- 第3个PS监听10.1.1.3:60000
- 第4个PS监听10.1.1.4:60000

### 例子

用4个PS, 若干个WK训练.

```shell
# PS 1
./dist_trainer --sub_command=train --role=ps --ps_id=0 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --out_model=model --model=lr --model_config="sparse=1"
# PS 2
./dist_trainer --sub_command=train --role=ps --ps_id=1 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --out_model=model --model=lr --model_config="sparse=1"
# PS 3
./dist_trainer --sub_command=train --role=ps --ps_id=2 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --out_model=model --model=lr --model_config="sparse=1"
# PS 4
./dist_trainer --sub_command=train --role=ps --ps_id=3 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --out_model=model --model=lr --model_config="sparse=1"
# WK(运行几次就是几个WK)
./dist_trainer --sub_command=train --role=wk \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --out_model=model --model=lr --model_config="sparse=1"
```

用4个PS, 若干个WK预测.

```shell
# PS 1
./dist_trainer --sub_command=predict --role=ps --ps_id=0 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --in_model=model
# PS 2
./dist_trainer --sub_command=predict --role=ps --ps_id=1 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --in_model=model
# PS 3
./dist_trainer --sub_command=predict --role=ps --ps_id=2 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --in_model=model
# PS 4
./dist_trainer --sub_command=predict --role=ps --ps_id=3 \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --in_model=model
# WK(运行几次就是几个WK)
./dist_trainer --sub_command=predict --role=wk \
--cs_addr="127.0.0.1:61000" \
--ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
--in=in --in_model=model
```

## 在线推理

本例子提供静态库的在线推理方案.

### 头文件和核心类

头文件是["model\_server.h"](model_server.h).

核心类是ModelServer.

```c++
using feature_t = std::pair<uint64_t, float>;
using features_t = std::vector<feature_t>;

class ModelServer {
 public:
  ModelServer();
  ~ModelServer();
  ModelServer(const ModelServer&) = delete;
  ModelServer& operator=(const ModelServer&) = delete;

 public:
  // 加载模型文件, 返回是否成功.
  bool Load(const std::string& file);
  // 加载计算图文件, 返回是否成功.
  bool LoadGraph(const std::string& file);
  // 加载模型参数文件, 返回是否成功.
  bool LoadModel(const std::string& file);

 public:
  // 预测1条样本, 返回是否成功.
  // 输出1个预测值.
  bool Predict(const features_t& features, float* prob) const;
  // 预测1条样本, 返回是否成功.
  // 输出n个预测值.
  bool Predict(const features_t& features, std::vector<float>* probs) const;
  // 预测1批样本, 返回是否成功.
  // 输出batch * 1个预测值.
  //
  // 'batch_features'不能为空.
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  // 预测1批样本, 返回是否成功.
  // 输出batch * n个预测值.
  //
  // 'batch_features'不能为空.
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  // 为DTNModel预测1批样本, 返回是否成功.
  // 输出batch * n个预测值.
  //
  // 'batch_item_features'不能为空.
  bool DTNBatchPredict(const features_t& user_features,
                       const std::vector<features_t>& batch_item_features,
                       std::vector<std::vector<float>>* batch_probs) const;

 public:
  using op_context_ptr_t = std::unique_ptr<OpContext, void (*)(OpContext*)>;
  op_context_ptr_t NewOpContext() const;
  // 下面的几个Predict函数和上面的对应.
  // 它们接受'NewOpContext'返回的'OpContext'对象, 它们通常用来复用'OpContext'对象.
  bool Predict(OpContext* op_context, const features_t& features,
               float* prob) const;
  bool Predict(OpContext* op_context, const features_t& features,
               std::vector<float>* probs) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  bool DTNBatchPredict(OpContext* op_context, const features_t& user_features,
                       const std::vector<features_t>& batch_item_features,
                       std::vector<std::vector<float>>* batch_probs) const;
};
```

多线程安全性.

- 多线程调用LoadXXX, 不多线程安全.
- 多线程调用LoadXXX, XXXPredict, 不多线程安全.
- 多线程调用XXXPredict, 多线程安全.

模型更新时, 涉及多线程调用LoadXXX, XXXPredict.
通常采用"双词表"或"加锁"的方式保证多线程安全.

ModelServer的使用参考["model\_server\_demo\_main.cc"](model_server_demo_main.cc).

### 模型文件, 计算图文件和模型参数文件

模型文件, 即ModelServer::Load函数加载的文件.

计算图文件, 即ModelServer::LoadGraph函数加载的文件.

模型参数文件, 即ModelServer::LoadModel函数加载的文件.

它们3者的关系是: 将计算图文件和模型参数文件拼接起来, 就得到模型文件.

```shell
cat [计算图文件] [模型参数文件] > [模型文件]
```

下面介绍几种模式下的计算图文件和模型参数文件.

#### 无分片模式

计算图文件是"graph.bin".

模型参数文件是"model.bin".

```
分片信息文件
shard.bin

计算图文件
graph.bin

模型参数文件
model.bin

优化器参数文件
optimizer.bin

...
```

#### 分片模式, 分片数是1

计算图文件是"graph.bin".

模型参数文件是"model.bin.0".

```
分片信息文件
shard.bin

计算图文件
graph.bin

模型参数文件
model.bin.0

优化器参数文件
optimizer.bin.0

...
```

#### 分片模式, 分片数大于1

计算图文件是"graph.bin".

模型参数文件是"model.bin.x", 模型参数文件的数量是分片数.

```
分片信息文件
shard.bin

计算图文件
graph.bin

模型参数文件
model.bin.0
model.bin.1
model.bin.2
model.bin.3
...

优化器参数文件
optimizer.bin.0
optimizer.bin.1
optimizer.bin.2
optimizer.bin.3
...

...
```

此时, 需要使用merge\_model\_shard工具将众多模型参数文件合并成1个, 才能供ModelServer::LoadModel函数加载.

```shell
./merge_model_shard --in_model=in --out_model=out
```

### 库文件

库文件是"librank.a"和deepx\_core的几个库.

部分库要以whole archive的方式链接.

linux下链接.

```shell
g++ ... \
    -Wl,--whole-archive librank.a -Wl,--no-whole-archive \
    -Wl,--whole-archive libdeepx_core.a -Wl,--no-whole-archive \
    libdeepx_lz4.a \
    libdeepx_z.a
```

其中, "libdeepx\_lz4.a"可以替换成其它liblz4库, "libdeepx\_z.a"可以替换成其它libz库.
