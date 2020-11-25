# 计算图

[TOC]

计算图在实现上分2层.

- 表示层.
  - Graph.
    - 主要由**节点**(GraphNode的子类)组成.
    - 表示节点连接关系.
- 实现层.
  - OpContext.
    - 主要由**算子**(Op的子类)组成.
    - 进行实际的计算.

![arch](pic/graph_arch.png)

## 表示层

Graph是表示层的核心数据结构, 它由节点组成.

顾名思义, Graph和节点仅仅表示计算, 而不进行计算.

### 表示层开发

所谓"开发模型", 主要就是开发表示层, 即用节点像搭积木一样搭出Graph.

内置节点的使用参考[算子使用手册](op_manual.md).

内置节点/算子可以满足绝大部分需求. 如果无法满足, 参考[算子开发](op_dev.md).

表示层开发要确定.

- 网络结构.
- 用于离线训练的损失函数.
- 用于在线推理的1个或多个目标.

#### 例子

参考[排序模型的"model\_zoo"](../example/rank/model_zoo).

#### 形状推理

形状推理(shape inference), 给定节点和它的所有输入节点, 推理节点的形状.

由节点参数非法导致的形状推理失败, 将会抛出异常.

其它原因导致的形状推理失败, 将被忽略. 这种失败不一定会导致实现层失败. 例如, 节点的形状带有"-1"通配符, 表示batch轴.

##### 例子

```c++
// 不借助形状推理
auto* lin = WideGroupEmbeddingLookup(...);
auto* quad = DeepGroupEmbeddingLookup(...);
auto* deep = StackedFullyConnect(...);
auto* Z1 = Concat("Z1", {lin, quad, deep});
// 手动推理lin, quad, deep的形状
int lin_col = ...;
int quad_col = ...;
int deep_col = ...;
// 手动推理Z1的形状
int Z1_col = lin_col + quad_col + deep_col;
auto* W = GetVariable("W", Shape(Z1_col, 32), ...);
auto* Z2 = GEMM("Z2", Z1, W, 0, 0);
```

可见, 不借助形状推理, 首先要手动推理lin, quad, deep的形状, 再手动推理Z1的形状, 最后使用Z1的形状.

手动推理繁琐, 容易出错.

```c++
// 借助形状推理
auto* lin = WideGroupEmbeddingLookup(...);
// 此时lin->shape()可用
auto* quad = DeepGroupEmbeddingLookup(...);
// 此时quad->shape()可用
auto* deep = StackedFullyConnect(...);
// 此时deep->shape()可用
auto* Z1 = Concat("Z1", {lin, quad, deep});
// 此时Z1->shape()可用
auto* W = GetVariable("W", Shape(Z1->shape()[1], 32), ...);
auto* Z2 = GEMM("Z2", Z1, W, 0, 0);
```

显然, 借助形状推理, 可以写出更紧凑的Graph.

## 实现层

Graph中的节点和OpContext中的算子对应.

节点表示计算, 算子进行计算.

例如.

- InstanceNode表示样本中的张量, InstanceOp获取并输出样本中的张量.
- VariableNode表示模型参数中的张量, VariableOp获取并输出模型参数中的张量.
- GEMMNode表示矩阵乘法, GEMMOp进行矩阵乘法.
