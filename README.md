# deepx\_core

deepx\_core是一个专注于张量计算/深度学习的基础库.

基于deepx\_core, 可以开发出数值计算/数值优化/凸优化/机器学习/深度学习/强化学习/图神经网络/无监督学习等应用.

## 编译

deepx\_core需要一个支持C++11的编译器.

```shell
make -j8
```

## 单元测试

```shell
make -j8 test
```

## 安装

```shell
make -j8 install PREFIX=/where/you/want/to/install
```

## 文档

[编译优化](doc/compilation.md)

[hdfs](doc/hdfs.md)

[特征](doc/feature.md)

[样本格式](doc/instance.md)

[张量](doc/tensor.md)

[计算图引擎](doc/graph_engine.md)

- [样本解析器](doc/instance_reader.md)

- [计算图](doc/graph.md)

- [算子使用手册](doc/op_manual.md)

- [算子开发](doc/op_dev.md)

- [优化器](doc/optimizer.md)

## 例子

[张量计算](example/computation/README.md)

[排序模型](example/rank/README.md)
