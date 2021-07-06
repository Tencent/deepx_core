# sage2

sage2是一个用于机器学习和深度学习的CPU高性能计算库.

## 编译

sage2需要一个支持C11和C++11的gcc或者clang.

```shell
make -j8
```

## 单元测试

```shell
make -j8 test
```

## 性能测试

```shell
make -j8 perftest
```

## 安装

```shell
make -j8 install PREFIX=/where/you/want/to/install
```

## 文档

[API](doc/api.md)

[Benchmark](https://git.code.oa.com/mmrecommend/sage2/blob/benchmark/benchmark.md)

## 限制

- 只支持linux和macOS.
- 只支持x86\_64构架.
    - 支持AVX, FMA, AVX2指令集.
    - 不支持AVX512指令集.
