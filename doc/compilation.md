# 编译优化

本文档介绍几种可以提升deepx性能的编译方法.

## 使用SIMD指令集

```shell
make -j8 SIMD=1
```

SIMD=1将添加以下编译参数, 用AVX/FMA/AVX2指令集生成向量化代码.

```
CFLAGS       += -ftree-vectorize -ffast-math -mavx -mfma -mavx2
CXXFLAGS     += -ftree-vectorize -ffast-math -mavx -mfma -mavx2
```

## 使用sage2(腾讯内部)

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2=1
```

若干单精度数学函数将使用sage2.

## 使用sage2\_sgemm(腾讯内部)

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2_SGEMM=1
```

单精度矩阵乘法(sgemm)将使用sage2\_sgemm.

## 使用sage2\_sgemm\_jit(腾讯内部)

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2_SGEMM_JIT=1
```

若干op内的单精度矩阵乘法(sgemm)将使用sage2\_sgemm\_jit系列函数.

## 使用MKL(腾讯内部)

在Intel CPU上, MKL可以加速sage2\_sgemm和sage2\_sgemm\_jit.

请从[prebuilt(腾讯内部)](https://git.code.oa.com/mmrecommend/prebuilt)获取"libmkl\_sage2.so"并和deepx程序一起发布.

deepx程序启动时, 将尝试动态加载"libmkl\_sage2.so".

如果加载失败, 将看到以下日志.

```
sage2_sgemm is using default kernel.
```

请检查.

1. "libmkl\_sage2.so"是否可以被加载到.
2. 是否是Intel CPU.

如果加载成功, 将看到以下日志.

```
sage2_sgemm is using MKL kernel.
```

## 推荐使用SIMD + sage2 + sage2\_sgemm + sage2\_sgemm\_jit + MKL(腾讯内部)

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SIMD=1 SAGE2=1 SAGE2_SGEMM=1 SAGE2_SGEMM_JIT=1
```

参考"使用MKL".
