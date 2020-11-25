# 编译优化

本文档介绍几种可以提升deepx性能的编译方法.

## 使用SIMD指令集

```shell
make -j8 SIMD=1
```

SIMD=1将会添加以下编译参数, 请了解它们的作用和副作用.

- -ftree-vectorize
- -ffast-math
- -mavx
- -mfma
- -mavx2

## 使用sage2

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2=1
```

若干单精度数学函数将使用sage2.

## 使用sage2\_sgemm

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2_SGEMM=1
```

单精度矩阵乘法(sgemm)将使用sage2\_sgemm.

sage2\_sgemm动态加载"libmkl\_sage2.so", 使用sage2\_sgemm的程序要和"libmkl\_sage2.so"一起发布并放在同一目录.

请从[prebuilt](https://git.code.oa.com/mmrecommend/prebuilt)获取"libmkl\_sage2.so".

## 使用sage2\_sgemm\_jit

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SAGE2_SGEMM_JIT=1
```

若干op内的单精度矩阵乘法(sgemm)将使用sage2\_sgemm\_jit系列函数.

sage2\_sgemm\_jit动态加载"libmkl\_sage2.so", 使用sage2\_sgemm\_jit的程序要和"libmkl\_sage2.so"一起发布并放在同一目录.

请从[prebuilt](https://git.code.oa.com/mmrecommend/prebuilt)获取"libmkl\_sage2.so".

## 使用SIMD + sage2 + sage2\_sgemm + sage2\_sgemm\_jit(推荐)

```shell
# 在项目根目录执行
git clone https://git.code.oa.com/mmrecommend/sage2.git
cd sage2 && make -j8 && cd ..
make -j8 SIMD=1 SAGE2=1 SAGE2_SGEMM=1 SAGE2_SGEMM_JIT=1
```
