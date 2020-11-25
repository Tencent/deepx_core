# 张量

张量是deepx的核心数据结构, 有以下几种张量.

| 类名 | 张量类型 | 简称 | 描述 | 用途 |
| - | - | - | - | - |
| Tensor<float\_t> | TENSOR\_TYPE\_TSR | TSR | 浮点型稠密张量 | 参数, 样本, 隐层, 梯度 |
| Tensor<int\_t> | TENSOR\_TYPE\_TSRI | TSRI | 整型稠密张量 | 样本 |
| Tensor&lt;std::string&gt; | TENSOR\_TYPE\_TSRS | TSRS | 字符串型稠密张量 | 样本 |
| SparseRowMatrix<int\_t, float\_t> | TENSOR\_TYPE\_SRM | SRM | 稀疏行矩阵 | 参数, 梯度 |
| CSRMatrix<int\_t, float\_t> | TENSOR\_TYPE\_CSR | CSR | CSR矩阵 | 样本 |
| SparseRowParam<int\_t, float\_t> | TENSOR\_TYPE\_SRP | SRP | 稀疏行矩阵(参数) | 参数 |
| SparseVectorParam<int\_t, float\_t> | TENSOR\_TYPE\_SVP | SVP | 稀疏向量(参数) | 参数 |
| SparseRowGrad<int\_t, float\_t> | TENSOR\_TYPE\_SRG | SRG | 稀疏行矩阵(梯度) | 梯度 |
| SparseVectorGrad<int\_t, float\_t> | TENSOR\_TYPE\_SVG | SVG | 稀疏向量(梯度) | 梯度 |

其中, float\_t是单精度浮点数, int\_t是64位无符号整数.

## Tensor

deepx最高支持8阶张量.

它的头文件是["tensor.h"](../include/deepx_core/tensor/tensor.h).

## SparseRowMatrix

Sparse Row Matrix, 行稀疏, 列稠密的矩阵.

下面的TSR用SRM这样表示.

| TSR行/列 | 0    | 1    | 2    | 3    |
| -------- | ---- | ---- | ---- | ---- |
| 0        | 0    | 0    | 0    | 0    |
| 1        | 1    | 2    | -1   | -2   |
| 2        | 0    | 0    | 0    | 0    |
| 3        | 0    | 0    | 0    | 0    |
| 4        | 0    | 0    | 0    | 0    |
| 5        | 1    | 1    | 2    | 2    |
| ...      | 0    | 0    | 0    | 0    |
| 10000000 | -1   | 0    | 1    | 2    |
| ...      | 0    | 0    | 0    | 0    |
| 20000000 | 1    | 0    | 2    | 0    |
| ...      | 0    | 0    | 0    | 0    |

| SRM行/列 | 0    | 1    | 2    | 3    |
| -------- | ---- | ---- | ---- | ---- |
| 1        | 1    | 2    | -1   | -2   |
| 5        | 1    | 1    | 2    | 2    |
| 10000000 | -1   | 0    | 1    | 2    |
| 20000000 | 1    | 0    | 2    | 0    |

特点.

- SRM不存储全0行, 节省空间.
- SRM是哈希结构, 键的空间是64位无符号整数的空间, 而TSR的键(行id)必须是连续封闭的.

它的头文件是["sparse\_row\_matrix.h"](../include/deepx_core/tensor/sparse_row_matrix.h).

## CSRMatrix

Compressed Sparse Row Matrix, 稀疏矩阵.

下面的TSR用CSR这样表示.

| TSR行/列 | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0        | 0    | 0    | 1    | 0    | 2    | 0    | 3    | 0    | 0    | 0    |
| 1        | 0    | 1    | 2    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 2        | 0    | 0    | 0    | 1    | 0    | 2    | 3    | 0    | 4    | 5    |

| CSR属性     | 类型 | 值                             |
| ----------- | ---- | ------------------------------ |
| row         | 整数 | 3                              |
| row\_offset | 数组 | [0, 3, 5, 10]                  |
| col         | 数组 | [2, 4, 6, 1, 2, 3, 5, 6, 8, 9] |
| value       | 数组 | [1, 2, 3, 1, 2, 1, 2, 3, 4, 5] |

特点.

- CSR只存储非0元素, 节省空间.
- CSR有紧凑的存储(1个整数和3个数组).
- CSR没有随机访问能力.

它的头文件是["csr\_matrix.h"](../include/deepx_core/tensor/csr_matrix.h).

## SparseRowParam

Sparse Row Matrix for Param, 概念上和SRM相同.

它的头文件是["sparse\_row\_param.h"](../include/deepx_core/tensor/sparse_row_param.h).

## SparseVectorParam

Sparse Vector for Param, 概念上和列数是1的SRM相同.

它的头文件是["sparse\_vector\_param.h"](../include/deepx_core/tensor/sparse_vector_param.h).

## SparseRowGrad

Sparse Row Matrix for Grad, 概念上和SRM相同.

它的头文件是["sparse\_row\_grad.h"](../include/deepx_core/tensor/sparse_row_grad.h).

## SparseVectorGrad

Sparse Vector for Grad, 概念上和列数是1的SRM相同.

它的头文件是["sparse\_vector\_grad.h"](../include/deepx_core/tensor/sparse_vector_grad.h).

## 为什么有多种稀疏张量?

所有稀疏张量在概念上均是SRM, 在实现上亦可用SRM取代.

为了追求极致的性能, 考虑到以下因素, 将不同稀疏张量用于不同用途.

- 稀疏向量的列数是1, 稀疏矩阵的列数通常大于1.
- 参数需要在内存中长期保存, 梯度不需要在内存中长期保存.
