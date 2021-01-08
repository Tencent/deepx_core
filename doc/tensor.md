# 张量

张量是deepx的核心数据结构, 有以下几种张量.

| 类名 | 张量类型 | 简称 | 描述 | 用途 |
| - | - | - | - | - |
| Tensor<float\_t> | TENSOR\_TYPE\_TSR | TSR | 浮点型稠密张量 | 参数, 样本, 隐层, 梯度 |
| Tensor<int\_t> | TENSOR\_TYPE\_TSRI | TSRI | 整型稠密张量 | 样本 |
| Tensor&lt;std::string&gt; | TENSOR\_TYPE\_TSRS | TSRS | 字符串型稠密张量 | 样本 |
| SparseRowMatrix<int\_t, float\_t> | TENSOR\_TYPE\_SRM | SRM | 稀疏行矩阵 | 参数, 梯度 |
| CSRMatrix<int\_t, float\_t> | TENSOR\_TYPE\_CSR | CSR | 压缩稀疏行矩阵 | 样本 |

其中, float\_t是单精度浮点数, int\_t是64位无符号整数.

## Tensor

Tensor, 稠密张量.

Tensor最高支持8阶稠密张量.

根据数据类型不同, 浮点型稠密张量简称TSR, 整型稠密张量简称TSRI, 字符串型稠密张量简称TSRS.

它的头文件是["tensor.h"](../include/deepx_core/tensor/tensor.h).

## SparseRowMatrix

SparseRowMatrix, 稀疏行矩阵, 简称SRM.

SRM由哈希结构组成. SRM的键是行id, 它的范围是int\_t的范围. SRM的值是行id对应的行.

全0行可以不存储, 节省空间.

SRM支持随机访问, 通过行id查找行非常高效.

下面的TSR可以这样用SRM表示.

| TSR行/列 | 0 | 1 | 2 | 3 |
| - | - | - | - | - |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 2 | -1 | -2 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| 5 | 1 | 1 | 2 | 2 |
| ... | 0 | 0 | 0 | 0 |
| 10000000 | -1 | 0 | 1 | 2 |
| ... | 0 | 0 | 0 | 0 |
| 20000000 | 1 | 0 | 2 | 0 |
| ... | 0 | 0 | 0 | 0 |

| SRM行id | SRM行 |
| - | - |
| 1 | [1, 2, -1, -2] |
| 5 | [1, 1, 2, 2] |
| 10000000 | [-1, 0, 1, 2] |
| 20000000 | [1, 0, 2, 0] |

它的头文件是["sparse\_row\_matrix.h"](../include/deepx_core/tensor/sparse_row_matrix.h).

## CSRMatrix

CSRMatrix(Compressed Sparse Row Matrix), 压缩稀疏行矩阵, 简称CSR.

CSR由1个整数和3个数组组成, 存储紧凑, 节省空间.

CSR支持遍历, 不支持随机访问.

下面的TSR可以这样用CSR表示.

| TSR行/列 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| - | - | - | - | - | - | - | - | - | - | - |
| 0 | 0 | 0 | 1 | 0 | 2 | 0 | 3 | 0 | 0 | 0 |
| 1 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 1 | 0 | 2 | 3 | 0 | 4 | 5 |

| CSR属性 | 类型 | 值 |
| - | - | - |
| row | 整数 | 3 |
| row\_offset | 数组 | [0, 3, 5, 10] |
| col | 数组 | [2, 4, 6, 1, 2, 3, 5, 6, 8, 9] |
| value | 数组 | [1, 2, 3, 1, 2, 1, 2, 3, 4, 5] |

它的头文件是["csr\_matrix.h"](../include/deepx_core/tensor/csr_matrix.h).
