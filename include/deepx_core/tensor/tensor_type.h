// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once

namespace deepx_core {

/************************************************************************/
/* TENSOR_DTYPE */
/************************************************************************/
enum TENSOR_DTYPE {
  TENSOR_DTYPE_NONE = 0,
  TENSOR_DTYPE_FLOAT32 = 1,
  TENSOR_DTYPE_FLOAT64 = 2,
  TENSOR_DTYPE_FLOAT16 = 3,
  TENSOR_DTYPE_BOOL = 4,
  TENSOR_DTYPE_INT8 = 5,
  TENSOR_DTYPE_INT16 = 6,
  TENSOR_DTYPE_INT32 = 7,
  TENSOR_DTYPE_INT64 = 8,
  TENSOR_DTYPE_UINT8 = 9,
  TENSOR_DTYPE_UINT16 = 10,
  TENSOR_DTYPE_UINT32 = 11,
  TENSOR_DTYPE_UINT64 = 12,
};

/************************************************************************/
/* TENSOR_TYPE */
/************************************************************************/
enum TENSOR_TYPE {
  TENSOR_TYPE_NONE = 0,
  TENSOR_TYPE_TSR = 1,
  TENSOR_TYPE_SRM = 2,
  TENSOR_TYPE_CSR = 3,
  TENSOR_TYPE_TSRI = 4,
  TENSOR_TYPE_TSRS = 5,
  TENSOR_TYPE_SRP = 10,  // backward compatibility
  TENSOR_TYPE_SVP = 11,  // backward compatibility
  TENSOR_TYPE_SRG = 12,  // backward compatibility
  TENSOR_TYPE_SVG = 13,  // backward compatibility
};

/************************************************************************/
/* TENSOR_INITIALIZER_TYPE */
/************************************************************************/
enum TENSOR_INITIALIZER_TYPE {
  TENSOR_INITIALIZER_TYPE_NONE = 0,
  TENSOR_INITIALIZER_TYPE_ZEROS = 1,
  TENSOR_INITIALIZER_TYPE_ONES = 2,
  TENSOR_INITIALIZER_TYPE_CONSTANT = 3,
  TENSOR_INITIALIZER_TYPE_RAND = 10,
  TENSOR_INITIALIZER_TYPE_RANDN = 11,
  TENSOR_INITIALIZER_TYPE_RAND_LECUN = 12,
  TENSOR_INITIALIZER_TYPE_RANDN_LECUN = 13,
  TENSOR_INITIALIZER_TYPE_RAND_XAVIER = 14,
  TENSOR_INITIALIZER_TYPE_RANDN_XAVIER = 15,
  TENSOR_INITIALIZER_TYPE_RAND_HE = 16,
  TENSOR_INITIALIZER_TYPE_RANDN_HE = 17,
  TENSOR_INITIALIZER_TYPE_RAND_INT = 18,
  TENSOR_INITIALIZER_TYPE_ARANGE = 19,
};

}  // namespace deepx_core
