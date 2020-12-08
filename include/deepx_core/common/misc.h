// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <string>
#include <vector>

namespace deepx_core {

size_t GetVerboseBatch(int verbose) noexcept;
std::string GetOutputPredictFile(const std::string& dir,
                                 const std::string& file);

// If 'v' is like "d1,d2,d3,...,dn",
// then 'deep_dims' will be like {d1, d2, d3, ..., dn}.
//
// 'di' must be positive.
bool ParseDeepDims(const std::string& v, std::vector<int>* deep_dims,
                   const char* gflag);

// If 'v' is like "d1,d2,d3,...,dn" and 'dn' is 1,
// then 'deep_dims' will be like {d1, d2, d3, ..., dn}.
//
// If 'v' is like "d1,d2,d3,...,dn" and 'dn' is not 1,
// then 'deep_dims' will be like {d1, d2, d3, ..., dn, 1}.
//
// 'di' must be positive.
bool ParseDeepDimsAppendOne(const std::string& v, std::vector<int>* deep_dims,
                            const char* gflag);

}  // namespace deepx_core
