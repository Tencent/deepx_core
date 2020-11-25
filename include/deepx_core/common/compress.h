// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <string>

namespace deepx_core {

// NOTE: compression/decompression functions are endian sensitive.
bool Compress(const std::string& in, std::string* out);
bool HighCompress(const std::string& in, std::string* out);
bool Decompress(const char* in, int in_size, std::string* out);
bool Decompress(const std::string& in, std::string* out);

}  // namespace deepx_core
