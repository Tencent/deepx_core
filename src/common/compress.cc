// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/compress.h>
#include <lz4.h>
#include <lz4hc.h>

namespace deepx_core {

bool Compress(const std::string& in, std::string* _out) {
  int in_size = (int)in.size();
  if (in_size == 0) {
    return false;
  }

  int out_bound = LZ4_compressBound(in_size);
  _out->resize(sizeof(int) + out_bound);
  char* out = &(*_out)[0];
  *(int*)out = in_size;
  out += sizeof(int);

  int out_size = LZ4_compress_default(&in[0], out, in_size, out_bound);
  if (out_size == 0 || out_size > out_bound) {
    return false;
  }

  _out->resize(sizeof(int) + out_size);
  return true;
}

bool HighCompress(const std::string& in, std::string* _out) {
  int in_size = (int)in.size();
  if (in_size == 0) {
    return false;
  }

  int out_bound = LZ4_compressBound(in_size);
  _out->resize(sizeof(int) + out_bound);
  char* out = &(*_out)[0];
  *(int*)out = in_size;
  out += sizeof(int);

  int out_size = LZ4_compress_HC(&in[0], out, in_size, out_bound, 9);
  if (out_size == 0 || out_size > out_bound) {
    return false;
  }

  _out->resize(sizeof(int) + out_size);
  return true;
}

bool Decompress(const char* in, int in_size, std::string* _out) {
  int out_size = *(const int*)in;
  in += sizeof(int);
  in_size -= sizeof(int);
  _out->resize(out_size);
  char* out = &(*_out)[0];
  return LZ4_decompress_safe(in, out, in_size, out_size) == out_size;
}

bool Decompress(const std::string& in, std::string* out) {
  return Decompress(in.data(), (int)in.size(), out);
}

}  // namespace deepx_core
