// Copyright 2021 the deepx authors
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#if OS_POSIX == 1
#include <deepx_core/common/stream.h>
#include <deepx_core/contrib/we_ps/client/we_ps_client_impl.h>
#include <deepx_core/dx_log.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace deepx_core {
namespace {

int socket_unix_stream() noexcept {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd == -1) {
    DXERROR("Failed to socket, errno=%d(%s).", errno, strerror(errno));
    return -1;
  }
  return fd;
}

int connect_abstract_unix(int fd, const char* path) noexcept {
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  addr.sun_path[0] = '\0';
#if __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
  snprintf(addr.sun_path + 1, sizeof(addr.sun_path) - 1, "%s", path);
#if __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
  if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    DXERROR("Failed to connect, fd=%d, path=%s, errno=%d(%s).", fd, path, errno,
            strerror(errno));
    return -1;
  }
  return 0;
}

int _close(int fd) noexcept {
  int ok;
  do {
    ok = close(fd);
  } while (ok == -1 && errno == EINTR);
  if (ok == -1) {
    DXERROR("Failed to close, fd=%d, errno=%d(%s).", fd, errno,
            strerror(errno));
  }
  return ok;
}

int writen(int fd, const char* p, size_t size) noexcept {
  size_t left = size;
  ssize_t n = 0;
  while (left > 0) {
    n = write(fd, p, left);
    if (n == -1 && errno == EINTR) {
      continue;
    } else if (n == -1) {
      DXERROR("Failed to write, fd=%d, size=%zu, left=%zu, errno=%d(%s).", fd,
              size, left, errno, strerror(errno));
      return -1;
    } else {
      left -= n;
      p += n;
    }
  }
  return (int)size;
}

int writen(int fd, const std::string& buf) noexcept {
  return writen(fd, &buf[0], buf.size());
}

int readn(int fd, char* p, size_t size) noexcept {
  size_t left = size;
  ssize_t n = 0;
  while (left > 0) {
    n = read(fd, p, left);
    if (n == -1 && errno == EINTR) {
      continue;
    } else if (n == -1 || n == 0) {
      DXERROR("Failed to read, fd=%d, size=%zu, left=%zu, errno=%d(%s).", fd,
              size, left, errno, strerror(errno));
      return -1;
    } else {
      left -= n;
      p += n;
    }
  }
  return 0;
}

int readn(int fd, std::string* buf) noexcept {
  return readn(fd, &(*buf)[0], buf->size());
}

}  // namespace

/************************************************************************/
/* WePSProxyClient */
/************************************************************************/
class WePSProxyClient : public WePSClient {
 private:
  int model_id_ = 0;
  std::string client_uuid_;
  int fd_ = -1;
  std::string out_buf_;
  std::string in_buf_;

 private:
#if !defined NDEBUG
  // NOTE: constraint of WePS.
  static constexpr size_t MAX_TSR_KEY_SIZE = 210;
  static constexpr size_t MAX_SRM_KEY_SIZE = 210;
  static constexpr size_t MAX_SRM_VALUE_SIZE = 19 * 1024 * 1024;
  static constexpr size_t MAX_GRAPH_VALUE_SIZE = MAX_SRM_VALUE_SIZE;
#endif
  static size_t _EncodeTSRSize(const std::string& name, const tsr_t& W,
                               int encode_shape) noexcept;
  static void _EncodeTSR(char** pp, const std::string& name, const tsr_t& W,
                         int encode_shape) noexcept;
  static void _DecodeTSR(const char** pp, std::string* name, int* total_dim,
                         const float_t** data) noexcept;
  bool EncodeTSR(const std::string& name, const tsr_t& W, int encode_shape);
  bool DecodeTSR(tsr_t* W);
  bool EncodeTSR(const TensorMap& param, int encode_shape);
  bool DecodeTSR(TensorMap* param);

 private:
  enum ENCODE_SRM_FLAG {
    ENCODE_SRM_FLAG_SET,
    ENCODE_SRM_FLAG_UPDATE_SET,
    ENCODE_SRM_FLAG_UPDATE_DELTA,
  };
  static size_t _EncodeSRMSize(const std::string& name, const srm_t& W,
                               int flag) noexcept;
  static void _EncodeSRM(char** pp, const std::string& name, const srm_t& W,
                         int flag) noexcept;
  static size_t _EncodeSRMIdSetSize(const std::string& name,
                                    const id_set_t& id_set) noexcept;
  static void _EncodeSRMIdSet(char** pp, const std::string& name,
                              const id_set_t& id_set) noexcept;
  static void _DecodeSRMEntry(const char** pp, std::string* name, int_t* id,
                              int* exist, int* col,
                              const float_t** embedding) noexcept;
  bool EncodeSRM(const std::string& name, const srm_t& W, int flag);
  bool EncodeSRMIdSet(const std::string& name, const id_set_t& id_set);
  bool DecodeSRM(srm_t* W);
  bool EncodeSRM(const TensorMap& param, int flag);
  bool EncodeSRMIdSet(
      const std::unordered_map<std::string, id_set_t>& id_set_map);
  bool DecodeSRM(TensorMap* param);

 private:
  bool EncodeGraphKey();
  bool EncodeGraph(const Graph& graph);
  bool DecodeGraph(Graph* graph, int* exist);

 private:
  bool Connect();
  void Close() noexcept;
  bool WriteCmd(uint16_t cmd);
  bool WriteMessage();
  bool ReadError();
  bool ReadMessage();

 public:
  ~WePSProxyClient() override { Close(); }
  DEFINE_WE_PS_CLIENT_LIKE(WePSProxyClient);
  bool InitConfig(const AnyMap& config) override;
  bool InitConfig(const StringMap& config) override;

 public:
  bool SetTSR(const std::string& name, const tsr_t& W) override;
  bool GetTSR(const std::string& name, tsr_t* W) override;
  bool UpdateTSR(const std::string& name, const tsr_t& delta_W,
                 tsr_t* new_W) override;

  bool SetTSR(const TensorMap& param) override;
  bool GetTSR(TensorMap* param) override;
  bool UpdateTSR(const TensorMap& delta_param, TensorMap* new_param) override;

  bool SetSRM(const std::string& name, const srm_t& W) override;
  bool GetSRM(const std::string& name, const id_set_t& id_set,
              srm_t* W) override;
  bool UpdateSRM(const std::string& name, const srm_t& delta_W) override;

  bool SetSRM(const TensorMap& param) override;
  bool GetSRM(const std::unordered_map<std::string, id_set_t>& id_set_map,
              TensorMap* param) override;
  bool UpdateSRM(const TensorMap& delta_param) override;

  bool SetGraph(const Graph& graph) override;
  bool GetGraph(Graph* graph, int* exist) override;
};

size_t WePSProxyClient::_EncodeTSRSize(const std::string& name, const tsr_t& W,
                                       int encode_shape) noexcept {
  DXASSERT(name.size() <= MAX_TSR_KEY_SIZE);
  size_t size = 0;
  size += sizeof(uint64_t) + name.size();
  if (encode_shape) {
    size += sizeof(uint64_t) + W.rank() * sizeof(uint64_t);
  }
  size += sizeof(uint64_t) + W.total_dim() * sizeof(float_t);
  return size;
}

void WePSProxyClient::_EncodeTSR(char** pp, const std::string& name,
                                 const tsr_t& W, int encode_shape) noexcept {
  DXASSERT(name.size() <= MAX_TSR_KEY_SIZE);
  char* p = *pp;
  *(uint64_t*)p = name.size();
  p += sizeof(uint64_t);
  memcpy(p, name.data(), name.size());
  p += name.size();

  if (encode_shape) {
    *(uint64_t*)p = W.rank();
    p += sizeof(uint64_t);
    for (int i = 0; i < W.rank(); ++i) {
      *(uint64_t*)p = (uint64_t)W.dim(i);
      p += sizeof(uint64_t);
    }
  }

  *(uint64_t*)p = W.total_dim() * sizeof(float_t);
  p += sizeof(uint64_t);
  memcpy(p, W.data(), W.total_dim() * sizeof(float_t));
  p += W.total_dim() * sizeof(float_t);
  *pp = p;
}

void WePSProxyClient::_DecodeTSR(const char** pp, std::string* name,
                                 int* total_dim,
                                 const float_t** data) noexcept {
  const char* p = *pp;
  size_t name_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  DXASSERT(name_size <= MAX_TSR_KEY_SIZE);
  if (name) {
    name->assign(p, name_size);
  }
  p += name_size;

  size_t data_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  if (total_dim) {
    *total_dim = (int)(data_size / sizeof(float_t));
  }
  if (data) {
    *data = (const float_t*)p;
  }
  p += data_size;
  *pp = p;
}

bool WePSProxyClient::EncodeTSR(const std::string& name, const tsr_t& W,
                                int encode_shape) {
  if (W.empty()) {
    return false;
  }

  size_t tsr_size = _EncodeTSRSize(name, W, encode_shape);
  out_buf_.resize(tsr_size);

  char* p = &out_buf_[0];
  _EncodeTSR(&p, name, W, encode_shape);
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::DecodeTSR(tsr_t* W) {
  const char* p = in_buf_.data();
  int total_dim;
  const float_t* data;
  _DecodeTSR(&p, nullptr, &total_dim, &data);
  W->set_data(data, total_dim);
  DXASSERT(p == in_buf_.data() + in_buf_.size());
  return true;
}

bool WePSProxyClient::EncodeTSR(const TensorMap& param, int encode_shape) {
  size_t out_buf_size = 0;
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      const auto& W = Wany.unsafe_to_ref<tsr_t>();
      if (!W.empty()) {
        out_buf_size += _EncodeTSRSize(name, W, encode_shape);
      }
    }
  }
  if (out_buf_size == 0) {
    return false;
  }

  out_buf_.resize(out_buf_size);

  char* p = &out_buf_[0];
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      const auto& W = Wany.unsafe_to_ref<tsr_t>();
      if (!W.empty()) {
        _EncodeTSR(&p, name, W, encode_shape);
      }
    }
  }
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::DecodeTSR(TensorMap* param) {
  const char* p = in_buf_.data();
  std::string name;
  int total_dim;
  const float_t* data;
  for (;;) {
    _DecodeTSR(&p, &name, &total_dim, &data);
    auto& W = param->get<tsr_t>(name);
    W.set_data(data, total_dim);
    DXASSERT(p <= in_buf_.data() + in_buf_.size());
    if (p == in_buf_.data() + in_buf_.size()) {
      break;
    }
  }
  return true;
}

size_t WePSProxyClient::_EncodeSRMSize(const std::string& name, const srm_t& W,
                                       int flag) noexcept {
  size_t key_size = name.size() + sizeof(int_t);
  size_t value_size = W.col() * sizeof(float_t);
  DXASSERT(key_size <= MAX_SRM_KEY_SIZE);
  DXASSERT(value_size <= MAX_SRM_VALUE_SIZE);
  size_t entry_size = 0;
  entry_size += sizeof(uint64_t);
  entry_size += key_size;
  entry_size += sizeof(uint64_t);
  switch (flag) {
    case ENCODE_SRM_FLAG_SET:
      break;
    case ENCODE_SRM_FLAG_UPDATE_SET:
    case ENCODE_SRM_FLAG_UPDATE_DELTA:
      entry_size += sizeof(uint8_t);
      break;
  }
  entry_size += value_size;
  return entry_size * W.size();
}

void WePSProxyClient::_EncodeSRM(char** pp, const std::string& name,
                                 const srm_t& W, int flag) noexcept {
  size_t key_size = name.size() + sizeof(int_t);
  size_t value_size = W.col() * sizeof(float_t);
  DXASSERT(key_size <= MAX_SRM_KEY_SIZE);
  DXASSERT(value_size <= MAX_SRM_VALUE_SIZE);
  char* p = *pp;
  for (const auto& entry : W) {
    *(uint64_t*)p = key_size;
    p += sizeof(uint64_t);
    memcpy(p, name.data(), name.size());
    p += name.size();
    *(int_t*)p = entry.first;
    p += sizeof(int_t);
    *(uint64_t*)p = value_size;
    p += sizeof(uint64_t);
    switch (flag) {
      case ENCODE_SRM_FLAG_SET:
        break;
      case ENCODE_SRM_FLAG_UPDATE_SET:
        *(uint8_t*)p = 1;
        p += sizeof(uint8_t);
        break;
      case ENCODE_SRM_FLAG_UPDATE_DELTA:
        *(uint8_t*)p = 0;
        p += sizeof(uint8_t);
        break;
    }
    memcpy(p, entry.second, value_size);
    p += value_size;
  }
  *pp = p;
}

size_t WePSProxyClient::_EncodeSRMIdSetSize(const std::string& name,
                                            const id_set_t& id_set) noexcept {
  size_t key_size = name.size() + sizeof(int_t);
  DXASSERT(key_size <= MAX_SRM_KEY_SIZE);
  size_t entry_size = 0;
  entry_size += sizeof(uint64_t);
  entry_size += key_size;
  return entry_size * id_set.size();
}

void WePSProxyClient::_EncodeSRMIdSet(char** pp, const std::string& name,
                                      const id_set_t& id_set) noexcept {
  size_t key_size = name.size() + sizeof(int_t);
  DXASSERT(key_size <= MAX_SRM_KEY_SIZE);
  char* p = *pp;
  for (int_t id : id_set) {
    *(uint64_t*)p = key_size;
    p += sizeof(uint64_t);
    memcpy(p, name.data(), name.size());
    p += name.size();
    *(int_t*)p = id;
    p += sizeof(int_t);
  }
  *pp = p;
}

void WePSProxyClient::_DecodeSRMEntry(const char** pp, std::string* name,
                                      int_t* id, int* exist, int* col,
                                      const float_t** embedding) noexcept {
  const char* p = *pp;
  size_t key_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  DXASSERT(key_size >= sizeof(int_t));
  DXASSERT(key_size <= MAX_SRM_KEY_SIZE);
  size_t name_size = key_size - sizeof(int_t);
  if (name) {
    name->assign(p, name_size);
  }
  p += name_size;
  if (id) {
    *id = *(const int_t*)p;
  }
  p += sizeof(int_t);

  size_t value_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  if (value_size != (size_t)UINT64_MAX) {
    DXASSERT(value_size <= MAX_SRM_VALUE_SIZE);
    if (exist) {
      *exist = 1;
    }
    if (col) {
      *col = (int)(value_size / sizeof(float_t));
    }
    if (embedding) {
      *embedding = (const float_t*)p;
    }
    p += value_size;
  } else {
    if (exist) {
      *exist = 0;
    }
  }
  *pp = p;
}

bool WePSProxyClient::EncodeSRM(const std::string& name, const srm_t& W,
                                int flag) {
  if (W.empty()) {
    return false;
  }

  out_buf_.resize(sizeof(uint8_t) + _EncodeSRMSize(name, W, flag));

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  _EncodeSRM(&p, name, W, flag);
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::EncodeSRMIdSet(const std::string& name,
                                     const id_set_t& id_set) {
  if (id_set.empty()) {
    return false;
  }

  out_buf_.resize(sizeof(uint8_t) + _EncodeSRMIdSetSize(name, id_set));

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  _EncodeSRMIdSet(&p, name, id_set);
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::DecodeSRM(srm_t* W) {
  const char* p = in_buf_.data();
  int_t id;
  int exist;
  int col;
  const float_t* embedding;
  W->zeros();
  for (;;) {
    _DecodeSRMEntry(&p, nullptr, &id, &exist, &col, &embedding);
    if (exist) {
      DXASSERT(W->col() == col);
      W->assign(id, embedding);
    }
    DXASSERT(p <= in_buf_.data() + in_buf_.size());
    if (p == in_buf_.data() + in_buf_.size()) {
      break;
    }
  }
  return true;
}

bool WePSProxyClient::EncodeSRM(const TensorMap& param, int flag) {
  size_t srm_size = 0;
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      if (!W.empty()) {
        srm_size += _EncodeSRMSize(name, W, flag);
      }
    }
  }
  if (srm_size == 0) {
    return false;
  }

  out_buf_.resize(sizeof(uint8_t) + srm_size);

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      if (!W.empty()) {
        _EncodeSRM(&p, name, W, flag);
      }
    }
  }
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::EncodeSRMIdSet(
    const std::unordered_map<std::string, id_set_t>& id_set_map) {
  size_t srm_id_set_size = 0;
  for (const auto& entry : id_set_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    if (!id_set.empty()) {
      srm_id_set_size += _EncodeSRMIdSetSize(name, id_set);
    }
  }
  if (srm_id_set_size == 0) {
    return false;
  }

  out_buf_.resize(sizeof(uint8_t) + srm_id_set_size);

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  for (const auto& entry : id_set_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    if (!id_set.empty()) {
      _EncodeSRMIdSet(&p, name, id_set);
    }
  }
  DXASSERT(p == out_buf_.data() + out_buf_.size());
  return true;
}

bool WePSProxyClient::DecodeSRM(TensorMap* param) {
  const char* p = in_buf_.data();
  std::string name;
  int_t id;
  int exist;
  int col;
  const float_t* embedding;
  param->ClearSRMValue();
  for (;;) {
    _DecodeSRMEntry(&p, &name, &id, &exist, &col, &embedding);
    if (exist) {
      auto& W = param->get<srm_t>(name);
      DXASSERT(W.col() == col);
      W.assign(id, embedding);
    }
    DXASSERT(p <= in_buf_.data() + in_buf_.size());
    if (p == in_buf_.data() + in_buf_.size()) {
      break;
    }
  }
  return true;
}

bool WePSProxyClient::EncodeGraphKey() {
  std::string key = WePSClient::GetGraphKey();
  size_t key_size = key.size();
  out_buf_.resize(sizeof(uint8_t) + sizeof(uint64_t) + key_size);

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  *(uint64_t*)p = key_size;
  p += sizeof(uint64_t);
  memcpy(p, key.data(), key_size);
#if !defined NDEBUG
  p += key_size;
  DXASSERT(p == out_buf_.data() + out_buf_.size());
#endif
  return true;
}

bool WePSProxyClient::EncodeGraph(const Graph& graph) {
  OutputStringStream os;
  if (!graph.Write(os)) {
    return false;
  }

  std::string key = WePSClient::GetGraphKey();
  size_t key_size = key.size();
  size_t value_size = os.GetSize();
  DXASSERT(value_size <= MAX_GRAPH_VALUE_SIZE);
  out_buf_.resize(sizeof(uint8_t) + sizeof(uint64_t) + key_size +
                  sizeof(uint64_t) + value_size);

  char* p = &out_buf_[0];
  *(uint8_t*)p = 1;  // unique
  p += sizeof(uint8_t);
  *(uint64_t*)p = key_size;
  p += sizeof(uint64_t);
  memcpy(p, key.data(), key_size);
  p += key_size;
  *(uint64_t*)p = value_size;
  p += sizeof(uint64_t);
  memcpy(p, os.GetData(), value_size);
#if !defined NDEBUG
  p += value_size;
  DXASSERT(p == out_buf_.data() + out_buf_.size());
#endif
  return true;
}

bool WePSProxyClient::DecodeGraph(Graph* graph, int* exist) {
  graph->clear();

  const char* p = in_buf_.data();
  size_t key_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  DXASSERT(key_size == WePSClient::GetGraphKey().size());
  p += key_size;

  size_t value_size = (size_t)(*(const uint64_t*)p);  // NOLINT
  p += sizeof(uint64_t);
  if (value_size != (size_t)UINT64_MAX) {
    DXASSERT(value_size <= MAX_GRAPH_VALUE_SIZE);
    *exist = 1;
    InputStringStream is;
    is.SetView(p, value_size);
    if (!graph->Read(is)) {
      return false;
    }
#if !defined NDEBUG
    p += value_size;
    DXASSERT(p == in_buf_.data() + in_buf_.size());
#endif
  } else {
    *exist = 0;
  }
  return true;
}

bool WePSProxyClient::Connect() {
  Close();

  fd_ = socket_unix_stream();
  if (fd_ == -1) {
    return false;
  }

  char path[256];
  snprintf(path, sizeof(path), "psstor_proxy:%d/%s", model_id_,
           client_uuid_.c_str());
  if (connect_abstract_unix(fd_, path) == -1) {
    return false;
  }
  return true;
}

void WePSProxyClient::Close() noexcept {
  if (fd_ != -1) {
    (void)_close(fd_);
    fd_ = -1;
  }
}

bool WePSProxyClient::WriteCmd(uint16_t cmd) {
  return writen(fd_, (const char*)&cmd, sizeof(cmd)) != -1;
}

bool WePSProxyClient::WriteMessage() {
  uint64_t size = out_buf_.size();
  return writen(fd_, (const char*)&size, sizeof(size)) != -1 &&
         writen(fd_, out_buf_) != -1;
}

bool WePSProxyClient::ReadError() {
  in_buf_.resize(sizeof(uint8_t));
  if (readn(fd_, &in_buf_) == -1) {
    return false;
  }

  uint8_t error = *(const uint8_t*)in_buf_.data();
  if (error) {
    DXERROR("Proxy error.");
    return false;
  }
  return true;
}

bool WePSProxyClient::ReadMessage() {
  uint64_t size;
  if (readn(fd_, (char*)&size, sizeof(size)) == -1) {
    return false;
  }

  in_buf_.resize((size_t)size);
  if (readn(fd_, &in_buf_) == -1) {
    return false;
  }
  return true;
}

bool WePSProxyClient::InitConfig(const AnyMap& config) {
  if (config.count("model_id") == 0) {
    DXERROR("Please specify model_id.");
    return false;
  }
  if (config.count("client_uuid") == 0) {
    DXERROR("Please specify client_uuid.");
    return false;
  }
  model_id_ = std::stoi(config.at("model_id").to_ref<std::string>());
  client_uuid_ = config.at("client_uuid").to_ref<std::string>();
  return true;
}

bool WePSProxyClient::InitConfig(const StringMap& config) {
  if (config.count("model_id") == 0) {
    DXERROR("Please specify model_id.");
    return false;
  }
  if (config.count("client_uuid") == 0) {
    DXERROR("Please specify client_uuid.");
    return false;
  }
  model_id_ = std::stoi(config.at("model_id"));
  client_uuid_ = config.at("client_uuid");
  return true;
}

bool WePSProxyClient::SetTSR(const std::string& name, const tsr_t& W) {
  return !EncodeTSR(name, W, 1) ||
         (Connect() && WriteCmd(6) && WriteMessage() && ReadError());
}

bool WePSProxyClient::GetTSR(const std::string& name, tsr_t* W) {
  return !EncodeTSR(name, *W, 0) ||
         (Connect() && WriteCmd(5) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeTSR(W));
}

bool WePSProxyClient::UpdateTSR(const std::string& name, const tsr_t& delta_W,
                                tsr_t* new_W) {
  return !EncodeTSR(name, delta_W, 0) ||
         (Connect() && WriteCmd(0) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeTSR(new_W));
}

bool WePSProxyClient::SetTSR(const TensorMap& param) {
  return !EncodeTSR(param, 1) ||
         (Connect() && WriteCmd(6) && WriteMessage() && ReadError());
}

bool WePSProxyClient::GetTSR(TensorMap* param) {
  return !EncodeTSR(*param, 0) ||
         (Connect() && WriteCmd(5) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeTSR(param));
}

bool WePSProxyClient::UpdateTSR(const TensorMap& delta_param,
                                TensorMap* new_param) {
  return !EncodeTSR(delta_param, 0) ||
         (Connect() && WriteCmd(0) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeTSR(new_param));
}

bool WePSProxyClient::SetSRM(const std::string& name, const srm_t& W) {
  return !EncodeSRM(name, W, ENCODE_SRM_FLAG_SET) ||
         (Connect() && WriteCmd(1) && WriteMessage() && ReadError());
}

bool WePSProxyClient::GetSRM(const std::string& name, const id_set_t& id_set,
                             srm_t* W) {
  W->zeros();
  return !EncodeSRMIdSet(name, id_set) ||
         (Connect() && WriteCmd(2) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeSRM(W));
}

bool WePSProxyClient::UpdateSRM(const std::string& name, const srm_t& delta_W) {
  return !EncodeSRM(name, delta_W, ENCODE_SRM_FLAG_UPDATE_DELTA) ||
         (Connect() && WriteCmd(8) && WriteMessage() && ReadError());
}

bool WePSProxyClient::SetSRM(const TensorMap& param) {
  return !EncodeSRM(param, ENCODE_SRM_FLAG_SET) ||
         (Connect() && WriteCmd(1) && WriteMessage() && ReadError());
}

bool WePSProxyClient::GetSRM(
    const std::unordered_map<std::string, id_set_t>& id_set_map,
    TensorMap* param) {
  param->ClearSRMValue();
  return !EncodeSRMIdSet(id_set_map) ||
         (Connect() && WriteCmd(2) && WriteMessage() && ReadError() &&
          ReadMessage() && DecodeSRM(param));
}

bool WePSProxyClient::UpdateSRM(const TensorMap& delta_param) {
  return !EncodeSRM(delta_param, ENCODE_SRM_FLAG_UPDATE_DELTA) ||
         (Connect() && WriteCmd(8) && WriteMessage() && ReadError());
}

bool WePSProxyClient::SetGraph(const Graph& graph) {
  return EncodeGraph(graph) && Connect() && WriteCmd(12) && WriteMessage() &&
         ReadError();
}

bool WePSProxyClient::GetGraph(Graph* graph, int* exist) {
  return EncodeGraphKey() && Connect() && WriteCmd(13) && WriteMessage() &&
         ReadError() && ReadMessage() && DecodeGraph(graph, exist);
}

WE_PS_CLIENT_REGISTER(WePSProxyClient, "WePSProxyClient");
WE_PS_CLIENT_REGISTER(WePSProxyClient, "proxy");

}  // namespace deepx_core
#endif
