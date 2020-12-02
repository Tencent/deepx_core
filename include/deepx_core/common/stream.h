// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>  // std::is_pod, ...
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Stream functions */
/************************************************************************/
bool IsDirSeparator(char c) noexcept;
std::string basename(const std::string& path);
std::string dirname(const std::string& path);
std::string CanonicalizePath(const std::string& path);
void CanonicalizePath(std::string* path);
bool IsHDFSPath(const std::string& path) noexcept;
bool IsGzipFile(const std::string& file) noexcept;
std::string StdinStdoutPath();
bool IsStdinStdoutPath(const std::string& path) noexcept;

/************************************************************************/
/* FilePath */
/************************************************************************/
class FilePath {
 private:
  std::string path_;

 public:
  FilePath() = default;
  FilePath(const char* s) : path_(s) {}  // NOLINT
  FilePath(const char* s, std::size_t size) : path_(s, size) {}
  FilePath(const std::string& path) : path_(path) {}  // NOLINT
  template <typename II>
  FilePath(II first, II last) : path_(first, last) {}

  bool empty() const noexcept { return path_.empty(); }
  void clear() noexcept { path_.clear(); }

  const std::string& str() const noexcept { return path_; }
  const char* c_str() const noexcept { return path_.c_str(); }

  FilePath basename() const;
  FilePath dirname() const;
  FilePath canonical() const;

 public:
  using iterator = std::string::iterator;
  using const_iterator = std::string::const_iterator;
  using reverse_iterator = std::string::reverse_iterator;
  using const_reverse_iterator = std::string::const_reverse_iterator;

  iterator begin() noexcept { return path_.begin(); }
  const_iterator begin() const noexcept { return path_.begin(); }
  const_iterator cbegin() const noexcept { return path_.cbegin(); }
  iterator end() noexcept { return path_.end(); }
  const_iterator end() const noexcept { return path_.end(); }
  const_iterator cend() const noexcept { return path_.cend(); }
  reverse_iterator rbegin() noexcept { return path_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return path_.rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return path_.crbegin(); }
  reverse_iterator rend() noexcept { return path_.rend(); }
  const_reverse_iterator rend() const noexcept { return path_.rend(); }
  const_reverse_iterator crend() const noexcept { return path_.crend(); }
};

inline bool operator==(const FilePath& left, const FilePath& right) {
  return left.str() == right.str();
}

inline bool operator!=(const FilePath& left, const FilePath& right) {
  return !(left == right);
}

inline bool operator==(const FilePath& left, const std::string& right) {
  return left.str() == right;
}

inline bool operator!=(const FilePath& left, const std::string& right) {
  return !(left == right);
}

inline bool operator==(const std::string& left, const FilePath& right) {
  return left == right.str();
}

inline bool operator!=(const std::string& left, const FilePath& right) {
  return !(left == right);
}

inline bool operator==(const FilePath& left, const char* right) {
  return left.str() == right;
}

inline bool operator!=(const FilePath& left, const char* right) {
  return !(left == right);
}

inline bool operator==(const char* left, const FilePath& right) {
  return left == right.str();
}

inline bool operator!=(const char* left, const FilePath& right) {
  return !(left == right);
}

/************************************************************************/
/* FILE_TYPE */
/************************************************************************/
enum FILE_TYPE {
  FILE_TYPE_NONE = 0,
  FILE_TYPE_DIR = 0x01,
  FILE_TYPE_REG_FILE = 0x02,
  FILE_TYPE_SYM_LINK = 0x04,
  FILE_TYPE_OTHER = 0x1000,
};

/************************************************************************/
/* FileStat */
/************************************************************************/
class FileStat {
 private:
  int exists_ = 0;
  int type_ = FILE_TYPE_NONE;
  std::size_t file_size_ = 0;

 public:
  void set_exists(int exists) noexcept { exists_ = exists; }
  void set_type(int type) noexcept { type_ = type; }
  void set_file_size(std::size_t file_size) noexcept { file_size_ = file_size; }
  void clear() noexcept;

 public:
  bool Exists() const noexcept { return exists_ != 0; }
  bool IsDir() const noexcept { return type_ & FILE_TYPE_DIR; }
  bool IsFile() const noexcept {
    return type_ & (FILE_TYPE_REG_FILE | FILE_TYPE_SYM_LINK);
  }
  bool IsRegFile() const noexcept { return type_ & FILE_TYPE_REG_FILE; }
  bool IsSymLink() const noexcept { return type_ & FILE_TYPE_SYM_LINK; }
  bool IsOther() const noexcept { return type_ & FILE_TYPE_OTHER; }
  std::size_t GetFileSize() const noexcept { return file_size_; }

 public:
  static FileStat StdinStdout() noexcept;
};

/************************************************************************/
/* FileSystem */
/************************************************************************/
class FileSystem {
 public:
  virtual ~FileSystem() = default;

  virtual bool Stat(const FilePath& path, FileStat* stat) = 0;
  bool Exists(const FilePath& path);
  bool IsDir(const FilePath& path);
  bool IsFile(const FilePath& path);
  bool IsRegFile(const FilePath& path);
  bool IsSymLink(const FilePath& path);
  bool IsOther(const FilePath& path);
  bool GetFileSize(const FilePath& path, std::size_t* size);

  // List 'path'.
  //
  // If 'skip_dir' is true, child dirs will be skipped.
  virtual bool List(const FilePath& path, bool skip_dir,
                    std::vector<std::pair<FilePath, FileStat>>* children) = 0;

  // List 'path' recursively.
  //
  // If 'skip_dir' is true, child dirs will be skipped.
  virtual bool ListRecursive(
      const FilePath& path, bool skip_dir,
      std::vector<std::pair<FilePath, FileStat>>* children) = 0;

  virtual bool MakeDir(const FilePath& dir) = 0;

  virtual bool Move(const FilePath& old_path, const FilePath& new_path) = 0;

  // Backup 'old_path' to 'new_path'.
  // 'new_path' = 'old_path' + timestamp.
  //
  // Return if 'old_path' exists and it has been backed up to 'new_path'.
  bool BackupIfExists(const FilePath& old_path, FilePath* new_path);
};

/************************************************************************/
/* LocalFileSystem */
/************************************************************************/
class LocalFileSystem : public FileSystem {
 public:
  bool Stat(const FilePath& path, FileStat* stat) override;
  bool List(const FilePath& path, bool skip_dir,
            std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool ListRecursive(
      const FilePath& path, bool skip_dir,
      std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool MakeDir(const FilePath& dir) override;
  bool Move(const FilePath& old_path, const FilePath& new_path) override;
};

/************************************************************************/
/* StreamBase */
/************************************************************************/
class StreamBase {
 protected:
  int bad_ = 0;

 public:
  void clear_bad() noexcept { bad_ = 0; }
  void set_bad() noexcept { bad_ = 1; }
  bool bad() const noexcept { return bad_ != 0; }
  explicit operator bool() const noexcept { return bad_ == 0; }
  bool operator!() const noexcept { return bad_ != 0; }

 public:
  StreamBase() = default;
  virtual ~StreamBase() = default;
  StreamBase(const StreamBase&) = delete;
  StreamBase& operator=(const StreamBase&) = delete;
};

/************************************************************************/
/* OutputStream */
/************************************************************************/
class OutputStream : virtual public StreamBase {
 public:
  virtual std::size_t Write(const void* data, std::size_t size) = 0;
  virtual bool Flush() { return true; }

  template <typename T>
  void BatchWrite(const T& t);

  template <typename T, typename... Args>
  void BatchWrite(const T& t, Args&&... args);

  template <typename... Args>
  void WriteObject(Args&&... args);
};

template <typename T>
void OutputStream::BatchWrite(const T& t) {
  *this << t;
}

template <typename T, typename... Args>
void OutputStream::BatchWrite(const T& t, Args&&... args) {
  *this << t;
  BatchWrite(std::forward<Args>(args)...);
}

template <typename... Args>
void OutputStream::WriteObject(Args&&... args) {
  BatchWrite(std::forward<Args>(args)...);
}

/************************************************************************/
/* InputStream */
/************************************************************************/
class InputStream : virtual public StreamBase {
 public:
  virtual std::size_t Read(void* data, std::size_t size) = 0;
  virtual char ReadChar() = 0;
  virtual std::size_t Peek(void* data, std::size_t size) = 0;

 public:
  template <typename T>
  void BatchRead(T& t);

  template <typename T, typename... Args>
  void BatchRead(T& t, Args&&... args);

  template <typename... Args>
  void ReadObject(Args&&... args);
};

template <typename T, typename... Args>
void InputStream::BatchRead(T& t, Args&&... args) {
  *this >> t;
  BatchRead(std::forward<Args>(args)...);
}

template <typename T>
void InputStream::BatchRead(T& t) {
  *this >> t;
}

template <typename... Args>
void InputStream::ReadObject(Args&&... args) {
  BatchRead(std::forward<Args>(args)...);
}

/************************************************************************/
/* GetLine */
/************************************************************************/
InputStream& GetLine(InputStream& is, std::string& line);              // NOLINT
InputStream& GetLine(InputStream& is, std::string& line, char delim);  // NOLINT

/************************************************************************/
/* IOStream */
/************************************************************************/
class IOStream : virtual public InputStream, virtual public OutputStream {};

/************************************************************************/
/* BufferedInputStream */
/************************************************************************/
class BufferedInputStream : public InputStream {
 protected:
  InputStream* const is_;
  std::size_t buf_size_;
  std::string buf_;
  char* begin_;
  char* cur_;
  char* end_;

 protected:
  std::size_t FillEmptyBuf();
  std::size_t EnsureBuf(std::size_t need_bytes);

 public:
  explicit BufferedInputStream(InputStream* is,
                               std::size_t buf_size = 64 * 1024  // magic number
  );
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;
};

/************************************************************************/
/* GunzipInputStream */
/************************************************************************/
class GunzipInputStream : public InputStream {
 protected:
  InputStream* const is_;
  void* zs_;
  std::size_t comp_buf_size_;
  std::string comp_buf_;
  char* comp_begin_;
  std::size_t buf_size_;
  std::string buf_;
  char* begin_;
  char* cur_;
  char* end_;

 protected:
  std::size_t FillEmptyBuf();
  std::size_t EnsureBuf(std::size_t need_bytes);

 public:
  explicit GunzipInputStream(InputStream* is,
                             std::size_t buf_size = 64 * 1024  // magic number
  );
  ~GunzipInputStream() override;
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;
};

/************************************************************************/
/* FILE_OPEN_MODE */
/************************************************************************/
enum FILE_OPEN_MODE {
  FILE_OPEN_MODE_NONE = 0,
  FILE_OPEN_MODE_APPEND = 0x01,
  FILE_OPEN_MODE_TRUNCATE = 0x02,
  FILE_OPEN_MODE_IN = 0x04,
  FILE_OPEN_MODE_OUT = 0x08,
  FILE_OPEN_MODE_BINARY = 0x10,
};

/************************************************************************/
/* CFileStream */
/************************************************************************/
class CFileStream : public IOStream {
 protected:
  void* f_ = nullptr;
  int mode_ = FILE_OPEN_MODE_NONE;

 public:
  CFileStream();
  ~CFileStream() override;
  std::size_t Write(const void* data, std::size_t size) override;
  bool Flush() override;
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;

 public:
  bool Open(const std::string& file, int mode);
  bool IsOpen() const noexcept;
  void Close() noexcept;
};

/************************************************************************/
/* OutputStringStream */
/************************************************************************/
class OutputStringStream : public OutputStream {
 protected:
  std::string buf_;
  std::string* buf_ptr_ = &buf_;

 public:
  std::size_t Write(const void* data, std::size_t size) override;

 public:
  void SetString(std::string s) noexcept {
    buf_ = std::move(s);
    buf_ptr_ = &buf_;
  }

  void SetView(std::string* s) noexcept {
    buf_.clear();
    buf_ptr_ = s;
  }

  std::string GetString() const { return *buf_ptr_; }

  const char* GetData() const noexcept { return buf_ptr_->data(); }

  std::size_t GetSize() const noexcept { return buf_ptr_->size(); }

  std::pair<const char*, std::size_t> GetBuf() const noexcept {
    return std::make_pair(buf_ptr_->data(), buf_ptr_->size());
  }

  void clear() noexcept { buf_ptr_->clear(); }

 public:
  void BeginMessage();
  void EndMessage() noexcept;
};

/************************************************************************/
/* InputStringStream */
/************************************************************************/
class InputStringStream : public InputStream {
 protected:
  std::string buf_;
  const char* cur_;
  const char* end_;

 protected:
  void Init(const char* data, std::size_t size);

 public:
  InputStringStream();
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;
  std::size_t Skip(std::size_t size);

 public:
  void SetString(const char* s) {
    buf_ = s;
    Init(buf_.data(), buf_.size());
  }

  void SetString(const std::string& s) {
    buf_ = s;
    Init(buf_.data(), buf_.size());
  }

  void SetString(std::string s) noexcept {
    buf_ = std::move(s);
    Init(buf_.data(), buf_.size());
  }

  void SetString(const char* data, std::size_t size) {
    buf_.assign(data, size);
    Init(buf_.data(), buf_.size());
  }

  void SetString(const std::pair<const char*, std::size_t>& buf) {
    buf_.assign(buf.first, buf.second);
    Init(buf_.data(), buf_.size());
  }

  void SetView(const std::string& s) noexcept { Init(s.data(), s.size()); }

  void SetView(const char* data, std::size_t size) noexcept {
    Init(data, size);
  }

  void SetView(const std::pair<const char*, std::size_t>& buf) noexcept {
    Init(buf.first, buf.second);
  }

  std::string GetString() const {
    std::size_t size = end_ - cur_;
    if (size) {
      return std::string(cur_, size);
    } else {
      return "";
    }
  }

  const char* GetData() const noexcept {
    std::size_t size = end_ - cur_;
    if (size) {
      return cur_;
    } else {
      return nullptr;
    }
  }

  std::size_t GetSize() const noexcept { return end_ - cur_; }

  std::pair<const char*, std::size_t> GetBuf() const noexcept {
    std::size_t size = end_ - cur_;
    if (size) {
      return std::make_pair(cur_, size);
    } else {
      return std::make_pair(nullptr, 0);
    }
  }

  void clear() noexcept { Init(nullptr, 0); }
};

/************************************************************************/
/* HDFS functions */
/************************************************************************/
// All HDFS stuffs can only be used when HasHDFS() returns true.
bool HasHDFS() noexcept;

/************************************************************************/
/* HDFSHandle */
/************************************************************************/
class HDFSHandle {
 private:
  void* raw_handle_ = nullptr;

 public:
  void* raw_handle() const noexcept { return raw_handle_; }

 public:
  ~HDFSHandle();
  bool Connect(const char* name_node_host, uint16_t name_node_port);
  bool Connect(const std::string& name_node_host, uint16_t name_node_port);
  bool ConnectDefault();
  bool IsOpen() const noexcept;
  void Close() noexcept;
};

/************************************************************************/
/* HDFSFileSystem */
/************************************************************************/
class HDFSFileSystem : public FileSystem {
 protected:
  HDFSHandle* const handle_;

 public:
  explicit HDFSFileSystem(HDFSHandle* handle);
  bool Stat(const FilePath& path, FileStat* stat) override;
  bool List(const FilePath& path, bool skip_dir,
            std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool ListRecursive(
      const FilePath& path, bool skip_dir,
      std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool MakeDir(const FilePath& dir) override;
  bool Move(const FilePath& old_path, const FilePath& new_path) override;
};

/************************************************************************/
/* HDFSFileStream */
/************************************************************************/
class HDFSFileStream : public IOStream {
 protected:
  HDFSHandle* const handle_ = nullptr;
  void* f_ = nullptr;
  int mode_ = FILE_OPEN_MODE_NONE;

 public:
  explicit HDFSFileStream(HDFSHandle* handle);
  ~HDFSFileStream() override;
  std::size_t Write(const void* data, std::size_t size) override;
  bool Flush() override;
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;

 public:
  bool Open(const std::string& file, int mode);
  bool IsOpen() const noexcept;
  void Close() noexcept;
};

/************************************************************************/
/* AutoFileSystem */
/************************************************************************/
class AutoFileSystem : public FileSystem {
 protected:
  std::unique_ptr<HDFSHandle> hdfs_handle_;
  std::unique_ptr<FileSystem> fs_;

 public:
  using FileSystem::BackupIfExists;
  using FileSystem::Exists;
  using FileSystem::GetFileSize;
  using FileSystem::IsDir;
  using FileSystem::IsFile;
  using FileSystem::IsOther;
  using FileSystem::IsRegFile;
  using FileSystem::IsSymLink;

  bool Stat(const FilePath& path, FileStat* stat) override;
  bool List(const FilePath& path, bool skip_dir,
            std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool ListRecursive(
      const FilePath& path, bool skip_dir,
      std::vector<std::pair<FilePath, FileStat>>* children) override;
  bool MakeDir(const FilePath& dir) override;
  bool Move(const FilePath& old_path, const FilePath& new_path) override;

 public:
  bool Open(const std::string& path);
  bool IsOpen() const noexcept;
  void Close() noexcept;

 public:
  static bool Exists(const std::string& path);
  static bool IsDir(const std::string& path);
  static bool IsFile(const std::string& path);
  static bool IsRegFile(const std::string& path);
  static bool IsSymLink(const std::string& path);
  static bool IsOther(const std::string& path);
  static bool GetFileSize(const std::string& path, std::size_t* size);
  static bool List(const std::string& path, bool skip_dir,
                   std::vector<std::string>* children);
  static bool ListRecursive(const std::string& path, bool skip_dir,
                            std::vector<std::string>* children);
  static bool MakeDir(const std::string& dir);
  static bool Move(const std::string& old_path, const std::string& new_path);
  static bool BackupIfExists(const std::string& old_path,
                             std::string* new_path);
};

/************************************************************************/
/* AutoInputFileStream */
/************************************************************************/
class AutoInputFileStream : public InputStream {
 protected:
  std::unique_ptr<HDFSHandle> hdfs_handle_;
  std::unique_ptr<InputStream> is_extra_;
  std::unique_ptr<InputStream> is_;

 public:
  AutoInputFileStream();
  std::size_t Read(void* data, std::size_t size) override;
  char ReadChar() override;
  std::size_t Peek(void* data, std::size_t size) override;

 public:
  bool Open(const std::string& file);
  bool IsOpen() const noexcept;
  void Close() noexcept;
};

/************************************************************************/
/* AutoOutputFileStream */
/************************************************************************/
class AutoOutputFileStream : public OutputStream {
 protected:
  std::unique_ptr<HDFSHandle> hdfs_handle_;
  std::unique_ptr<OutputStream> os_;

 public:
  AutoOutputFileStream();
  std::size_t Write(const void* data, std::size_t size) override;
  bool Flush() override;

 public:
  bool Open(const std::string& file);
  bool IsOpen() const noexcept;
  void Close() noexcept;
};

/************************************************************************/
/* Serialization functions */
/************************************************************************/
// NOTE: serialization functions are endian sensitive.
template <typename T>
OutputStream& operator<<(OutputStream& os, const T& t);
template <typename T>
InputStream& operator>>(InputStream& is, T& t);
template <typename T>
InputStringStream& ReadView(InputStringStream& is, T& t);  // NOLINT
OutputStream& operator<<(OutputStream& os, const std::string& s);
InputStream& operator>>(InputStream& is, std::string& s);
template <typename T, typename A>
OutputStream& operator<<(OutputStream& os, const std::vector<T, A>& v);
template <typename T, typename A>
InputStream& operator>>(InputStream& is, std::vector<T, A>& v);
template <typename T, typename A>
InputStringStream& ReadView(InputStringStream& is,  // NOLINT
                            std::vector<T, A>& v);  // NOLINT
template <typename T1, typename T2>
OutputStream& operator<<(OutputStream& os, const std::pair<T1, T2>& p);
template <typename T1, typename T2>
InputStream& operator>>(InputStream& is, std::pair<T1, T2>& p);
template <typename T1, typename T2>
InputStringStream& ReadView(InputStringStream& is,  // NOLINT
                            std::pair<T1, T2>& p);  // NOLINT
template <typename K, typename T, typename H, typename P, typename A>
OutputStream& operator<<(OutputStream& os,
                         const std::unordered_map<K, T, H, P, A>& m);
template <typename K, typename T, typename H, typename P, typename A>
InputStream& operator>>(InputStream& is, std::unordered_map<K, T, H, P, A>& m);
template <typename K, typename T, typename H, typename P, typename A>
InputStringStream& ReadView(InputStringStream& is,                  // NOLINT
                            std::unordered_map<K, T, H, P, A>& m);  // NOLINT
template <typename V, typename H, typename P, typename A>
OutputStream& operator<<(OutputStream& os,
                         const std::unordered_set<V, H, P, A>& s);
template <typename V, typename H, typename P, typename A>
InputStream& operator>>(InputStream& is, std::unordered_set<V, H, P, A>& s);
template <typename V, typename H, typename P, typename A>
InputStringStream& ReadView(InputStringStream& is,               // NOLINT
                            std::unordered_set<V, H, P, A>& s);  // NOLINT

template <typename T>
OutputStream& operator<<(OutputStream& os, const T& t) {
  static_assert(std::is_pod<T>::value, "T must be POD.");
  os.Write(&t, sizeof(t));
  return os;
}

template <typename T>
InputStream& operator>>(InputStream& is, T& t) {
  static_assert(std::is_pod<T>::value, "T must be POD.");
  is.Read(&t, sizeof(t));
  return is;
}

template <typename T>
InputStringStream& ReadView(InputStringStream& is, T& t) {
  // no actual view
  is >> t;
  return is;
}

namespace detail {

template <typename T, typename A>
void WriteVector(OutputStream& os, const std::vector<T, A>& v,  // NOLINT
                 std::true_type /*is_pod*/) {
  int size = (int)v.size();
  os.Write(&size, sizeof(size));
  if (size > 0) {
    os.Write(v.data(), sizeof(T) * size);
  }
}

template <typename T, typename A>
void WriteVector(OutputStream& os, const std::vector<T, A>& v,  // NOLINT
                 std::false_type /*is_pod*/) {
  int size = (int)v.size();
  os.Write(&size, sizeof(size));
  for (int i = 0; i < size; ++i) {
    os << v[i];
    if (!os) {
      break;
    }
  }
}

template <typename T, typename A>
void ReadVector(InputStream& is, std::vector<T, A>& v,  // NOLINT
                std::true_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      v.resize(size);
      is.Read(&v[0], sizeof(T) * size);
    } else {
      v.clear();
    }
  }
}

template <typename T, typename A>
void ReadVector(InputStream& is, std::vector<T, A>& v,  // NOLINT
                std::false_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      v.resize(size);
      for (int i = 0; i < size; ++i) {
        is >> v[i];
        if (!is) {
          return;
        }
      }
    } else {
      v.clear();
    }
  }
}

template <typename T, typename A>
void ReadVectorView(InputStringStream& is, std::vector<T, A>& v,  // NOLINT
                    std::true_type is_pod) {
  // no actual view
  ReadVector(is, v, is_pod);
}

template <typename T, typename A>
void ReadVectorView(InputStringStream& is, std::vector<T, A>& v,  // NOLINT
                    std::false_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      v.resize(size);
      for (int i = 0; i < size; ++i) {
        ReadView(is, v[i]);
        if (!is) {
          return;
        }
      }
    } else {
      v.clear();
    }
  }
}

}  // namespace detail

template <typename T, typename A>
OutputStream& operator<<(OutputStream& os, const std::vector<T, A>& v) {
  detail::WriteVector(os, v, typename std::is_pod<T>::type());
  return os;
}

template <typename T, typename A>
InputStream& operator>>(InputStream& is, std::vector<T, A>& v) {
  detail::ReadVector(is, v, typename std::is_pod<T>::type());
  return is;
}

template <typename T, typename A>
InputStringStream& ReadView(InputStringStream& is, std::vector<T, A>& v) {
  detail::ReadVectorView(is, v, typename std::is_pod<T>::type());
  return is;
}

template <typename T1, typename T2>
OutputStream& operator<<(OutputStream& os, const std::pair<T1, T2>& p) {
  os << p.first << p.second;
  return os;
}

template <typename T1, typename T2>
InputStream& operator>>(InputStream& is, std::pair<T1, T2>& p) {
  is >> p.first >> p.second;
  return is;
}

template <typename T1, typename T2>
InputStringStream& ReadView(InputStringStream& is, std::pair<T1, T2>& p) {
  ReadView(is, p.first);
  ReadView(is, p.second);
  return is;
}

template <typename K, typename T, typename H, typename P, typename A>
OutputStream& operator<<(OutputStream& os,
                         const std::unordered_map<K, T, H, P, A>& m) {
  int version = 0x0a0c72e7;            // magic number version
  uint64_t size = (uint64_t)m.size();  // NOLINT
  os << version;
  os << size;

  auto first = m.begin();
  auto last = m.end();
  for (; first != last; ++first) {
    os << first->first << first->second;
    if (!os) {
      break;
    }
  }
  return os;
}

template <typename K, typename T, typename H, typename P, typename A>
InputStream& operator>>(InputStream& is, std::unordered_map<K, T, H, P, A>& m) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  std::size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (std::size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (std::size_t)size_i;
  }
  if (!is) {
    return is;
  }

  m.clear();
  if (size > 0) {
    K key;
    T value;
    m.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      is >> key >> value;
      if (!is) {
        return is;
      }
      m.emplace(std::move(key), std::move(value));
    }
  }
  return is;
}

template <typename K, typename T, typename H, typename P, typename A>
InputStringStream& ReadView(InputStringStream& is,
                            std::unordered_map<K, T, H, P, A>& m) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  std::size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (std::size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (std::size_t)size_i;
  }
  if (!is) {
    return is;
  }

  m.clear();
  if (size > 0) {
    K key;
    T value;
    m.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      ReadView(is, key);
      ReadView(is, value);
      if (!is) {
        return is;
      }
      m.emplace(std::move(key), std::move(value));
    }
  }
  return is;
}

template <typename V, typename H, typename P, typename A>
OutputStream& operator<<(OutputStream& os,
                         const std::unordered_set<V, H, P, A>& s) {
  int version = 0x0a0c72e7;            // magic number version
  uint64_t size = (uint64_t)s.size();  // NOLINT
  os << version;
  os << size;

  auto first = s.begin();
  auto last = s.end();
  for (; first != last; ++first) {
    os << *first;
    if (!os) {
      break;
    }
  }
  return os;
}

template <typename V, typename H, typename P, typename A>
InputStream& operator>>(InputStream& is, std::unordered_set<V, H, P, A>& s) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  std::size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (std::size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (std::size_t)size_i;
  }
  if (!is) {
    return is;
  }

  s.clear();
  if (size > 0) {
    V value;
    s.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      is >> value;
      if (!is) {
        return is;
      }
      s.emplace(std::move(value));
    }
  }
  return is;
}

template <typename V, typename H, typename P, typename A>
InputStringStream& ReadView(InputStringStream& is,
                            std::unordered_set<V, H, P, A>& s) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  std::size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (std::size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (std::size_t)size_i;
  }
  if (!is) {
    return is;
  }

  s.clear();
  if (size > 0) {
    V value;
    s.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      ReadView(is, value);
      if (!is) {
        return is;
      }
      s.emplace(std::move(value));
    }
  }
  return is;
}

template <typename T>
bool SerializeToString(const T& t, std::string* buf) {
  buf->clear();
  OutputStringStream os;
  os.SetView(buf);
  os << t;
  return (bool)os;
}

template <typename T>
bool ParseFromString(const std::string& buf, T* t) {
  InputStringStream is;
  is.SetView(buf);
  is >> *t;
  return (bool)is;
}

template <typename T>
bool ParseFromArray(const char* buf, std::size_t size, T* t) {
  InputStringStream is;
  is.SetView(buf, size);
  is >> *t;
  return (bool)is;
}

template <typename T>
bool ParseViewFromString(const std::string& buf, T* t) {
  InputStringStream is;
  is.SetView(buf);
  ReadView(is, *t);
  return (bool)is;
}

template <typename T>
bool ParseViewFromArray(const char* buf, std::size_t size, T* t) {
  InputStringStream is;
  is.SetView(buf, size);
  ReadView(is, *t);
  return (bool)is;
}

}  // namespace deepx_core
