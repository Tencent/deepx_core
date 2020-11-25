// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#if OS_WIN == 1
#if !defined WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <deepx_core/dx_log.h>
#include <hdfs_c.h>
#include <zlib.h>
#include <algorithm>  // std::find_if, std::sort
#include <cerrno>
#include <cstdio>
#include <cstdlib>  // std::getenv
#include <cstring>  // std::memcpy
#include <ctime>

#if HAVE_STREAM_GFLAGS == 1
#include <gflags/gflags.h>
DEFINE_string(hdfs_ugi, "", "HDFS config string for hadoop.job.ugi");
DEFINE_string(hdfs_user, "", "HDFS user name");
#else
namespace {

std::string FLAGS_hdfs_ugi;
std::string FLAGS_hdfs_user;

}  // namespace
#endif

namespace deepx_core {

/************************************************************************/
/* Stream functions */
/************************************************************************/
bool IsDirSeparator(char c) noexcept { return c == '\\' || c == '/'; }

std::string basename(const std::string& path) {
  if (path == "/") {
    return "/";
  }

  std::string new_path = path;
  while (!new_path.empty()) {
    if (IsDirSeparator(new_path.back())) {
      new_path.pop_back();
    } else {
      break;
    }
  }

  auto it = std::find_if(new_path.rbegin(), new_path.rend(), IsDirSeparator);
  return std::string(it.base(), new_path.end());
}

std::string dirname(const std::string& path) {
  if (path == "/") {
    return "/";
  }

  if (path == "." || path == "..") {
    return ".";
  }

  std::string new_path = path;
  while (!new_path.empty()) {
    if (IsDirSeparator(new_path.back())) {
      new_path.pop_back();
    } else {
      break;
    }
  }

  auto it = std::find_if(new_path.rbegin(), new_path.rend(), IsDirSeparator);
  if (it == new_path.rend()) {
    return ".";
  }

  ++it;
  return std::string(new_path.begin(), it.base());
}

std::string CanonicalizePath(const std::string& path) {
  if (path.empty()) {
    return "";
  }

  std::string new_path = path;
  while (!new_path.empty()) {
    char c = new_path.back();
    if (IsDirSeparator(c)) {
      new_path.pop_back();
    } else {
      break;
    }
  }

  if (new_path.empty()) {
    return "/";
  }

  return new_path;
}

void CanonicalizePath(std::string* path) { *path = CanonicalizePath(*path); }

bool IsHDFSPath(const std::string& path) noexcept {
  return path.size() >= 7 && path.compare(0, 7, "hdfs://") == 0;
}

bool IsGzipFile(const std::string& file) noexcept {
  return file.size() >= 3 && file.compare(file.size() - 3, 3, ".gz") == 0;
}

std::string StdinStdoutPath() { return "-"; }

bool IsStdinStdoutPath(const std::string& path) noexcept { return path == "-"; }

static bool LocalFileExists(const std::string& file) noexcept {
  LocalFileSystem fs;
  FileStat stat;
  return fs.Stat(file, &stat) && stat.IsFile();
}

/************************************************************************/
/* FilePath */
/************************************************************************/
FilePath FilePath::basename() const { return deepx_core::basename(path_); }

FilePath FilePath::dirname() const { return deepx_core::dirname(path_); }

FilePath FilePath::canonical() const { return CanonicalizePath(path_); }

/************************************************************************/
/* FileStat */
/************************************************************************/
void FileStat::clear() noexcept {
  exists_ = 0;
  type_ = FILE_TYPE_NONE;
  file_size_ = 0;
}

FileStat FileStat::StdinStdout() noexcept {
  FileStat stat;
  stat.exists_ = 1;
  stat.type_ = FILE_TYPE_OTHER;
  stat.file_size_ = 0;
  return stat;
}

/************************************************************************/
/* FileSystem */
/************************************************************************/
bool FileSystem::Exists(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.Exists();
}

bool FileSystem::IsDir(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.IsDir();
}

bool FileSystem::IsFile(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.IsFile();
}

bool FileSystem::IsRegFile(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.IsRegFile();
}

bool FileSystem::IsSymLink(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.IsSymLink();
}

bool FileSystem::IsOther(const FilePath& path) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  return stat.IsOther();
}

bool FileSystem::GetFileSize(const FilePath& path, std::size_t* size) {
  FileStat stat;
  if (!Stat(path, &stat)) {
    return false;
  }
  *size = stat.GetFileSize();
  return true;
}

bool FileSystem::BackupIfExists(const FilePath& old_path, FilePath* new_path) {
  if (IsStdinStdoutPath(old_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid old_path: %s.", old_path.c_str());
  }

  if (Exists(old_path)) {
    std::time_t now = std::time(nullptr);
    *new_path = old_path.str() + "." + std::to_string(now);
    return Move(old_path, *new_path);
  }
  return false;
}

/************************************************************************/
/* LocalFileSystem */
/************************************************************************/
#if OS_WIN == 1
bool LocalFileSystem::Stat(const FilePath& path, FileStat* stat) {
  if (IsStdinStdoutPath(path.str())) {
    *stat = FileStat::StdinStdout();
    return true;
  }

  WIN32_FILE_ATTRIBUTE_DATA attr;
  if (GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &attr) == 0) {
    stat->clear();
    return false;
  }

  stat->set_exists(1);
  int type;
  if (attr.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    type = FILE_TYPE_DIR;
  } else if (attr.dwFileAttributes &
             (FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_ARCHIVE |
              FILE_ATTRIBUTE_NORMAL)) {
    type = FILE_TYPE_REG_FILE;
  } else if (attr.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    type = FILE_TYPE_SYM_LINK;
  } else {
    type = FILE_TYPE_OTHER;
  }
  stat->set_type(type);
  LARGE_INTEGER size;
  size.HighPart = attr.nFileSizeHigh;
  size.LowPart = attr.nFileSizeLow;
  stat->set_file_size((std::size_t)size.QuadPart);
  return true;
}

static bool _List(LocalFileSystem* fs, const FilePath& path, bool skip_dir,
                  std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  HANDLE handle;
  WIN32_FIND_DATAA find_data;
  std::string pattern = path.str() + "/*";
  std::string child_path;
  FileStat child_stat;
  handle = FindFirstFileA(pattern.c_str(), &find_data);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  do {
    // skip . .. and stealth files
    if (find_data.cFileName[0] == '.') {
      continue;
    }

    child_path = path.str() + "/" + find_data.cFileName;
    if (!fs->Stat(child_path, &child_stat)) {
      (void)FindClose(handle);
      return false;
    }

    if (skip_dir && child_stat.IsDir()) {
      continue;
    } else {
      children->emplace_back(child_path, child_stat);
    }
  } while (FindNextFileA(handle, &find_data) != 0);
  (void)FindClose(handle);
  return true;
}

static bool _ListRecursive(
    LocalFileSystem* fs, const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  HANDLE handle;
  WIN32_FIND_DATAA find_data;
  std::string pattern = path.str() + "/*";
  std::string child_path;
  FileStat child_stat;
  handle = FindFirstFileA(pattern.c_str(), &find_data);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  do {
    // skip . .. and stealth files
    if (find_data.cFileName[0] == '.') {
      continue;
    }

    child_path = path.str() + "/" + find_data.cFileName;
    if (!fs->Stat(child_path, &child_stat)) {
      (void)FindClose(handle);
      return false;
    }

    if (child_stat.IsDir()) {
      if (!skip_dir) {
        children->emplace_back(child_path, child_stat);
      }
      if (!_ListRecursive(fs, child_path, skip_dir, children)) {
        (void)FindClose(handle);
        return false;
      }
    } else {
      children->emplace_back(child_path, child_stat);
    }
  } while (FindNextFileA(handle, &find_data) != 0);
  (void)FindClose(handle);
  return true;
}

bool LocalFileSystem::MakeDir(const FilePath& dir) {
  if (IsStdinStdoutPath(dir.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid dir: %s.", dir.c_str());
  }

  if (CreateDirectoryA(dir.c_str(), nullptr) == 0) {
    DXERROR("Failed to CreateDirectoryA, dir=%s, errno=%d.", dir.c_str(),
            (int)GetLastError());
    return false;
  }
  return true;
}

bool LocalFileSystem::Move(const FilePath& old_path, const FilePath& new_path) {
  if (IsStdinStdoutPath(old_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid old_path: %s.", old_path.c_str());
  }

  if (IsStdinStdoutPath(new_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid new_path: %s.", new_path.c_str());
  }

  if (MoveFileA(old_path.c_str(), new_path.c_str()) == 0) {
    DXERROR("Failed to MoveFileA, old_path=%s, new_path=%s, errno=%d.",
            old_path.c_str(), new_path.c_str(), (int)GetLastError());
    return false;
  }
  return true;
}
#else
bool LocalFileSystem::Stat(const FilePath& path, FileStat* _stat) {
  if (IsStdinStdoutPath(path.str())) {
    *_stat = FileStat::StdinStdout();
    return true;
  }

  struct stat buf;
  if (lstat(path.c_str(), &buf) == -1) {
    _stat->clear();
    return false;
  }

  _stat->set_exists(1);
  int type;
  if (S_ISDIR(buf.st_mode)) {
    type = FILE_TYPE_DIR;
  } else if (S_ISREG(buf.st_mode)) {
    type = FILE_TYPE_REG_FILE;
  } else if (S_ISLNK(buf.st_mode)) {
    type = FILE_TYPE_SYM_LINK;
  } else {
    type = FILE_TYPE_OTHER;
  }
  _stat->set_type(type);
  _stat->set_file_size((std::size_t)buf.st_size);
  return true;
}

static bool _List(LocalFileSystem* fs, const FilePath& path, bool skip_dir,
                  std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  DIR* d;
  dirent* f;
  std::string child_path;
  FileStat child_stat;
  d = opendir(path.c_str());
  if (d == nullptr) {
    return false;
  }

  while ((f = readdir(d))) {
    // skip . .. and stealth files
    if (f->d_name[0] == '.') {
      continue;
    }

    child_path = path.str() + "/" + f->d_name;
    if (!fs->Stat(child_path, &child_stat)) {
      (void)closedir(d);
      return false;
    }

    if (skip_dir && child_stat.IsDir()) {
      continue;
    } else {
      children->emplace_back(child_path, child_stat);
    }
  }
  (void)closedir(d);
  return true;
}

static bool _ListRecursive(
    LocalFileSystem* fs, const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  DIR* d;
  dirent* f;
  std::string child_path;
  FileStat child_stat;
  d = opendir(path.c_str());
  if (d == nullptr) {
    return false;
  }

  while ((f = readdir(d))) {
    // skip . .. and stealth files
    if (f->d_name[0] == '.') {
      continue;
    }

    child_path = path.str() + "/" + f->d_name;
    if (!fs->Stat(child_path, &child_stat)) {
      (void)closedir(d);
      return false;
    }

    if (child_stat.IsDir()) {
      if (!skip_dir) {
        children->emplace_back(child_path, child_stat);
      }
      if (!_ListRecursive(fs, child_path, skip_dir, children)) {
        (void)closedir(d);
        return false;
      }
    } else {
      children->emplace_back(child_path, child_stat);
    }
  }
  (void)closedir(d);
  return true;
}

bool LocalFileSystem::MakeDir(const FilePath& dir) {
  if (IsStdinStdoutPath(dir.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid dir: %s.", dir.c_str());
  }

  if (mkdir(dir.c_str(), 0755) == -1) {
    DXERROR("Failed to mkdir, dir=%s, errno=%d(%s).", dir.c_str(), errno,
            strerror(errno));
    return false;
  }
  return true;
}

bool LocalFileSystem::Move(const FilePath& old_path, const FilePath& new_path) {
  if (IsStdinStdoutPath(old_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid old_path: %s.", old_path.c_str());
  }

  if (IsStdinStdoutPath(new_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid new_path: %s.", new_path.c_str());
  }

  if (rename(old_path.c_str(), new_path.c_str()) == -1) {
    DXERROR("Failed to rename, old_path=%s, new_path=%s, errno=%d(%s).",
            old_path.c_str(), new_path.c_str(), errno, strerror(errno));
    return false;
  }
  return true;
}
#endif

bool LocalFileSystem::List(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  children->clear();

  if (IsStdinStdoutPath(path.str())) {
    children->emplace_back(path, FileStat::StdinStdout());
    return true;
  }

  if (!_List(this, path, skip_dir, children)) {
    return false;
  }

  std::sort(children->begin(), children->end(),
            [](const std::pair<FilePath, FileStat>& left,
               const std::pair<FilePath, FileStat>& right) {
              return left.first.str() < right.first.str();
            });
  return true;
}

bool LocalFileSystem::ListRecursive(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  children->clear();

  if (IsStdinStdoutPath(path.str())) {
    children->emplace_back(path, FileStat::StdinStdout());
    return true;
  }

  if (!_ListRecursive(this, path, skip_dir, children)) {
    return false;
  }

  std::sort(children->begin(), children->end(),
            [](const std::pair<FilePath, FileStat>& left,
               const std::pair<FilePath, FileStat>& right) {
              return left.first.str() < right.first.str();
            });
  return true;
}

/************************************************************************/
/* GetLine */
/************************************************************************/
InputStream& GetLine(InputStream& is, std::string& line) {
  return GetLine(is, line, '\n');
}

InputStream& GetLine(InputStream& is, std::string& line, char delim) {
  char c;
  line.clear();
  for (;;) {
    c = is.ReadChar();
    if (!is || c == delim) {
      break;
    }
    line.push_back(c);
  }
  return is;
}

/************************************************************************/
/* BufferedInputStream */
/************************************************************************/
BufferedInputStream::BufferedInputStream(InputStream* is, std::size_t buf_size)
    : is_(is) {
  if (is_ == nullptr) {
    bad_ = 1;
    buf_size_ = 0;
    begin_ = nullptr;
    cur_ = nullptr;
    end_ = nullptr;
  } else {
    bad_ = 0;
    buf_size_ = buf_size;
    buf_.resize(buf_size_);
    begin_ = &buf_[0];
    cur_ = begin_;
    end_ = cur_;
  }
}

std::size_t BufferedInputStream::FillEmptyBuf() {
  std::size_t avail_bytes = is_->Read(begin_, buf_size_);
  if (avail_bytes == 0) {
    bad_ = 1;
    return 0;
  }

  cur_ = begin_;
  end_ = cur_ + avail_bytes;
  return avail_bytes;
}

std::size_t BufferedInputStream::EnsureBuf(std::size_t need_bytes) {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes >= need_bytes) {
    return avail_bytes;
  }

  std::size_t cur_offset = cur_ - begin_;
  std::size_t end_offset = end_ - begin_;
  std::size_t need_buf_size = cur_offset + need_bytes;
  if (need_buf_size > buf_size_) {
    buf_size_ = need_buf_size;
    buf_.resize(need_buf_size);
    begin_ = &buf_[0];
    cur_ = begin_ + cur_offset;
    end_ = begin_ + end_offset;
  }

  std::size_t bytes = is_->Read(end_, buf_size_ - end_offset);
  if (bytes == 0) {
    bad_ = 1;
    return avail_bytes;
  }

  avail_bytes += bytes;
  end_ += bytes;
  return avail_bytes;
}

std::size_t BufferedInputStream::Read(void* data, std::size_t size) {
  std::size_t need_bytes = size;
  std::size_t avail_bytes = end_ - cur_;
  for (;;) {
    if (avail_bytes >= need_bytes) {
      std::memcpy(data, cur_, need_bytes);
      cur_ += need_bytes;
      return size;
    } else if (avail_bytes > 0) {
      std::memcpy(data, cur_, avail_bytes);
      data = (char*)data + avail_bytes;
      cur_ += avail_bytes;
      need_bytes -= avail_bytes;
    }

    avail_bytes = FillEmptyBuf();
    if (avail_bytes == 0) {
      return size - need_bytes;
    }
  }
}

char BufferedInputStream::ReadChar() {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes == 0) {
    avail_bytes = FillEmptyBuf();
  }

  if (avail_bytes > 0) {
    return *cur_++;
  }
  return (char)-1;
}

std::size_t BufferedInputStream::Peek(void* data, std::size_t size) {
  std::size_t avail_bytes = EnsureBuf(size);
  if (size > avail_bytes) {
    size = avail_bytes;
  }
  std::memcpy(data, cur_, size);
  return size;
}

/************************************************************************/
/* GunzipInputStream */
/************************************************************************/
GunzipInputStream::GunzipInputStream(InputStream* is, std::size_t buf_size)
    : is_(is) {
  if (is_ == nullptr) {
    bad_ = 1;
    zs_ = nullptr;
    comp_buf_size_ = 0;
    comp_begin_ = nullptr;
    buf_size_ = 0;
    begin_ = nullptr;
    cur_ = nullptr;
    end_ = nullptr;
  } else {
    bad_ = 0;
    zs_ = new z_stream;
    std::memset(zs_, 0, sizeof(z_stream));
    (void)inflateInit2((z_stream*)zs_, (16 + MAX_WBITS));
    comp_buf_size_ = buf_size;
    comp_buf_.resize(comp_buf_size_);
    comp_begin_ = &comp_buf_[0];
    buf_size_ = comp_buf_size_ * 8;  // magic number
    buf_.resize(buf_size_);
    begin_ = &buf_[0];
    cur_ = begin_;
    end_ = cur_;
  }
}

GunzipInputStream::~GunzipInputStream() {
  if (zs_) {
    (void)inflateEnd((z_stream*)zs_);
    delete (z_stream*)zs_;
  }
}

std::size_t GunzipInputStream::FillEmptyBuf() {
  z_stream* zs = (z_stream*)zs_;  // NOLINT
  if (zs->avail_in == 0) {
    std::size_t comp_bytes = is_->Read(comp_begin_, comp_buf_size_);
    if (comp_bytes == 0) {
      bad_ = 1;
      return 0;
    }
    zs->next_in = (Bytef*)comp_begin_;
    zs->avail_in = (uInt)comp_bytes;
  }
  zs->next_out = (Bytef*)&buf_[0];
  zs->avail_out = (uInt)buf_size_;

  int ok = inflate(zs, Z_SYNC_FLUSH);
  if (ok != Z_OK && ok != Z_STREAM_END) {
    bad_ = 1;
    return 0;
  }

  std::size_t avail_bytes = buf_size_ - (std::size_t)zs->avail_out;
  cur_ = begin_;
  end_ = cur_ + avail_bytes;
  return avail_bytes;
}

std::size_t GunzipInputStream::EnsureBuf(std::size_t need_bytes) {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes >= need_bytes) {
    return avail_bytes;
  }

  z_stream* zs = (z_stream*)zs_;  // NOLINT
  std::size_t cur_offset = cur_ - begin_;
  std::size_t end_offset = end_ - begin_;
  std::size_t need_buf_size = cur_offset + need_bytes;
  if (need_buf_size > buf_size_) {
    buf_size_ = need_buf_size;
    buf_.resize(need_buf_size);
    begin_ = &buf_[0];
    cur_ = begin_ + cur_offset;
    end_ = begin_ + end_offset;
  }
  zs->next_out = (Bytef*)end_;
  zs->avail_out = (uInt)(buf_size_ - end_offset);

  for (;;) {
    if (zs->avail_in == 0) {
      std::size_t comp_bytes = is_->Read(comp_begin_, comp_buf_size_);
      if (comp_bytes == 0) {
        bad_ = 1;
        return avail_bytes;
      }
      zs->next_in = (Bytef*)comp_begin_;
      zs->avail_in = (uInt)comp_bytes;
    }

    std::size_t prev_avail_out = (std::size_t)zs->avail_out;  // NOLINT
    int ok = inflate(zs, Z_SYNC_FLUSH);
    if (ok != Z_OK && ok != Z_STREAM_END) {
      bad_ = 1;
      return avail_bytes;
    }

    std::size_t bytes = prev_avail_out - (std::size_t)zs->avail_out;
    avail_bytes += bytes;
    end_ += bytes;
    if (avail_bytes >= need_bytes) {
      return avail_bytes;
    }
  }
}

std::size_t GunzipInputStream::Read(void* data, std::size_t size) {
  std::size_t need_bytes = size;
  std::size_t avail_bytes = end_ - cur_;
  for (;;) {
    if (avail_bytes >= need_bytes) {
      std::memcpy(data, cur_, need_bytes);
      cur_ += need_bytes;
      return size;
    } else if (avail_bytes > 0) {
      std::memcpy(data, cur_, avail_bytes);
      data = (char*)data + avail_bytes;
      cur_ += avail_bytes;
      need_bytes -= avail_bytes;
    }

    avail_bytes = FillEmptyBuf();
    if (avail_bytes == 0) {
      return size - need_bytes;
    }
  }
}

char GunzipInputStream::ReadChar() {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes == 0) {
    avail_bytes = FillEmptyBuf();
  }

  if (avail_bytes > 0) {
    return *cur_++;
  }
  return (char)-1;
}

std::size_t GunzipInputStream::Peek(void* data, std::size_t size) {
  std::size_t avail_bytes = EnsureBuf(size);
  if (size > avail_bytes) {
    size = avail_bytes;
  }
  std::memcpy(data, cur_, size);
  return size;
}

/************************************************************************/
/* CFileStream */
/************************************************************************/
#if OS_DARWIN == 1
#define fgetc getc_unlocked
#define feof feof_unlocked
#define ferror ferror_unlocked
#elif OS_LINUX == 1
#define fwrite fwrite_unlocked
#define fflush fflush_unlocked
#define fread fread_unlocked
#define fgetc getc_unlocked
#define feof feof_unlocked
#define ferror ferror_unlocked
#endif

static const char* GetFopenMode(int _mode) {
  int binary = _mode & FILE_OPEN_MODE_BINARY;
  int mode = _mode & ~FILE_OPEN_MODE_BINARY;
  if (mode == FILE_OPEN_MODE_IN) {
    return binary ? "rb" : "r";
  } else if (mode == FILE_OPEN_MODE_OUT ||
             mode == (FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_TRUNCATE)) {
    return binary ? "wb" : "w";
  } else if (mode == FILE_OPEN_MODE_APPEND ||
             mode == (FILE_OPEN_MODE_APPEND | FILE_OPEN_MODE_OUT)) {
    return binary ? "ab" : "a";
  } else if (mode == (FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_IN)) {
    return binary ? "r+b" : "r+";
  } else if (mode == (FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_IN |
                      FILE_OPEN_MODE_TRUNCATE)) {
    return binary ? "w+b" : "w+";
  } else if (mode == (FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_IN |
                      FILE_OPEN_MODE_APPEND) ||
             mode == (FILE_OPEN_MODE_IN | FILE_OPEN_MODE_APPEND)) {
    return binary ? "a+b" : "a+";
  } else {
    DXTHROW_INVALID_ARGUMENT("Invalid mode for fopen: %d.", _mode);
  }
}

CFileStream::CFileStream() { bad_ = 1; }

CFileStream::~CFileStream() { Close(); }

std::size_t CFileStream::Write(const void* data, std::size_t size) {
  std::size_t bytes = fwrite(data, 1, size, (FILE*)f_);
  if (bytes < size) {
    bad_ = 1;
  }
  return bytes;
}

bool CFileStream::Flush() {
  if ((mode_ & FILE_OPEN_MODE_OUT) && f_) {
    if (fflush((FILE*)f_) == EOF) {
      DXERROR("Failed to fflush, errno=%d(%s).", errno, strerror(errno));
      return false;
    }
  }
  return true;
}

std::size_t CFileStream::Read(void* data, std::size_t size) {
  std::size_t bytes = fread(data, 1, size, (FILE*)f_);
  if (bytes < size) {
    bad_ = 1;
  }
  return bytes;
}

char CFileStream::ReadChar() {
  int c = fgetc((FILE*)f_);
  if (c == EOF) {
    bad_ = feof((FILE*)f_) || ferror((FILE*)f_);
  }
  return (char)c;
}

std::size_t CFileStream::Peek(void* data, std::size_t size) {
  long offset = ftell((FILE*)f_);  // NOLINT
  if (offset == -1) {
    DXTHROW_RUNTIME_ERROR("Failed to ftell, errno=%d(%s).", errno,
                          strerror(errno));
  }
  std::size_t bytes = Read(data, size);
  if (fseek((FILE*)f_, offset, SEEK_SET) == -1) {
    DXTHROW_RUNTIME_ERROR("Failed to fseek, errno=%d(%s).", errno,
                          strerror(errno));
  }
  return bytes;
}

bool CFileStream::Open(const std::string& file, int mode) {
  Close();

  if (IsStdinStdoutPath(file)) {
    if (mode & FILE_OPEN_MODE_IN) {
      f_ = stdin;
    } else if (mode & FILE_OPEN_MODE_OUT) {
      f_ = stdout;
    } else {
      DXTHROW_INVALID_ARGUMENT("Invalid mode: %d.", mode);
    }
  } else {
    f_ = fopen(file.c_str(), GetFopenMode(mode));
  }

  if (f_ == nullptr) {
    bad_ = 1;
    mode_ = FILE_OPEN_MODE_NONE;
    return false;
  } else {
    bad_ = 0;
    mode_ = mode;
    return true;
  }
}

bool CFileStream::IsOpen() const noexcept { return f_; }

void CFileStream::Close() noexcept {
  if (f_) {
    if (f_ != stdin && f_ != stdout) {
      (void)fclose((FILE*)f_);
    }
  }
  bad_ = 1;
  f_ = nullptr;
  mode_ = FILE_OPEN_MODE_NONE;
}

/************************************************************************/
/* OutputStringStream */
/************************************************************************/
std::size_t OutputStringStream::Write(const void* data, std::size_t size) {
  buf_ptr_->insert(buf_ptr_->end(), (const char*)data,
                   (const char*)data + size);
  return size;
}

void OutputStringStream::BeginMessage() {
  buf_ptr_->clear();
  int place_holder = 0;
  *this << place_holder;
}

void OutputStringStream::EndMessage() noexcept {
  *(int*)(&(*buf_ptr_)[0]) = (int)buf_ptr_->size();
}

/************************************************************************/
/* InputStringStream */
/************************************************************************/
void InputStringStream::Init(const char* data, std::size_t size) {
  if (size == 0) {
    bad_ = 1;
    cur_ = nullptr;
    end_ = nullptr;
  } else {
    bad_ = 0;
    cur_ = data;
    end_ = cur_ + size;
  }
}

InputStringStream::InputStringStream() { Init(nullptr, 0); }

std::size_t InputStringStream::Read(void* data, std::size_t size) {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes >= size) {
    std::memcpy(data, cur_, size);
    cur_ += size;
    return size;
  } else if (avail_bytes > 0) {
    std::memcpy(data, cur_, avail_bytes);
    cur_ += avail_bytes;
    bad_ = 1;
    return avail_bytes;
  } else {
    bad_ = 1;
    return 0;
  }
}

char InputStringStream::ReadChar() {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes > 0) {
    return *cur_++;
  }
  bad_ = 1;
  return (char)-1;
}

std::size_t InputStringStream::Peek(void* data, std::size_t size) {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes >= size) {
    std::memcpy(data, cur_, size);
    return size;
  } else if (avail_bytes > 0) {
    std::memcpy(data, cur_, avail_bytes);
    bad_ = 1;
    return avail_bytes;
  } else {
    bad_ = 1;
    return 0;
  }
}

std::size_t InputStringStream::Skip(std::size_t size) {
  std::size_t avail_bytes = end_ - cur_;
  if (avail_bytes >= size) {
    cur_ += size;
    return size;
  } else if (avail_bytes > 0) {
    cur_ += avail_bytes;
    bad_ = 1;
    return avail_bytes;
  } else {
    bad_ = 1;
    return 0;
  }
}

/************************************************************************/
/* HDFS functions */
/************************************************************************/
namespace {

#define DEFINE_HDFS_FUNC(name) decltype(&name) p##name = nullptr
DEFINE_HDFS_FUNC(hdfsNewBuilder);
DEFINE_HDFS_FUNC(hdfsBuilderSetForceNewInstance);
DEFINE_HDFS_FUNC(hdfsBuilderConfSetStr);
DEFINE_HDFS_FUNC(hdfsBuilderSetNameNode);
DEFINE_HDFS_FUNC(hdfsBuilderSetNameNodePort);
DEFINE_HDFS_FUNC(hdfsBuilderConnect);
DEFINE_HDFS_FUNC(hdfsConnectAsUserNewInstance);
DEFINE_HDFS_FUNC(hdfsConnectNewInstance);
DEFINE_HDFS_FUNC(hdfsDisconnect);
DEFINE_HDFS_FUNC(hdfsFreeFileInfo);
DEFINE_HDFS_FUNC(hdfsGetPathInfo);
DEFINE_HDFS_FUNC(hdfsListDirectory);
DEFINE_HDFS_FUNC(hdfsGlobStatus);
DEFINE_HDFS_FUNC(hdfsRename);
DEFINE_HDFS_FUNC(hdfsCreateDirectory);
DEFINE_HDFS_FUNC(hdfsOpenFile);
DEFINE_HDFS_FUNC(hdfsCloseFile);
DEFINE_HDFS_FUNC(hdfsSeek);
DEFINE_HDFS_FUNC(hdfsTell);
DEFINE_HDFS_FUNC(hdfsRead);
DEFINE_HDFS_FUNC(hdfsWrite);
DEFINE_HDFS_FUNC(hdfsHFlush);
#undef DEFINE_HDFS_FUNC

}  // namespace

#if OS_POSIX == 1
namespace {

bool LoadHDFSFunc(const char* so) {
  void* handle;
  if (so) {
    handle = dlopen(so, RTLD_NOW);
    if (handle == nullptr) {
      return false;
    }
  } else {
    handle = RTLD_DEFAULT;
  }

#define LOAD_HDFS_FUNC(name)                    \
  do {                                          \
    *(void**)(&p##name) = dlsym(handle, #name); \
    if (p##name == nullptr) {                   \
      if (handle) {                             \
        dlclose(handle);                        \
      }                                         \
      return false;                             \
    }                                           \
  } while (0)
#define LOAD_OPTIONAL_HDFS_FUNC(name)           \
  do {                                          \
    *(void**)(&p##name) = dlsym(handle, #name); \
  } while (0)
  LOAD_OPTIONAL_HDFS_FUNC(hdfsNewBuilder);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsBuilderSetForceNewInstance);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsBuilderConfSetStr);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsBuilderSetNameNode);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsBuilderSetNameNodePort);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsBuilderConnect);
  LOAD_HDFS_FUNC(hdfsConnectAsUserNewInstance);
  LOAD_HDFS_FUNC(hdfsConnectNewInstance);
  LOAD_HDFS_FUNC(hdfsDisconnect);
  LOAD_HDFS_FUNC(hdfsFreeFileInfo);
  LOAD_HDFS_FUNC(hdfsGetPathInfo);
  LOAD_HDFS_FUNC(hdfsListDirectory);
  LOAD_OPTIONAL_HDFS_FUNC(hdfsGlobStatus);
  LOAD_HDFS_FUNC(hdfsRename);
  LOAD_HDFS_FUNC(hdfsCreateDirectory);
  LOAD_HDFS_FUNC(hdfsOpenFile);
  LOAD_HDFS_FUNC(hdfsCloseFile);
  LOAD_HDFS_FUNC(hdfsSeek);
  LOAD_HDFS_FUNC(hdfsTell);
  LOAD_HDFS_FUNC(hdfsRead);
  LOAD_HDFS_FUNC(hdfsWrite);
  LOAD_HDFS_FUNC(hdfsHFlush);
#undef LOAD_HDFS_FUNC
#undef LOAD_OPTIONAL_HDFS_FUNC

  if (so) {
    DXINFO("Loaded libhdfs functions from %s.", so);
  } else {
    DXINFO("Loaded libhdfs functions.");
  }
  return true;
}

class HDFSFuncInitializer {
 private:
  int loaded_ = 0;

 public:
  bool loaded() const noexcept { return loaded_ != 0; }

 public:
  HDFSFuncInitializer() {
    std::vector<const char*> sos = {"./libhdfs.so", "./libhdfs.so.1",
                                    "libhdfs.so", "libhdfs.so.1", nullptr};
    for (const char* so : sos) {
      if (LoadHDFSFunc(so)) {
        loaded_ = 1;
        return;
      }
    }
    DXINFO("libhdfs is unavailable.");
  }
} hdfs_func_initializer;

}  // namespace

bool HasHDFS() noexcept { return hdfs_func_initializer.loaded(); }
#else
bool HasHDFS() noexcept { return false; }
#endif

namespace {

/************************************************************************/
/* HDFSRawHandleManager */
/************************************************************************/
class HDFSRawHandleManager {
 public:
  void* Get(const std::string& host, uint16_t port, const std::string& ugi,
            const std::string& user) {
    void* raw_handle;
    if (!ugi.empty()) {
      if (phdfsNewBuilder == nullptr ||
          phdfsBuilderSetForceNewInstance == nullptr ||
          phdfsBuilderSetNameNode == nullptr ||
          phdfsBuilderSetNameNodePort == nullptr ||
          phdfsBuilderConfSetStr == nullptr || phdfsBuilderConnect == nullptr) {
        DXTHROW_RUNTIME_ERROR("hdfs builder functions were not loaded.");
      }
      hdfsBuilder* builder = phdfsNewBuilder();
      phdfsBuilderSetForceNewInstance(builder);
      phdfsBuilderSetNameNode(builder, host.c_str());
      phdfsBuilderSetNameNodePort(builder, port);
      phdfsBuilderConfSetStr(builder, "hadoop.job.ugi", ugi.c_str());
      raw_handle = phdfsBuilderConnect(builder);
      if (raw_handle == nullptr) {
        DXERROR(
            "Failed to hdfsBuilderConnect, host=%s, port=%d, ugi=%s, "
            "errno=%d(%s).",
            host.c_str(), (int)port, ugi.c_str(), errno, strerror(errno));
      }
    } else if (!user.empty()) {
      raw_handle =
          phdfsConnectAsUserNewInstance(host.c_str(), port, user.c_str());
      if (raw_handle == nullptr) {
        DXERROR(
            "Failed to hdfsConnectAsUserNewInstance, host=%s, port=%d, "
            "user=%s, errno=%d(%s).",
            host.c_str(), (int)port, user.c_str(), errno, strerror(errno));
      }
    } else {
      raw_handle = phdfsConnectNewInstance(host.c_str(), port);
      if (raw_handle == nullptr) {
        DXERROR(
            "Failed to hdfsConnectNewInstance, host=%s, port=%d, errno=%d(%s).",
            host.c_str(), (int)port, errno, strerror(errno));
      }
    }
    return raw_handle;
  }

  void Release(void* raw_handle) noexcept {
    (void)phdfsDisconnect((hdfsFS)raw_handle);
  }
} hdfs_raw_handle_manager;

}  // namespace

/************************************************************************/
/* HDFSHandle */
/************************************************************************/
HDFSHandle::~HDFSHandle() { Close(); }

bool HDFSHandle::Connect(const char* name_node_host, uint16_t name_node_port) {
  Close();

  std::string ugi, user;
  const char* DEEPX_HDFS_UGI = std::getenv("DEEPX_HDFS_UGI");
  const char* DEEPX_HDFS_USER = std::getenv("DEEPX_HDFS_USER");
  if (DEEPX_HDFS_UGI) {
    ugi = DEEPX_HDFS_UGI;
  } else {
    ugi = FLAGS_hdfs_ugi;
  }
  if (DEEPX_HDFS_USER) {
    user = DEEPX_HDFS_USER;
  } else {
    user = FLAGS_hdfs_user;
  }

  raw_handle_ =
      hdfs_raw_handle_manager.Get(name_node_host, name_node_port, ugi, user);
  return IsOpen();
}

bool HDFSHandle::Connect(const std::string& name_node_host,
                         uint16_t name_node_port) {
  return Connect(name_node_host.c_str(), name_node_port);
}

bool HDFSHandle::ConnectDefault() { return Connect("default", 0); }

bool HDFSHandle::IsOpen() const noexcept { return raw_handle_; }

void HDFSHandle::Close() noexcept {
  if (raw_handle_) {
    hdfs_raw_handle_manager.Release(raw_handle_);
    raw_handle_ = nullptr;
  }
}

/************************************************************************/
/* HDFSFileSystem */
/************************************************************************/
static void HDFSInfo2Stat(const hdfsFileInfo& info, FileStat* stat) noexcept {
  stat->set_exists(1);
  if (info.mKind == kObjectKindDirectory) {
    stat->set_type(FILE_TYPE_DIR);
  } else if (info.mKind == kObjectKindFile) {
    stat->set_type(FILE_TYPE_REG_FILE);
  } else {
    stat->set_type(FILE_TYPE_OTHER);
  }
  stat->set_file_size((std::size_t)info.mSize);
}

HDFSFileSystem::HDFSFileSystem(HDFSHandle* handle) : handle_(handle) {}

bool HDFSFileSystem::Stat(const FilePath& path, FileStat* stat) {
  if (IsStdinStdoutPath(path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid path: %s.", path.c_str());
  }

  hdfsFileInfo* info =
      phdfsGetPathInfo((hdfsFS)handle_->raw_handle(),  // NOLINT
                       path.c_str());
  if (info == nullptr) {
    DXERROR("Failed to hdfsGetPathInfo, path=%s, errno=%d(%s).", path.c_str(),
            errno, strerror(errno));
    stat->clear();
    return false;
  }

  HDFSInfo2Stat(*info, stat);
  phdfsFreeFileInfo(info, 1);
  return true;
}

static bool IsGlobPattern(const std::string& pattern) noexcept {
  if (pattern.find('?') != std::string::npos) {
    return true;
  }
  if (pattern.find('*') != std::string::npos) {
    return true;
  }
  if (pattern.find('[') != std::string::npos) {
    return true;
  }
  return false;
}

static bool _GlobPattern(hdfsFS handle, const FilePath& pattern, bool skip_dir,
                         std::vector<std::pair<FilePath, FileStat>>* children) {
  if (phdfsGlobStatus == nullptr) {
    DXTHROW_RUNTIME_ERROR("hdfsGlobStatus was not loaded.");
  }

  hdfsFileInfo* infos;
  int n;
  std::string child_path;
  FileStat child_stat;
  infos = phdfsGlobStatus(handle, pattern.c_str(), &n);
  if (infos == nullptr) {
    if (errno) {
      DXERROR("Failed to hdfsGlobStatus, pattern=%s, errno=%d(%s).",
              pattern.c_str(), errno, strerror(errno));
      return false;
    } else {
      // empty dir
      return true;
    }
  }

  for (int i = 0; i < n; ++i) {
    const hdfsFileInfo& info = infos[i];
    child_path = info.mName;
    HDFSInfo2Stat(info, &child_stat);
    if (skip_dir && child_stat.IsDir()) {
      continue;
    } else {
      children->emplace_back(child_path, child_stat);
    }
  }
  phdfsFreeFileInfo(infos, n);
  return true;
}

static bool _List(HDFSFileSystem* fs, hdfsFS handle, const FilePath& path,
                  bool skip_dir,
                  std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  hdfsFileInfo* infos;
  int n;
  std::string child_path;
  FileStat child_stat;
  infos = phdfsListDirectory(handle, path.c_str(), &n);
  if (infos == nullptr) {
    if (errno) {
      DXERROR("Failed to hdfsListDirectory, path=%s, errno=%d(%s).",
              path.c_str(), errno, strerror(errno));
      return false;
    } else {
      // empty dir
      return true;
    }
  }

  for (int i = 0; i < n; ++i) {
    const hdfsFileInfo& info = infos[i];
    child_path = info.mName;
    HDFSInfo2Stat(info, &child_stat);
    if (skip_dir && child_stat.IsDir()) {
      continue;
    } else {
      children->emplace_back(child_path, child_stat);
    }
  }
  phdfsFreeFileInfo(infos, n);
  return true;
}

static bool _ListRecursive(
    HDFSFileSystem* fs, hdfsFS handle, const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  FileStat stat;
  if (!fs->Stat(path, &stat)) {
    return false;
  }

  if (stat.IsFile()) {
    children->emplace_back(path, stat);
    return true;
  }

  hdfsFileInfo* infos;
  int n;
  std::string child_path;
  FileStat child_stat;
  infos = phdfsListDirectory(handle, path.c_str(), &n);
  if (infos == nullptr) {
    if (errno) {
      DXERROR("Failed to hdfsListDirectory, path=%s, errno=%d(%s).",
              path.c_str(), errno, strerror(errno));
      return false;
    } else {
      // empty dir
      return true;
    }
  }

  for (int i = 0; i < n; ++i) {
    const hdfsFileInfo& info = infos[i];
    child_path = info.mName;
    HDFSInfo2Stat(info, &child_stat);
    if (child_stat.IsDir()) {
      if (!skip_dir) {
        children->emplace_back(child_path, child_stat);
      }
      if (!_ListRecursive(fs, handle, child_path, skip_dir, children)) {
        return false;
      }
    } else {
      children->emplace_back(child_path, child_stat);
    }
  }
  phdfsFreeFileInfo(infos, n);
  return true;
}

bool HDFSFileSystem::List(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  children->clear();

  if (IsStdinStdoutPath(path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid path: %s.", path.c_str());
  }

  if (IsGlobPattern(path.str())) {
    return _GlobPattern((hdfsFS)handle_->raw_handle(),  // NOLINT
                        path, skip_dir, children);
  }

  if (!_List(this, (hdfsFS)handle_->raw_handle(),  // NOLINT
             path, skip_dir, children)) {
    return false;
  }

  std::sort(children->begin(), children->end(),
            [](const std::pair<FilePath, FileStat>& left,
               const std::pair<FilePath, FileStat>& right) {
              return left.first.str() < right.first.str();
            });
  return true;
}

bool HDFSFileSystem::ListRecursive(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  children->clear();

  if (IsStdinStdoutPath(path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid path: %s.", path.c_str());
  }

  if (IsGlobPattern(path.str())) {
    return _GlobPattern((hdfsFS)handle_->raw_handle(),  // NOLINT
                        path, skip_dir, children);
  }

  if (!_ListRecursive(this, (hdfsFS)handle_->raw_handle(),  // NOLINT
                      path, skip_dir, children)) {
    return false;
  }

  std::sort(children->begin(), children->end(),
            [](const std::pair<FilePath, FileStat>& left,
               const std::pair<FilePath, FileStat>& right) {
              return left.first.str() < right.first.str();
            });
  return true;
}

bool HDFSFileSystem::MakeDir(const FilePath& dir) {
  if (IsStdinStdoutPath(dir.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid dir: %s.", dir.c_str());
  }

  if (phdfsCreateDirectory((hdfsFS)handle_->raw_handle(),  // NOLINT
                           dir.c_str()) == -1) {
    DXERROR("Failed to hdfsCreateDirectory, dir=%s, errno=%d(%s).", dir.c_str(),
            errno, strerror(errno));
    return false;
  }
  return true;
}

bool HDFSFileSystem::Move(const FilePath& old_path, const FilePath& new_path) {
  if (IsStdinStdoutPath(old_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid old_path: %s.", old_path.c_str());
  }

  if (IsStdinStdoutPath(new_path.str())) {
    DXTHROW_INVALID_ARGUMENT("Invalid new_path: %s.", new_path.c_str());
  }

  if (phdfsRename((hdfsFS)handle_->raw_handle(),  // NOLINT
                  old_path.c_str(), new_path.c_str()) == -1) {
    DXERROR("Failed to hdfsRename, old_path=%s, new_path=%s, errno=%d(%s).",
            old_path.c_str(), new_path.c_str(), errno, strerror(errno));
    return false;
  }
  return true;
}

/************************************************************************/
/* HDFSFileStream */
/************************************************************************/
static int GetHDFSOpenMode(int mode) {
  if (mode == FILE_OPEN_MODE_IN) {
    return O_RDONLY;
  } else if (mode == FILE_OPEN_MODE_OUT) {
    return O_WRONLY;
  } else if (mode == (FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_APPEND)) {
    return O_WRONLY | O_APPEND;
  } else {
    DXTHROW_INVALID_ARGUMENT("Invalid mode for hdfsOpenFile: %d.", mode);
  }
}

HDFSFileStream::HDFSFileStream(HDFSHandle* handle) : handle_(handle) {
  bad_ = 1;
}

HDFSFileStream::~HDFSFileStream() { Close(); }

std::size_t HDFSFileStream::Write(const void* _data, std::size_t _size) {
  static constexpr std::size_t MAX_HDFS_WRITE_BYTES =
      128 * 1024 * 1024;  // magic number
  const char* data = (const char*)_data;
  std::size_t size = _size;
  for (;;) {
    std::size_t _bytes =
        size <= MAX_HDFS_WRITE_BYTES ? size : MAX_HDFS_WRITE_BYTES;
    tSize bytes = phdfsWrite((hdfsFS)handle_->raw_handle(),  // NOLINT
                             (hdfsFile)f_, data, (tSize)_bytes);
    if (bytes > 0) {
      data += bytes;
      size -= bytes;
      if (size == 0) {
        return _size;
      }
    } else if (errno == EINTR || errno == EAGAIN) {
      continue;
    } else {
      DXERROR("Failed to hdfsWrite, size=%zu, errno=%d(%s).", _bytes, errno,
              strerror(errno));
      bad_ = 1;
      break;
    }
  }
  return _size - size;
}

bool HDFSFileStream::Flush() {
  if (mode_ & FILE_OPEN_MODE_OUT) {
    if (phdfsHFlush((hdfsFS)handle_->raw_handle(),  // NOLINT
                    (hdfsFile)f_) == -1) {
      DXERROR("Failed to hdfsHFlush, errno=%d(%s).", errno, strerror(errno));
      return false;
    }
  }
  return true;
}

std::size_t HDFSFileStream::Read(void* _data, std::size_t _size) {
  char* data = (char*)_data;
  std::size_t size = _size;
  for (;;) {
    tSize bytes = phdfsRead((hdfsFS)handle_->raw_handle(),  // NOLINT
                            (hdfsFile)f_, data, (tSize)size);
    if (bytes > 0) {
      data += bytes;
      size -= bytes;
      if (size == 0) {
        return _size;
      }
    } else if (bytes == 0) {
      bad_ = 1;
      break;
    } else if (errno == EINTR || errno == EAGAIN) {
      continue;
    } else {
      DXERROR("Failed to hdfsRead, size=%zu, errno=%d(%s).", size, errno,
              strerror(errno));
      bad_ = 1;
      break;
    }
  }
  return _size - size;
}

char HDFSFileStream::ReadChar() {
  char c;
  for (;;) {
    tSize bytes = phdfsRead((hdfsFS)handle_->raw_handle(),  // NOLINT
                            (hdfsFile)f_, &c, 1);
    if (bytes > 0) {
      return c;
    } else if (bytes == 0) {
      bad_ = 1;
      break;
    } else if (errno == EINTR || errno == EAGAIN) {
      continue;
    } else {
      DXERROR("Failed to hdfsRead, size=%zu, errno=%d(%s).", (std::size_t)1,
              errno, strerror(errno));
      bad_ = 1;
      break;
    }
  }
  return (char)-1;
}

std::size_t HDFSFileStream::Peek(void* data, std::size_t size) {
  tOffset offset = phdfsTell((hdfsFS)handle_->raw_handle(),  // NOLINT
                             (hdfsFile)f_);
  if (offset == -1) {
    DXTHROW_RUNTIME_ERROR("Failed to hdfsTell, errno=%d(%s).", errno,
                          strerror(errno));
  }
  std::size_t bytes = Read(data, size);
  if (phdfsSeek((hdfsFS)handle_->raw_handle(),  // NOLINT
                (hdfsFile)f_, offset) == -1) {
    DXTHROW_RUNTIME_ERROR("Failed to hdfsSeek, errno=%d(%s).", errno,
                          strerror(errno));
  }
  return bytes;
}

bool HDFSFileStream::Open(const std::string& file, int mode) {
  Close();

  if (IsStdinStdoutPath(file)) {
    DXTHROW_INVALID_ARGUMENT("Invalid file: %s.", file.c_str());
  }

  f_ = phdfsOpenFile((hdfsFS)handle_->raw_handle(),  // NOLINT
                     file.c_str(), GetHDFSOpenMode(mode), 0, 0, 0);
  if (f_ == nullptr) {
    DXERROR("Failed to hdfsOpenFile, file=%s, errno=%d(%s).", file.c_str(),
            errno, strerror(errno));
    bad_ = 1;
    mode_ = FILE_OPEN_MODE_NONE;
    return false;
  } else {
    bad_ = 0;
    mode_ = mode;
    return true;
  }
}

bool HDFSFileStream::IsOpen() const noexcept { return f_; }

void HDFSFileStream::Close() noexcept {
  if (f_) {
    (void)phdfsCloseFile((hdfsFS)handle_->raw_handle(),  // NOLINT
                         (hdfsFile)f_);
  }
  bad_ = 1;
  f_ = nullptr;
  mode_ = FILE_OPEN_MODE_NONE;
}

/************************************************************************/
/* AutoFileSystem */
/************************************************************************/
static bool ParseHDFSNameNode(const std::string& path,
                              std::string* name_node_host,
                              uint16_t* name_node_port) noexcept {
  std::size_t i = 7;
  std::size_t j = path.find('/', i);
  if (j == std::string::npos) {
    DXERROR("Invalid hdfs path: %s.", path.c_str());
    return false;
  }

  if (j == i) {
    // hdfs:///...
    *name_node_host = "default";
    *name_node_port = 0;
    return true;
  }

  std::size_t k = path.find(':', i);
  if (k == std::string::npos) {
    // hdfs://.../...
    *name_node_host = path.substr(i, j - i);
    *name_node_port = 0;
    return true;
  }

  if (k > j) {
    // hdfs://.../...:...
    DXERROR("Invalid hdfs path: %s.", path.c_str());
    return false;
  }

  // hdfs://host:port/...
  *name_node_host = path.substr(i, k - i);
  *name_node_port = (uint16_t)std::stoi(path.substr(k + 1, j - k));
  return true;
}

bool AutoFileSystem::Stat(const FilePath& path, FileStat* stat) {
  return fs_->Stat(path, stat);
}

bool AutoFileSystem::List(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  return fs_->List(path, skip_dir, children);
}

bool AutoFileSystem::ListRecursive(
    const FilePath& path, bool skip_dir,
    std::vector<std::pair<FilePath, FileStat>>* children) {
  return fs_->ListRecursive(path, skip_dir, children);
}

bool AutoFileSystem::MakeDir(const FilePath& dir) { return fs_->MakeDir(dir); }

bool AutoFileSystem::Move(const FilePath& old_path, const FilePath& new_path) {
  return fs_->Move(old_path, new_path);
}

bool AutoFileSystem::Open(const std::string& path) {
  Close();

  if (IsHDFSPath(path)) {
    if (!HasHDFS()) {
      return false;
    }

    std::string name_node_host;
    uint16_t name_node_port;
    if (!ParseHDFSNameNode(path, &name_node_host, &name_node_port)) {
      return false;
    }

    hdfs_handle_.reset(new HDFSHandle);
    if (!hdfs_handle_->Connect(name_node_host, name_node_port)) {
      return false;
    }

    fs_.reset(new HDFSFileSystem(hdfs_handle_.get()));
    return true;
  }

  fs_.reset(new LocalFileSystem);
  return true;
}

bool AutoFileSystem::IsOpen() const noexcept { return fs_.get(); }

void AutoFileSystem::Close() noexcept {
  fs_.reset();
  // put it last
  hdfs_handle_.reset();
}

bool AutoFileSystem::Exists(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.Exists(FilePath(path));
}

bool AutoFileSystem::IsDir(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.IsDir(FilePath(path));
}

bool AutoFileSystem::IsFile(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.IsFile(FilePath(path));
}

bool AutoFileSystem::IsRegFile(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.IsRegFile(FilePath(path));
}

bool AutoFileSystem::IsSymLink(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.IsSymLink(FilePath(path));
}

bool AutoFileSystem::IsOther(const std::string& path) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.IsOther(FilePath(path));
}

bool AutoFileSystem::GetFileSize(const std::string& path, std::size_t* size) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }
  return fs.GetFileSize(FilePath(path), size);
}

bool AutoFileSystem::List(const std::string& path, bool skip_dir,
                          std::vector<std::string>* _children) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }

  std::vector<std::pair<FilePath, FileStat>> children;
  if (!fs.List(FilePath(path), skip_dir, &children)) {
    return false;
  }

  for (const auto& child : children) {
    _children->emplace_back(child.first.str());
  }
  return true;
}

bool AutoFileSystem::ListRecursive(const std::string& path, bool skip_dir,
                                   std::vector<std::string>* _children) {
  AutoFileSystem fs;
  if (!fs.Open(path)) {
    return false;
  }

  std::vector<std::pair<FilePath, FileStat>> children;
  if (!fs.ListRecursive(FilePath(path), skip_dir, &children)) {
    return false;
  }

  for (const auto& child : children) {
    _children->emplace_back(child.first.str());
  }
  return true;
}

bool AutoFileSystem::MakeDir(const std::string& dir) {
  AutoFileSystem fs;
  if (!fs.Open(dir)) {
    return false;
  }
  return fs.MakeDir(FilePath(dir));
}

bool AutoFileSystem::Move(const std::string& old_path,
                          const std::string& new_path) {
  AutoFileSystem fs;
  if (!fs.Open(old_path)) {
    return false;
  }
  return fs.Move(FilePath(old_path), FilePath(new_path));
}

bool AutoFileSystem::BackupIfExists(const std::string& old_path,
                                    std::string* _new_path) {
  AutoFileSystem fs;
  if (!fs.Open(old_path)) {
    return false;
  }

  FilePath new_path;
  if (!fs.BackupIfExists(FilePath(old_path), &new_path)) {
    return false;
  }

  *_new_path = new_path.str();
  return true;
}

/************************************************************************/
/* AutoInputFileStream */
/************************************************************************/
AutoInputFileStream::AutoInputFileStream() { bad_ = 1; }

std::size_t AutoInputFileStream::Read(void* data, std::size_t size) {
  std::size_t bytes = is_->Read(data, size);
  bad_ = is_->bad();
  return bytes;
}

char AutoInputFileStream::ReadChar() {
  char c = is_->ReadChar();
  bad_ = is_->bad();
  return c;
}

std::size_t AutoInputFileStream::Peek(void* data, std::size_t size) {
  std::size_t bytes = is_->Peek(data, size);
  bad_ = is_->bad();
  return bytes;
}

bool AutoInputFileStream::Open(const std::string& file) {
  Close();

  if (IsHDFSPath(file)) {
    if (!HasHDFS()) {
      return false;
    }

    std::string name_node_host;
    uint16_t name_node_port;
    if (!ParseHDFSNameNode(file, &name_node_host, &name_node_port)) {
      return false;
    }

    std::unique_ptr<HDFSHandle> hdfs_handle(new HDFSHandle);
    if (!hdfs_handle->Connect(name_node_host, name_node_port)) {
      return false;
    }

    std::unique_ptr<HDFSFileStream> is_extra(
        new HDFSFileStream(hdfs_handle.get()));
    if (!is_extra->Open(file, FILE_OPEN_MODE_IN)) {
      return false;
    }

    hdfs_handle_ = std::move(hdfs_handle);
    is_extra_ = std::move(is_extra);
    if (IsGzipFile(file)) {
      is_.reset(new GunzipInputStream(is_extra_.get()));
    } else {
      is_.reset(new BufferedInputStream(is_extra_.get()));
    }
    bad_ = 0;
    return true;
  }

  if (IsStdinStdoutPath(file) || LocalFileExists(file)) {
    std::unique_ptr<CFileStream> is_extra(new CFileStream);
    if (!is_extra->Open(file, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY)) {
      return false;
    }

    is_extra_ = std::move(is_extra);
    if (IsGzipFile(file)) {
      is_.reset(new GunzipInputStream(is_extra_.get()));
    } else {
      is_.reset(new BufferedInputStream(is_extra_.get()));
    }
    bad_ = 0;
    return true;
  }

  return false;
}

bool AutoInputFileStream::IsOpen() const noexcept { return is_.get(); }

void AutoInputFileStream::Close() noexcept {
  bad_ = 1;
  is_extra_.reset();
  is_.reset();
  // put it last
  hdfs_handle_.reset();
}

/************************************************************************/
/* AutoOutputFileStream */
/************************************************************************/
AutoOutputFileStream::AutoOutputFileStream() { bad_ = 1; }

std::size_t AutoOutputFileStream::Write(const void* data, std::size_t size) {
  std::size_t bytes = os_->Write(data, size);
  bad_ = os_->bad();
  return bytes;
}

bool AutoOutputFileStream::Flush() { return os_->Flush(); }

bool AutoOutputFileStream::Open(const std::string& file) {
  Close();

  if (IsHDFSPath(file)) {
    if (!HasHDFS()) {
      return false;
    }

    std::string name_node_host;
    uint16_t name_node_port;
    if (!ParseHDFSNameNode(file, &name_node_host, &name_node_port)) {
      return false;
    }

    std::unique_ptr<HDFSHandle> hdfs_handle(new HDFSHandle);
    if (!hdfs_handle->Connect(name_node_host, name_node_port)) {
      return false;
    }

    std::unique_ptr<HDFSFileStream> os(new HDFSFileStream(hdfs_handle.get()));
    if (!os->Open(file, FILE_OPEN_MODE_OUT)) {
      return false;
    }

    hdfs_handle_ = std::move(hdfs_handle);
    os_ = std::move(os);
    bad_ = 0;
    return true;
  }

  std::unique_ptr<CFileStream> os(new CFileStream);
  if (!os->Open(file, FILE_OPEN_MODE_OUT | FILE_OPEN_MODE_BINARY)) {
    return false;
  }
  os_ = std::move(os);
  bad_ = 0;
  return true;
}

bool AutoOutputFileStream::IsOpen() const noexcept { return os_.get(); }

void AutoOutputFileStream::Close() noexcept {
  bad_ = 1;
  os_.reset();
  // put it last
  hdfs_handle_.reset();
}

/************************************************************************/
/* Serialization functions */
/************************************************************************/
OutputStream& operator<<(OutputStream& os, const std::string& s) {
  int size = (int)s.size();
  os.Write(&size, sizeof(size));
  if (size > 0) {
    os.Write(s.data(), s.size());
  }
  return os;
}

InputStream& operator>>(InputStream& is, std::string& s) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      s.resize(size);
      is.Read(&s[0], size);
    } else {
      s.clear();
    }
  }
  return is;
}

}  // namespace deepx_core
