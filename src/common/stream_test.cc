// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <gtest/gtest.h>
#include <cstring>  // strncmp
#include <numeric>  // std::accumulate
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Stream functions */
/************************************************************************/
class StreamTest : public testing::Test {};

TEST_F(StreamTest, IsDirSeparator) {
  EXPECT_TRUE(IsDirSeparator('\\'));
  EXPECT_TRUE(IsDirSeparator('/'));
  EXPECT_FALSE(IsDirSeparator('1'));
  EXPECT_FALSE(IsDirSeparator('a'));
}

TEST_F(StreamTest, basename) {
  EXPECT_EQ(basename("/usr/bin/gcc"), "gcc");
  EXPECT_EQ(basename("/usr/bin/"), "bin");
  EXPECT_EQ(basename("/usr/bin"), "bin");
  EXPECT_EQ(basename("bin/gcc"), "gcc");
  EXPECT_EQ(basename("gcc"), "gcc");
  EXPECT_EQ(basename("/"), "/");
  EXPECT_EQ(basename("."), ".");
  EXPECT_EQ(basename(".."), "..");
  EXPECT_EQ(basename(StdinStdoutPath()), StdinStdoutPath());
}

TEST_F(StreamTest, dirname) {
  EXPECT_EQ(dirname("/usr/bin/gcc"), "/usr/bin");
  EXPECT_EQ(dirname("/usr/bin/"), "/usr");
  EXPECT_EQ(dirname("/usr/bin"), "/usr");
  EXPECT_EQ(dirname("bin/gcc"), "bin");
  EXPECT_EQ(dirname("gcc"), ".");
  EXPECT_EQ(dirname("/"), "/");
  EXPECT_EQ(dirname("."), ".");
  EXPECT_EQ(dirname(".."), ".");
  EXPECT_EQ(dirname(StdinStdoutPath()), ".");
}

TEST_F(StreamTest, CanonicalizePath) {
  EXPECT_EQ(CanonicalizePath("/usr/bin/gcc"), "/usr/bin/gcc");
  EXPECT_EQ(CanonicalizePath("/usr/bin/"), "/usr/bin");
  EXPECT_EQ(CanonicalizePath("/usr/bin"), "/usr/bin");
  EXPECT_EQ(CanonicalizePath("bin/gcc"), "bin/gcc");
  EXPECT_EQ(CanonicalizePath("gcc"), "gcc");
  EXPECT_EQ(CanonicalizePath("/"), "/");
  EXPECT_EQ(CanonicalizePath("."), ".");
  EXPECT_EQ(CanonicalizePath(".."), "..");
  EXPECT_EQ(CanonicalizePath(""), "");
  EXPECT_EQ(CanonicalizePath(StdinStdoutPath()), StdinStdoutPath());
}

TEST_F(StreamTest, IsHDFSPath) {
  EXPECT_TRUE(IsHDFSPath("hdfs://usr/bin/gcc"));
  EXPECT_TRUE(IsHDFSPath("hdfs://gcc"));
  EXPECT_FALSE(IsHDFSPath("/usr/bin/gcc"));
  EXPECT_FALSE(IsHDFSPath("gcc"));
  EXPECT_FALSE(IsHDFSPath(StdinStdoutPath()));
}

TEST_F(StreamTest, IsGzipFile) {
  EXPECT_TRUE(IsGzipFile("hdfs://usr/bin/gcc.gz"));
  EXPECT_TRUE(IsGzipFile("hdfs://gcc.gz"));
  EXPECT_TRUE(IsGzipFile("/usr/bin/gcc.gz"));
  EXPECT_TRUE(IsGzipFile("gcc.gz"));
  EXPECT_FALSE(IsGzipFile("hdfs://usr/bin/gcc"));
  EXPECT_FALSE(IsGzipFile("hdfs://gcc"));
  EXPECT_FALSE(IsGzipFile("/usr/bin/gcc"));
  EXPECT_FALSE(IsGzipFile("gcc"));
  EXPECT_FALSE(IsGzipFile(StdinStdoutPath()));
}

TEST_F(StreamTest, IsStdinStdoutPath) {
  EXPECT_TRUE(IsStdinStdoutPath(StdinStdoutPath()));
  EXPECT_FALSE(IsStdinStdoutPath("gcc"));
}

/************************************************************************/
/* LocalFileSystem */
/************************************************************************/
class LocalFileSystemTest : public testing::Test {};

TEST_F(LocalFileSystemTest, Stat) {
  LocalFileSystem fs;
  FileStat stat;

  stat = FileStat::StdinStdout();
  EXPECT_TRUE(stat.Exists());
  EXPECT_FALSE(stat.IsDir());
  EXPECT_FALSE(stat.IsFile());
  EXPECT_FALSE(stat.IsRegFile());
  EXPECT_FALSE(stat.IsSymLink());
  EXPECT_TRUE(stat.IsOther());

  ASSERT_FALSE(fs.Stat("", &stat));
  EXPECT_FALSE(stat.Exists());

  ASSERT_FALSE(fs.Stat("not_exist", &stat));
  EXPECT_FALSE(stat.Exists());

  ASSERT_TRUE(fs.Stat("testdata/common/stream/dir", &stat));
  EXPECT_TRUE(stat.Exists());
  EXPECT_TRUE(stat.IsDir());
  EXPECT_FALSE(stat.IsFile());
  EXPECT_FALSE(stat.IsRegFile());
  EXPECT_FALSE(stat.IsSymLink());
  EXPECT_FALSE(stat.IsOther());

  ASSERT_TRUE(fs.Stat("testdata/common/stream/100.txt", &stat));
  EXPECT_TRUE(stat.Exists());
  EXPECT_FALSE(stat.IsDir());
  EXPECT_TRUE(stat.IsFile());
  EXPECT_TRUE(stat.IsRegFile());
  EXPECT_FALSE(stat.IsSymLink());
  EXPECT_FALSE(stat.IsOther());
  EXPECT_EQ(stat.GetFileSize(), 1132u);

  ASSERT_TRUE(fs.Stat("testdata/common/stream/empty", &stat));
  EXPECT_TRUE(stat.Exists());
  EXPECT_FALSE(stat.IsDir());
  EXPECT_TRUE(stat.IsFile());
  EXPECT_TRUE(stat.IsRegFile());
  EXPECT_FALSE(stat.IsSymLink());
  EXPECT_FALSE(stat.IsOther());
  EXPECT_EQ(stat.GetFileSize(), 0u);

#if OS_POSIX == 1
  ASSERT_TRUE(fs.Stat("testdata/common/stream/100.txt.link", &stat));
  EXPECT_TRUE(stat.Exists());
  EXPECT_FALSE(stat.IsDir());
  EXPECT_TRUE(stat.IsFile());
  EXPECT_FALSE(stat.IsRegFile());
  EXPECT_TRUE(stat.IsSymLink());
  EXPECT_FALSE(stat.IsOther());
  EXPECT_EQ(stat.GetFileSize(), 7u);

  ASSERT_TRUE(fs.Stat("testdata/common/stream/empty.link", &stat));
  EXPECT_TRUE(stat.Exists());
  EXPECT_FALSE(stat.IsDir());
  EXPECT_TRUE(stat.IsFile());
  EXPECT_FALSE(stat.IsRegFile());
  EXPECT_TRUE(stat.IsSymLink());
  EXPECT_FALSE(stat.IsOther());
  EXPECT_EQ(stat.GetFileSize(), 5u);
#endif
}

TEST_F(LocalFileSystemTest, List) {
  LocalFileSystem fs;
  std::vector<std::pair<FilePath, FileStat>> children;

  ASSERT_TRUE(fs.List(StdinStdoutPath(), true, &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, StdinStdoutPath());
  EXPECT_TRUE(children[0].second.IsOther());

  ASSERT_TRUE(fs.List("testdata/common/stream/dir", true, &children));
  EXPECT_EQ(children.size(), 0u);

  ASSERT_TRUE(fs.List("testdata/common/stream/dir", false, &children));
  EXPECT_EQ(children.size(), 3u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1");
  EXPECT_TRUE(children[0].second.IsDir());
  EXPECT_EQ(children[1].first, "testdata/common/stream/dir/dir2");
  EXPECT_TRUE(children[1].second.IsDir());
  EXPECT_EQ(children[2].first, "testdata/common/stream/dir/dir3");
  EXPECT_TRUE(children[2].second.IsDir());

  ASSERT_TRUE(
      fs.List("testdata/common/stream/dir/dir1/file1", true, &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[0].second.IsRegFile());

  ASSERT_TRUE(
      fs.List("testdata/common/stream/dir/dir1/file1", false, &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[0].second.IsRegFile());
}

TEST_F(LocalFileSystemTest, ListRecursive) {
  LocalFileSystem fs;
  std::vector<std::pair<FilePath, FileStat>> children;

  ASSERT_TRUE(fs.List(StdinStdoutPath(), true, &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, StdinStdoutPath());
  EXPECT_TRUE(children[0].second.IsOther());

  ASSERT_TRUE(fs.ListRecursive("testdata/common/stream/dir", true, &children));
  EXPECT_EQ(children.size(), 6u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[0].second.IsRegFile());
  EXPECT_EQ(children[1].first, "testdata/common/stream/dir/dir2/file1");
  EXPECT_TRUE(children[1].second.IsRegFile());
  EXPECT_EQ(children[2].first, "testdata/common/stream/dir/dir2/file2");
  EXPECT_TRUE(children[2].second.IsRegFile());
  EXPECT_EQ(children[3].first, "testdata/common/stream/dir/dir3/file1");
  EXPECT_TRUE(children[3].second.IsRegFile());
  EXPECT_EQ(children[4].first, "testdata/common/stream/dir/dir3/file2");
  EXPECT_TRUE(children[4].second.IsRegFile());
  EXPECT_EQ(children[5].first, "testdata/common/stream/dir/dir3/file3");
  EXPECT_TRUE(children[5].second.IsRegFile());

  ASSERT_TRUE(fs.ListRecursive("testdata/common/stream/dir", false, &children));
  EXPECT_EQ(children.size(), 9u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1");
  EXPECT_TRUE(children[0].second.IsDir());
  EXPECT_EQ(children[1].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[1].second.IsRegFile());
  EXPECT_EQ(children[2].first, "testdata/common/stream/dir/dir2");
  EXPECT_TRUE(children[2].second.IsDir());
  EXPECT_EQ(children[3].first, "testdata/common/stream/dir/dir2/file1");
  EXPECT_TRUE(children[3].second.IsRegFile());
  EXPECT_EQ(children[4].first, "testdata/common/stream/dir/dir2/file2");
  EXPECT_TRUE(children[4].second.IsRegFile());
  EXPECT_EQ(children[5].first, "testdata/common/stream/dir/dir3");
  EXPECT_TRUE(children[5].second.IsDir());
  EXPECT_EQ(children[6].first, "testdata/common/stream/dir/dir3/file1");
  EXPECT_TRUE(children[6].second.IsRegFile());
  EXPECT_EQ(children[7].first, "testdata/common/stream/dir/dir3/file2");
  EXPECT_TRUE(children[7].second.IsRegFile());
  EXPECT_EQ(children[8].first, "testdata/common/stream/dir/dir3/file3");
  EXPECT_TRUE(children[8].second.IsRegFile());

  ASSERT_TRUE(fs.ListRecursive("testdata/common/stream/dir/dir1/file1", true,
                               &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[0].second.IsRegFile());

  ASSERT_TRUE(fs.ListRecursive("testdata/common/stream/dir/dir1/file1", false,
                               &children));
  EXPECT_EQ(children.size(), 1u);
  EXPECT_EQ(children[0].first, "testdata/common/stream/dir/dir1/file1");
  EXPECT_TRUE(children[0].second.IsRegFile());
}

/************************************************************************/
/* BufferedInputStream */
/************************************************************************/
class BufferedInputStreamTest : public testing::Test {
 protected:
  std::string file_;

 protected:
  void SetUp() override { file_ = "testdata/common/stream/100.txt"; }

 protected:
  static void TestGetLine(InputStream& is) {  // NOLINT
    std::string line;
    std::vector<std::string> lines;
    size_t sum = 0;
    while (GetLine(is, line)) {
      lines.emplace_back(line);
      sum += std::accumulate(line.begin(), line.end(), 0);
    }
    EXPECT_EQ(lines.size(), 100u);
    EXPECT_EQ(sum, 49020u);
  }

  static void TestRead(InputStream& is) {  // NOLINT
    char buf[256];
    size_t bytes = 0;
    size_t sum = 0;
    for (;;) {
      bytes = is.Read(buf, sizeof(buf));
      sum += std::accumulate(buf, buf + bytes, 0);
      if (bytes == 0) {
        break;
      }
    }
    EXPECT_FALSE(is);
    EXPECT_EQ(sum, 50020u);
  }

  static void TestPeek(InputStream& is) {  // NOLINT
    char peek_buf[256], read_buf[256];
    size_t peek_bytes = 0, read_bytes = 0;
    size_t peek_sum = 0, read_sum = 0;
    for (;;) {
      peek_bytes = is.Peek(peek_buf, sizeof(peek_buf));
      read_bytes = is.Read(read_buf, sizeof(read_buf));
      EXPECT_EQ(peek_bytes, read_bytes);
      peek_sum += std::accumulate(peek_buf, peek_buf + peek_bytes, 0);
      read_sum += std::accumulate(read_buf, read_buf + read_bytes, 0);
      EXPECT_EQ(peek_sum, read_sum);
      if (peek_bytes == 0) {
        break;
      }
    }
    EXPECT_FALSE(is);
    EXPECT_EQ(peek_sum, 50020u);
  }
};

TEST_F(BufferedInputStreamTest, GetLine_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64);
  TestGetLine(is);
}

TEST_F(BufferedInputStreamTest, GetLine_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64 * 1024);
  TestGetLine(is);
}

TEST_F(BufferedInputStreamTest, Read_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64);
  TestRead(is);
}

TEST_F(BufferedInputStreamTest, Read_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64 * 1024);
  TestRead(is);
}

TEST_F(BufferedInputStreamTest, Peek_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64);
  TestPeek(is);
}

TEST_F(BufferedInputStreamTest, Peek_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN));
  BufferedInputStream is(&fs, 64 * 1024);
  TestPeek(is);
}

/************************************************************************/
/* GunzipInputStream */
/************************************************************************/
class GunzipInputStreamTest : public BufferedInputStreamTest {
 protected:
  void SetUp() override { file_ = "testdata/common/stream/100.txt.gz"; }
};

TEST_F(GunzipInputStreamTest, GetLine_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64);
  TestGetLine(is);
}

TEST_F(GunzipInputStreamTest, GetLine_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64 * 1024);
  TestGetLine(is);
}

TEST_F(GunzipInputStreamTest, Read_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64);
  TestRead(is);
}

TEST_F(GunzipInputStreamTest, Read_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64 * 1024);
  TestRead(is);
}

TEST_F(GunzipInputStreamTest, Peek_buf_size64) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64);
  TestPeek(is);
}

TEST_F(GunzipInputStreamTest, Peek_buf_size64k) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  GunzipInputStream is(&fs, 64 * 1024);
  TestPeek(is);
}

/************************************************************************/
/* CFileStream */
/************************************************************************/
class CFileStreamTest : public BufferedInputStreamTest {};

TEST_F(CFileStreamTest, GetLine) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  TestGetLine(fs);
}

TEST_F(CFileStreamTest, Read) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  TestRead(fs);
}

TEST_F(CFileStreamTest, Peek) {
  CFileStream fs;
  ASSERT_TRUE(fs.Open(file_, FILE_OPEN_MODE_IN | FILE_OPEN_MODE_BINARY));
  TestPeek(fs);
}

/************************************************************************/
/* OutputStringStream */
/************************************************************************/
class OutputStringStreamTest : public testing::Test {};

TEST_F(OutputStringStreamTest, Write) {
  OutputStringStream os;
  os.Write("1234", 4);
  EXPECT_EQ(os.GetString(), "1234");
  EXPECT_EQ(strncmp(os.GetData(), "1234", 4), 0);
  EXPECT_EQ(os.GetSize(), 4u);
  os.Write("5678", 4);
  EXPECT_EQ(os.GetString(), "12345678");
  EXPECT_EQ(strncmp(os.GetData(), "12345678", 8), 0);
  EXPECT_EQ(os.GetSize(), 8u);
}

TEST_F(OutputStringStreamTest, SetString_Write) {
  OutputStringStream os;
  os.SetString("1234");
  os.Write("5678", 4);
  EXPECT_EQ(os.GetString(), "12345678");
  EXPECT_EQ(strncmp(os.GetData(), "12345678", 8), 0);
  EXPECT_EQ(os.GetSize(), 8u);
}

TEST_F(OutputStringStreamTest, SetView_Write) {
  OutputStringStream os;
  std::string buf = "1234";
  os.SetView(&buf);
  os.Write("5678", 4);
  EXPECT_EQ(buf, "12345678");
  EXPECT_EQ(os.GetString(), "12345678");
  EXPECT_EQ(strncmp(os.GetData(), "12345678", 8), 0);
  EXPECT_EQ(os.GetSize(), 8u);
}

/************************************************************************/
/* InputStringStream */
/************************************************************************/
class InputStringStreamTest : public testing::Test {};

TEST_F(InputStringStreamTest, SetString_GetLine) {
  InputStringStream is;
  std::string line;
  std::vector<std::string> lines;
  std::vector<std::string> expected_lines{"1a2a", "3a4a", "", "1a2a", "3a4a"};
  is.SetString("1a2a\n3a4a\n\n1a2a\n3a4a\n");
  while (GetLine(is, line)) {
    lines.emplace_back(line);
  }
  EXPECT_EQ(lines, expected_lines);
}

TEST_F(InputStringStreamTest, SetString_GetLine_delim) {
  InputStringStream is;
  std::string line;
  std::vector<std::string> lines;
  std::vector<std::string> expected_lines{"1", "2", "3", "4"};
  is.SetString("1a2a3a4a");
  while (GetLine(is, line, 'a')) {
    lines.emplace_back(line);
  }
  EXPECT_EQ(lines, expected_lines);
}

TEST_F(InputStringStreamTest, SetString_Read) {
  InputStringStream is;
  char c;
  std::string s;
  is.SetString("1234");
  for (;;) {
    is.Read(&c, sizeof(c));
    if (!is) {
      break;
    }
    s.push_back(c);
  }
  EXPECT_EQ(s, "1234");
}

TEST_F(InputStringStreamTest, SetString_Peek_Skip) {
  InputStringStream is;
  char c;
  std::string s;
  is.SetString("1234");
  for (;;) {
    is.Peek(&c, sizeof(c));
    is.Skip(sizeof(c));
    if (!is) {
      break;
    }
    s.push_back(c);
  }
  EXPECT_EQ(s, "1234");
}

TEST_F(InputStringStreamTest, SetView_GetLine) {
  InputStringStream is;
  std::string buf = "1a2a\n3a4a\n\n1a2a\n3a4a\n";
  std::string line;
  std::vector<std::string> lines;
  std::vector<std::string> expected_lines{"1a2a", "3a4a", "", "1a2a", "3a4a"};
  is.SetView(buf);
  while (GetLine(is, line)) {
    lines.emplace_back(line);
  }
  EXPECT_EQ(lines, expected_lines);
}

TEST_F(InputStringStreamTest, SetView_GetLine_delim) {
  InputStringStream is;
  std::string buf = "1a2a3a4a";
  std::string line;
  std::vector<std::string> lines;
  std::vector<std::string> expected_lines{"1", "2", "3", "4"};
  is.SetView(buf);
  while (GetLine(is, line, 'a')) {
    lines.emplace_back(line);
  }
  EXPECT_EQ(lines, expected_lines);
}

TEST_F(InputStringStreamTest, SetView_Read) {
  InputStringStream is;
  std::string buf = "1234";
  char c;
  std::string s;
  is.SetView(buf);
  for (;;) {
    is.Read(&c, sizeof(c));
    if (!is) {
      break;
    }
    s.push_back(c);
  }
  EXPECT_EQ(s, "1234");
}

TEST_F(InputStringStreamTest, SetView_Peek_Skip) {
  InputStringStream is;
  std::string buf = "1234";
  char c;
  std::string s;
  is.SetView(buf);
  for (;;) {
    is.Peek(&c, sizeof(c));
    is.Skip(sizeof(c));
    if (!is) {
      break;
    }
    s.push_back(c);
  }
  EXPECT_EQ(s, "1234");
}

/************************************************************************/
/* Serialization functions */
/************************************************************************/
class SerializationTest : public testing::Test {
 protected:
  using vi_t = std::vector<int>;
  using vs_t = std::vector<std::string>;
  using pii_t = std::pair<int, int>;
  using mid_t = std::unordered_map<int, double>;
  using mivi_t = std::unordered_map<int, std::vector<int>>;
  using si_t = std::unordered_set<int>;

 protected:
  int i = 123, read_i;
  double d = 123, read_d;
  std::string s = "123", read_s;
  vi_t vi = {1, 2, 3}, read_vi;
  vs_t vs = {"1", "2", "3"}, read_vs;
  pii_t pii = {1, 2}, read_pii;
  mid_t mid = {{1, 1}, {2, 2}, {3, 3}}, read_mid;
  mivi_t mivi = {{1, {1, 2, 3}}, {2, {2, 3, 4}}, {3, {3, 4, 5}}}, read_mivi;
  si_t si = {1, 2, 3}, read_si;
};

TEST_F(SerializationTest, WriteObject_ReadObject) {
  OutputStringStream os;
  InputStringStream is;

  os.WriteObject(i, d, s, vi, vs, pii, mid, mivi, si);
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is.ReadObject(read_i, read_d, read_s, read_vi, read_vs, read_pii, read_mid,
                read_mivi, read_si);
  ASSERT_TRUE(is);

  EXPECT_EQ(i, read_i);
  EXPECT_EQ(d, read_d);
  EXPECT_EQ(s, read_s);
  EXPECT_EQ(vi, read_vi);
  EXPECT_EQ(vs, read_vs);
  EXPECT_EQ(pii, read_pii);
  EXPECT_EQ(mid, read_mid);
  EXPECT_EQ(mivi, read_mivi);
  EXPECT_EQ(si, read_si);
}

TEST_F(SerializationTest, SerializeToString_ParseFromString) {
#define TEST_T(t)                                 \
  do {                                            \
    std::string buf;                              \
    ASSERT_TRUE(SerializeToString(t, &buf));      \
    ASSERT_TRUE(ParseFromString(buf, &read_##t)); \
    EXPECT_EQ(t, read_##t);                       \
  } while (0)
  TEST_T(i);
  TEST_T(d);
  TEST_T(s);
  TEST_T(vi);
  TEST_T(vs);
  TEST_T(pii);
  TEST_T(mid);
  TEST_T(mivi);
  TEST_T(si);
#undef TEST_T
}

TEST_F(SerializationTest, SerializeToString_ParseFromArray) {
#define TEST_T(t)                                                   \
  do {                                                              \
    std::string buf;                                                \
    ASSERT_TRUE(SerializeToString(t, &buf));                        \
    ASSERT_TRUE(ParseFromArray(buf.data(), buf.size(), &read_##t)); \
    EXPECT_EQ(t, read_##t);                                         \
  } while (0)
  TEST_T(i);
  TEST_T(d);
  TEST_T(s);
  TEST_T(vi);
  TEST_T(vs);
  TEST_T(pii);
  TEST_T(mid);
  TEST_T(mivi);
  TEST_T(si);
#undef TEST_T
}

TEST_F(SerializationTest, SerializeToString_ParseViewFromString) {
#define TEST_T(t)                                     \
  do {                                                \
    std::string buf;                                  \
    ASSERT_TRUE(SerializeToString(t, &buf));          \
    ASSERT_TRUE(ParseViewFromString(buf, &read_##t)); \
    EXPECT_EQ(t, read_##t);                           \
  } while (0)
  TEST_T(i);
  TEST_T(d);
  TEST_T(s);
  TEST_T(vi);
  TEST_T(vs);
  TEST_T(pii);
  TEST_T(mid);
  TEST_T(mivi);
  TEST_T(si);
#undef TEST_T
}

TEST_F(SerializationTest, SerializeToString_ParseViewFromArray) {
#define TEST_T(t)                                                       \
  do {                                                                  \
    std::string buf;                                                    \
    ASSERT_TRUE(SerializeToString(t, &buf));                            \
    ASSERT_TRUE(ParseViewFromArray(buf.data(), buf.size(), &read_##t)); \
    EXPECT_EQ(t, read_##t);                                             \
  } while (0)
  TEST_T(i);
  TEST_T(d);
  TEST_T(s);
  TEST_T(vi);
  TEST_T(vs);
  TEST_T(pii);
  TEST_T(mid);
  TEST_T(mivi);
  TEST_T(si);
#undef TEST_T
}

}  // namespace deepx_core
