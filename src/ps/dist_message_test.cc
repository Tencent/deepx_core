// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/ps/dist_message.h>
#include <gtest/gtest.h>

namespace deepx_core {

class DistMessageTest : public testing::Test {};

TEST_F(DistMessageTest, WriteReadView) {
  DistMessage message;
  DistMessageView read_message;
  message.set_type(DIST_MESSAGE_TYPE_PULL_REQUEST);
  message.mutable_pull_request()->buf = "test";

  OutputStringStream os;
  InputStringStream is;

  os << message;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_message);
  ASSERT_TRUE(is);

  EXPECT_EQ(message.type(), read_message.type());
  EXPECT_EQ(message.pull_request().buf, read_message.pull_request().buf);
}

}  // namespace deepx_core
