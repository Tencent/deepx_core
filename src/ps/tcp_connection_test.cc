// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/tcp_connection.h>
#include <gtest/gtest.h>
#include <vector>

namespace deepx_core {

class TcpEndpointTest : public testing::Test {};

TEST_F(TcpEndpointTest, MakeTcpEndpoint_1) {
  TcpEndpoint endpoint;

  endpoint = MakeTcpEndpoint("127.0.0.1", 9527);
  EXPECT_TRUE(endpoint.address().is_v4());
  EXPECT_EQ(endpoint.port(), 9527u);

  endpoint = MakeTcpEndpoint("0.0.0.0", 9527);
  EXPECT_TRUE(endpoint.address().is_v4());

  endpoint = MakeTcpEndpoint("::1", 9527);
  EXPECT_TRUE(endpoint.address().is_v6());

  endpoint = MakeTcpEndpoint("::", 9527);
  EXPECT_TRUE(endpoint.address().is_v6());

  endpoint = MakeTcpEndpoint("2001:DB8:0:23:8:800:200C:417A", 9527);
  EXPECT_TRUE(endpoint.address().is_v6());

  EXPECT_ANY_THROW(MakeTcpEndpoint("", 9527));
  EXPECT_ANY_THROW(MakeTcpEndpoint("a", 9527));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1", 0));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1", -1));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1", 100000));
}

TEST_F(TcpEndpointTest, MakeTcpEndpoint_2) {
  TcpEndpoint endpoint;

  endpoint = MakeTcpEndpoint("127.0.0.1:9527");
  EXPECT_TRUE(endpoint.address().is_v4());
  EXPECT_EQ(endpoint.port(), 9527u);

  endpoint = MakeTcpEndpoint("0.0.0.0:9527");
  EXPECT_TRUE(endpoint.address().is_v4());

  endpoint = MakeTcpEndpoint("::1:9527");
  EXPECT_TRUE(endpoint.address().is_v6());

  endpoint = MakeTcpEndpoint(":::9527");
  EXPECT_TRUE(endpoint.address().is_v6());

  endpoint = MakeTcpEndpoint("2001:DB8:0:23:8:800:200C:417A:9527");
  EXPECT_TRUE(endpoint.address().is_v6());

  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1"));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1:"));
  EXPECT_ANY_THROW(MakeTcpEndpoint(":9527"));
  EXPECT_ANY_THROW(MakeTcpEndpoint("a:9527"));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1:0"));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1:-1"));
  EXPECT_ANY_THROW(MakeTcpEndpoint("127.0.0.1:100000"));
}

TEST_F(TcpEndpointTest, MakeTcpEndpoints) {
  std::vector<TcpEndpoint> endpoints =
      MakeTcpEndpoints("127.0.0.1:9527;127.0.0.1:9528;127.0.0.1:9529");
  EXPECT_EQ(endpoints.size(), 3u);
  EXPECT_TRUE(endpoints[0].address().is_v4());
  EXPECT_EQ(endpoints[0].port(), 9527u);
  EXPECT_TRUE(endpoints[1].address().is_v4());
  EXPECT_EQ(endpoints[1].port(), 9528u);
  EXPECT_TRUE(endpoints[2].address().is_v4());
  EXPECT_EQ(endpoints[2].port(), 9529u);
}

}  // namespace deepx_core
