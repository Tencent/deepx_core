// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/group_config.h>
#include <deepx_core/graph/graph_node.h>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* InstanceNode creator */
/************************************************************************/
constexpr int BATCH_PLACEHOLDER = 100;  // magic number
GraphNode* GetX();
GraphNode* GetX(int i);
GraphNode* GetXUser();
GraphNode* GetXCand();
GraphNode* GetXHist(int i);
GraphNode* GetXHistSize();
GraphNode* GetY(int label_size = 1);
GraphNode* GetW(int label_size = 1);
GraphNode* GetInstance(const std::string& name, const Shape& shape,
                       int tensor_type);

/************************************************************************/
/* embedding creator */
/************************************************************************/
// Share variables.
GraphNode* WideGroupEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                    const std::vector<GroupConfigItem3>& items,
                                    int sparse, int need_grad = 1);
// Share variables.
GraphNode* WideGroupEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                     const std::vector<GroupConfigItem3>& items,
                                     int sparse, int need_grad = 1);
// Share variables.
GraphNode* DeepGroupEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                    const std::vector<GroupConfigItem3>& items,
                                    int sparse, int need_grad = 1);
// Share variables.
GraphNode* DeepGroupEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                     const std::vector<GroupConfigItem3>& items,
                                     int sparse, int need_grad = 1);

/************************************************************************/
/* building block creator */
/************************************************************************/
// Share variables.
GraphNode* StackedFullyConnect(const std::string& prefix, GraphNode* X,
                               const std::vector<int>& deep_dims,
                               const std::string& activation = "relu");
// Share variables.
GraphNode* FullyConnect(const std::string& prefix, GraphNode* X, int out_dim);
// Share variables.
GraphNode* AddBias(const std::string& prefix, GraphNode* X);
// Share variables.
GraphNode* SelfAttention(const std::string& prefix, GraphNode* X, int n);
// Share variables.
GraphNode* CrossNet(const std::string& prefix, GraphNode* X, int cross);
// Share variables.
GraphNode* CIN(const std::string& prefix, GraphNode* X,
               const std::vector<int>& dims);
// Share variables.
GraphNode* RNNCell(const std::string& prefix, GraphNode* X, GraphNode* Hin,
                   int n);
// Share variables.
std::vector<GraphNode*> LSTMCell(const std::string& prefix, GraphNode* X,
                                 const std::vector<GraphNode*>& CHin, int n,
                                 int mask = 3);
// Share variables.
GraphNode* GRUCell(const std::string& prefix, GraphNode* X, GraphNode* Hin,
                   int n);
// Split 'X' along 'axis'.
std::vector<GraphNode*> Split(const std::string& prefix, GraphNode* X, int axis,
                              int n);
std::vector<GraphNode*> Split(GraphNode* X, int axis, int n);
std::vector<GraphNode*> Split(const std::string& prefix, GraphNode* X, int axis,
                              const std::vector<int>& split_dims);
std::vector<GraphNode*> Split(GraphNode* X, int axis,
                              const std::vector<int>& split_dims);
// Share variables.
GraphNode* BatchNorm(const std::string& prefix, GraphNode* X,
                     double moving_decay = 0.9);

/************************************************************************/
/* target creator */
/************************************************************************/
std::vector<GraphNode*> BinaryClassificationTarget(const std::string& prefix,
                                                   GraphNode* X, int has_w);
std::vector<GraphNode*> BinaryClassificationTarget(GraphNode* X, int has_w);

}  // namespace deepx_core
