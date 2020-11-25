// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* SimpItem */
/************************************************************************/
class SimpItem {
 public:
  using node_map_t = std::unordered_map<std::string, GraphNode*>;
  using heap_node_t = std::unique_ptr<GraphNode>;
  using heap_node_map_t = std::unordered_map<std::string, heap_node_t>;
  using node_set_t = std::unordered_set<GraphNode*>;
  using node_set_map_t = std::unordered_map<std::string, node_set_t>;

 private:
  std::vector<GraphNode*> target_;
  node_map_t name_2_target_;
  heap_node_map_t name_2_node_;
  node_set_map_t name_2_output_;

 public:
  int target_size() const noexcept { return (int)name_2_target_.size(); }
  bool is_target(const std::string& name) const noexcept;

  int node_size() const noexcept { return (int)name_2_node_.size(); }
  GraphNode* find_node(const std::string& name) noexcept;
  const GraphNode* find_node(const std::string& name) const noexcept;

  const node_set_t find_output(const std::string& name) const noexcept;

 public:
  SimpItem() = default;
  SimpItem(const SimpItem&) = delete;
  SimpItem& operator=(const SimpItem&) = delete;

 public:
  void clear() noexcept;
  void Add(GraphNode* node);
  void Prune();
  // Node-"replacement" can not be reachable from node-"name" along the
  // direction from target node to input node before calling this function.
  bool ReplaceInput(const std::string& name, const std::string& replaced,
                    const std::string& replacement);
  // Node-"replacement" can not be reachable from any output node of
  // node-"replaced" along the direction from target node to input
  // node before calling this function.
  bool ReplaceInputOfAllOutputs(const std::string& replaced,
                                const std::string& replacement);
  void GetTopologicalSortedNodes(std::vector<GraphNode*>* sorted,
                                 bool reverse = false) const;
  std::string NewNodeName(const std::string& old_name,
                          const std::vector<std::string>& append_scopes = {},
                          const std::string& prefix = "",
                          const std::string& suffix = "") const noexcept;
  bool FromGraph(const Graph& graph);
  bool ToGraph(Graph* graph);

 private:
  void BuildName2Output();
  // Along the direction from target node to input node.
  bool IsReachable(GraphNode* from, GraphNode* to) const;
  static bool Clone(GraphNode& from, GraphNode** to);  // NOLINT
  static bool Clone(const heap_node_map_t& from, node_map_t* to);
  static bool Clone(const node_map_t& from, heap_node_map_t* to);
};

}  // namespace deepx_core
