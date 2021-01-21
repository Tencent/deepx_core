// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph_node.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace deepx_core {

class Graph;

/************************************************************************/
/* Graph functions */
/************************************************************************/
bool SaveGraph(const std::string& dir, const Graph& graph);
bool LoadGraph(const std::string& dir, Graph* graph);
bool ParseFromString(const std::string& s, Graph* graph);

/************************************************************************/
/* GraphTarget */
/************************************************************************/
class GraphTarget {
 private:
  const GraphNode* node_ = nullptr;
  std::string name_;
  std::vector<GraphNode*> forward_;
  std::vector<std::string> forward_name_;
  friend class Graph;

 public:
  const GraphNode* node() const noexcept { return node_; }

  uint16_t node_id() const noexcept { return node_->node_id(); }

  const std::string& name() const noexcept { return name_; }

  int forward_size() const noexcept { return (int)forward_.size(); }
  template <typename Int>
  const GraphNode* forward(Int i) const noexcept {
    return forward_[(size_t)i];
  }
  const std::vector<GraphNode*>& forward() const noexcept { return forward_; }

  int forward_name_size() const noexcept { return (int)forward_name_.size(); }
  template <typename Int>
  const std::string& forward_name(Int i) const noexcept {
    return forward_name_[(size_t)i];
  }
  const std::vector<std::string>& forward_name() const noexcept {
    return forward_name_;
  }
};

/************************************************************************/
/* Graph */
/************************************************************************/
class Graph {
 private:
  int compiled_ = 0;
  std::vector<GraphTarget> target_;
  std::unordered_map<std::string, GraphTarget*> name_2_target_;
  std::unordered_map<std::string, GraphNode*> name_2_node_;
  std::vector<std::unique_ptr<GraphNode>> heap_node_;
  std::unordered_map<std::string, std::string> meta_;
  std::unordered_map<uint16_t, GraphTarget*> node_id_2_target_;
  std::unordered_map<uint16_t, GraphNode*> node_id_2_node_;

 public:
  int compiled() const noexcept { return compiled_; }

  int target_size() const noexcept { return (int)target_.size(); }
  template <typename Int>
  const GraphTarget& target(Int i) const noexcept {
    return target_[(size_t)i];
  }
  const std::vector<GraphTarget>& target() const noexcept { return target_; }

  const std::unordered_map<std::string, GraphTarget*>& name_2_target() const
      noexcept {
    return name_2_target_;
  }
  const GraphTarget* find_target(const std::string& name) const noexcept;

  const std::unordered_map<std::string, GraphNode*>& name_2_node() const
      noexcept {
    return name_2_node_;
  }
  const GraphNode* find_node(const std::string& name) const noexcept;

  std::unordered_map<std::string, std::string>& meta() noexcept {
    return meta_;
  }
  const std::unordered_map<std::string, std::string>& meta() const noexcept {
    return meta_;
  }

  std::unordered_map<uint16_t, GraphTarget*> node_id_2_target() const noexcept {
    return node_id_2_target_;
  }
  const GraphTarget* find_target(uint16_t node_id) const noexcept;

  std::unordered_map<uint16_t, GraphNode*> node_id_2_node() const noexcept {
    return node_id_2_node_;
  }
  const GraphNode* find_node(uint16_t node_id) const noexcept;

 private:
  static void CheckName(GraphNode* node,
                        std::unordered_set<std::string>* dedup_name);
  static void CompileTarget(GraphNode* node,
                            std::unordered_set<GraphNode*>* dedup_node,
                            std::unordered_set<std::string>* dedup_name,
                            GraphTarget* target);
  static void CompileTarget(GraphNode* node,
                            std::unordered_set<std::string>* dedup_name,
                            GraphTarget* target);
  void PostInit();

 public:
  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

 public:
  void clear() noexcept;
  bool Compile(const std::vector<GraphNode*>& target_nodes, int on_heap = 1);
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool WriteDot(std::string* s) const;
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);
  bool SaveDot(const std::string& file) const;
  std::string Dot() const;
  bool ParseFromString(const std::string& s);
};

}  // namespace deepx_core
