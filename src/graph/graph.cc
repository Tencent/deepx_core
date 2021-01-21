// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <algorithm>  // std::sort
#include <limits>     // std::numeric_limits
#include <set>
#include <sstream>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* Graph functions */
/************************************************************************/
namespace {

std::string GetGraphFile(const std::string& dir) { return dir + "/graph.bin"; }

}  // namespace

bool SaveGraph(const std::string& dir, const Graph& graph) {
  return graph.Save(GetGraphFile(dir));
}

bool LoadGraph(const std::string& dir, Graph* graph) {
  return graph->Load(GetGraphFile(dir));
}

bool ParseFromString(const std::string& s, Graph* graph) {
  return graph->ParseFromString(s);
}

/************************************************************************/
/* Graph */
/************************************************************************/
const GraphTarget* Graph::find_target(const std::string& name) const noexcept {
  auto it = name_2_target_.find(name);
  if (it == name_2_target_.end()) {
    return nullptr;
  }
  return it->second;
}

const GraphNode* Graph::find_node(const std::string& name) const noexcept {
  auto it = name_2_node_.find(name);
  if (it == name_2_node_.end()) {
    return nullptr;
  }
  return it->second;
}

const GraphTarget* Graph::find_target(uint16_t node_id) const noexcept {
  auto it = node_id_2_target_.find(node_id);
  if (it == node_id_2_target_.end()) {
    return nullptr;
  }
  return it->second;
}

const GraphNode* Graph::find_node(uint16_t node_id) const noexcept {
  auto it = node_id_2_node_.find(node_id);
  if (it == node_id_2_node_.end()) {
    return nullptr;
  }
  return it->second;
}

void Graph::CheckName(GraphNode* node,
                      std::unordered_set<std::string>* dedup_name) {
  std::string name;
  if (node->IsValidName()) {
    name = node->name();
  } else {
    std::string class_name = node->class_name();
    for (int i = 1;; ++i) {
      name = class_name + std::to_string(i);
      if (dedup_name->count(name) == 0) {
        DXINFO("Node name is invalid, a valid name is generated: %s.",
               name.c_str());
        node->name_ = name;
        break;
      }
    }
  }
  dedup_name->emplace(std::move(name));
}

void Graph::CompileTarget(GraphNode* node,
                          std::unordered_set<GraphNode*>* dedup_node,
                          std::unordered_set<std::string>* dedup_name,
                          GraphTarget* target) {
  CheckName(node, dedup_name);

  if (dedup_node->count(node) > 0) {
    return;
  }

  for (GraphNode* input : node->input()) {
    CompileTarget(input, dedup_node, dedup_name, target);
    if (input->need_grad()) {
      node->set_need_grad(1);
    }
  }

  dedup_node->emplace(node);
  target->forward_.emplace_back(node);
  target->forward_name_.emplace_back(node->name());
}

void Graph::CompileTarget(GraphNode* node,
                          std::unordered_set<std::string>* dedup_name,
                          GraphTarget* target) {
  std::unordered_set<GraphNode*> dedup_node;
  CheckName(node, dedup_name);
  target->node_ = node;
  target->name_ = node->name();
  CompileTarget(node, &dedup_node, dedup_name, target);
}

void Graph::PostInit() {
  // fill 'node_id_2_target_'
  node_id_2_target_.reserve(name_2_target_.size());
  for (auto& entry : name_2_target_) {
    GraphTarget* target = entry.second;
    node_id_2_target_.emplace(target->node_id(), target);
  }

  // fill 'node_id_2_node_'
  node_id_2_node_.reserve(name_2_node_.size());
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    node_id_2_node_.emplace(node->node_id(), node);
  }

  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    node->output_.clear();
    node->input_fork_ = 0;
    node->is_target_ = 0;
  }

  // fill 'GraphNode::output_'
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    for (GraphNode* input : node->input()) {
      input->output_.emplace(node);
    }
  }

  // fill 'GraphNode::input_fork_'
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    for (GraphNode* input : node->input()) {
      if (input->output_size() > 1) {
        node->input_fork_ = 1;
        break;
      }
    }
  }

  // fill 'GraphNode::is_target_'
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    if (find_target(node->node_id())) {
      node->is_target_ = 1;
    }
  }
}

void Graph::clear() noexcept {
  compiled_ = 0;
  target_.clear();
  name_2_target_.clear();
  name_2_node_.clear();
  heap_node_.clear();
  meta_.clear();
  node_id_2_target_.clear();
  node_id_2_node_.clear();
}

bool Graph::Compile(const std::vector<GraphNode*>& target_nodes, int on_heap) {
  clear();

  // fill 'target_', 'name_2_target_'
  std::unordered_set<GraphNode*> dedup_node;
  std::unordered_set<std::string> dedup_name;
  size_t target_size = target_nodes.size();
  target_.resize(target_size);
  name_2_target_.reserve(target_size);
  for (size_t i = 0; i < target_size; ++i) {
    GraphTarget& target = target_[i];
    CompileTarget(target_nodes[i], &dedup_name, &target);
    name_2_target_.emplace(target.name(), &target);
    dedup_node.insert(target.forward().begin(), target.forward().end());
  }

  // sort by name to get determinant iteration order
  std::vector<GraphNode*> sorted(dedup_node.begin(), dedup_node.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const GraphNode* a, const GraphNode* b) {
              return a->name() > b->name();
            });

  // fill 'name_2_node_'
  name_2_node_.reserve(sorted.size());
  for (GraphNode* node : sorted) {
    if (name_2_node_.count(node->name()) > 0) {
      DXERROR("Duplicate node name: %s.", node->name().c_str());
      return false;
    }
    name_2_node_.emplace(node->name(), node);
  }

  if (on_heap) {
    // fill 'heap_node_'
    heap_node_.reserve(sorted.size());
    for (GraphNode* node : sorted) {
      heap_node_.emplace_back(node);
    }
  }

  // fill 'GraphNode::node_id_'
  int node_id = 0;
  for (GraphNode* node : sorted) {
    node->node_id_ = (uint16_t)node_id;
    ++node_id;
    if (node_id > (int)std::numeric_limits<uint16_t>::max()) {
      DXERROR("Too large node id: %d.", node_id);
      return false;
    }
  }

  PostInit();
  compiled_ = 1;
  return true;
}

bool Graph::Write(OutputStream& os) const {
  if (!compiled_) {
    DXERROR("The graph is not compiled.");
    return false;
  }

  int version = 1;
  os << version;

  int node_size = (int)name_2_node_.size();
  os << node_size;
  std::string class_name;
  for (const auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    class_name = node->class_name();
    os << class_name;
    node->Write(os);
  }

  int target_size = (int)target_.size();
  os << target_size;
  for (const GraphTarget& target : target_) {
    os << target.name() << target.forward_name();
  }

  os << meta_;

  if (!os) {
    DXERROR("Failed to write graph.");
    return false;
  }

  DXINFO("Wrote a graph with %d nodes.", node_size);
  return true;
}

bool Graph::Read(InputStream& is) {
  clear();

  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read graph.");
    return false;
  }

  if (version > 1) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  int node_size;
  is >> node_size;
  if (!is) {
    DXERROR("Failed to read graph.");
    return false;
  }

  std::string class_name;
  for (int i = 0; i < node_size; ++i) {
    is >> class_name;
    if (!is) {
      DXERROR("Failed to read graph.");
      return false;
    }

    std::unique_ptr<GraphNode> node(GRAPH_NODE_NEW(class_name));
    if (!node) {
      DXERROR("Failed to create graph node: %s.", class_name.c_str());
      is.set_bad();
      return false;
    }

    node->Read(is);
    if (!is) {
      DXERROR("Failed to read graph.");
      return false;
    }

    name_2_node_.emplace(node->name(), node.get());
    heap_node_.emplace_back(std::move(node));
  }

  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second;
    for (const std::string& input_name : node->input_name_) {
      node->input_.emplace_back(name_2_node_.at(input_name));
    }
  }

  int target_size;
  is >> target_size;
  if (!is) {
    DXERROR("Failed to read graph.");
    return false;
  }

  target_.resize(target_size);
  for (GraphTarget& target : target_) {
    is >> target.name_ >> target.forward_name_;
    if (!is) {
      DXERROR("Failed to read graph.");
      return false;
    }

    target.node_ = name_2_node_.at(target.name());
    target.forward_.reserve(target.forward_name_size());
    for (const std::string& name : target.forward_name()) {
      target.forward_.emplace_back(name_2_node_.at(name));
    }

    name_2_target_.emplace(target.name(), &target);
  }

  if (version == 1) {
    is >> meta_;
    if (!is) {
      DXERROR("Failed to read graph.");
      return false;
    }
  }

  PostInit();
  compiled_ = 1;
  DXINFO("Read a graph with %zu nodes.", name_2_node_.size());
  return true;
}

static bool ExcludeDotNode(const GraphNode* node) noexcept {
  if (node->type_index() != typeid(VariableNode)) {
    return false;
  }

  for (const GraphNode* output : node->output()) {
    if (output->type_index() != typeid(GroupEmbeddingLookupNode) &&
        output->type_index() != typeid(GroupEmbeddingLookup2Node)) {
      return false;
    }
  }
  return true;
}

static std::string WriteDotNodeAttr(const GraphNode* node) {
  std::string attr;
  if (node->type_index() == typeid(GroupEmbeddingLookupNode)) {
    for (uint16_t group_id :
         ((const GroupEmbeddingLookupNode*)node)->group_ids()) {
      attr += "group_id=" + std::to_string(group_id) + "\\n";
    }
  } else if (node->type_index() == typeid(GroupEmbeddingLookup2Node)) {
    for (uint16_t group_id :
         ((const GroupEmbeddingLookup2Node*)node)->group_ids()) {
      attr += "group_id=" + std::to_string(group_id) + "\\n";
    }
  }
  return attr;
}

static void WriteDotNode(const GraphNode* node, std::set<std::string>* dedup) {
  if (!ExcludeDotNode(node)) {
    std::string text;
    const char* tensor_type;
    const char* color;

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR:
        tensor_type = "TSR";
        break;
      case TENSOR_TYPE_SRM:
        tensor_type = "SRM";
        break;
      case TENSOR_TYPE_CSR:
        tensor_type = "CSR";
        break;
      case TENSOR_TYPE_TSRI:
        tensor_type = "TSRI";
        break;
      case TENSOR_TYPE_TSRS:
        tensor_type = "TSRS";
        break;
      default:
        tensor_type = "";
        break;
    }

    switch (node->node_type()) {
      case GRAPH_NODE_TYPE_PARAM:
        color = "salmon";
        break;
      case GRAPH_NODE_TYPE_INSTANCE:
        color = "skyblue1";
        break;
      case GRAPH_NODE_TYPE_HIDDEN:
        color = "pink";
        break;
      default:
        color = "white";
        break;
    }

    text += "    ";
    text += node->name();
    text += "[label=\"";
    text += node->class_name();
    text += "\\n";
    text += node->name();
    text += " ";
    text += to_string(node->shape());
    text += " ";
    text += tensor_type;
    text += "\\n";
    text += WriteDotNodeAttr(node);
    text += "\", style=filled, fillcolor=";
    text += color;
    text += "];";
    dedup->emplace(text);
  }

  for (const GraphNode* input : node->input()) {
    WriteDotNode(input, dedup);
  }
}

static void WriteDotEdge(const GraphNode* node, std::set<std::string>* dedup) {
  for (const GraphNode* input : node->input()) {
    if (!ExcludeDotNode(input)) {
      std::string text;
      text += "    ";
      text += input->name();
      text += " -> ";
      text += node->name();
      text += ";";
      dedup->emplace(text);
    }
    WriteDotEdge(input, dedup);
  }
}

bool Graph::WriteDot(std::string* s) const {
  if (!compiled_) {
    DXERROR("The graph is not compiled.");
    return false;
  }

  std::ostringstream os;
  os << "digraph deepx {" << std::endl;
  os << "    node [ shape = \"record\" ];" << std::endl;
  os << "    rankdir = BT;" << std::endl;
  std::set<std::string> nodes, edges;
  for (const GraphTarget& target : target_) {
    WriteDotNode(target.node(), &nodes);
    WriteDotEdge(target.node(), &edges);
  }
  for (const std::string& node : nodes) {
    os << node << std::endl;
  }
  for (const std::string& edge : edges) {
    os << edge << std::endl;
  }
  os << "}" << std::endl;
  *s = os.str();
  return true;
}

bool Graph::Save(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving graph to %s...", file.c_str());
  if (!Write(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Graph::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading graph from %s...", file.c_str());
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Graph::SaveDot(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving graph to %s...", file.c_str());
  std::string s;
  if (!WriteDot(&s)) {
    return false;
  }
  os.Write(s.data(), s.size());
  if (!os) {
    DXERROR("Failed to write graph.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

std::string Graph::Dot() const {
  std::string s;
  (void)WriteDot(&s);
  return s;
}

bool Graph::ParseFromString(const std::string& s) {
  InputStringStream is;
  is.SetView(s);
  DXINFO("Parsing graph...");
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

}  // namespace deepx_core
