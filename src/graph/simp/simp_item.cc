// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "simp_item.h"
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <algorithm>  // std::sort
#include <list>
#include <queue>

namespace deepx_core {

bool SimpItem::is_target(const std::string& name) const noexcept {
  return name_2_target_.count(name) > 0;
}

GraphNode* SimpItem::find_node(const std::string& name) noexcept {
  auto it = name_2_node_.find(name);
  if (it == name_2_node_.end()) {
    return nullptr;
  }
  return it->second.get();
}

const GraphNode* SimpItem::find_node(const std::string& name) const noexcept {
  auto it = name_2_node_.find(name);
  if (it == name_2_node_.end()) {
    return nullptr;
  }
  return it->second.get();
}

auto SimpItem::find_output(const std::string& name) const noexcept
    -> const node_set_t {
  auto it = name_2_output_.find(name);
  if (it == name_2_output_.end()) {
    return node_set_t{};
  }
  return it->second;
}

void SimpItem::clear() noexcept {
  target_.clear();
  name_2_target_.clear();
  name_2_node_.clear();
  name_2_output_.clear();
}

void SimpItem::Add(GraphNode* node) {
  for (const GraphNode* input : node->input()) {
    DXCHECK_THROW(name_2_node_.count(input->name()) > 0);
  }

  if (name_2_node_.count(node->name()) == 0) {
    name_2_node_.emplace(node->name(), heap_node_t(node));
    for (GraphNode* input : node->input()) {
      name_2_output_[input->name()].insert(node);
    }
  } else {
    DXCHECK_THROW(name_2_node_.at(node->name()).get() == node);
  }
}

void SimpItem::Prune() {
  std::unordered_set<std::string> to_erase;
  to_erase.reserve(name_2_node_.size());
  for (const auto& entry : name_2_node_) {
    to_erase.insert(entry.first);
  }

  std::queue<std::string, std::list<std::string>> to_visit;
  for (const GraphNode* target : target_) {
    to_erase.erase(target->name());
    to_visit.push(target->name());
  }

  while (!to_visit.empty()) {
    std::string front = to_visit.front();
    to_visit.pop();
    for (const GraphNode* node : name_2_node_.at(front)->input()) {
      to_erase.erase(node->name());
      to_visit.push(node->name());
    }
  }

  for (const std::string& name : to_erase) {
    name_2_node_.erase(name);
  }

  BuildName2Output();
}

bool SimpItem::ReplaceInput(const std::string& name,
                            const std::string& replaced,
                            const std::string& replacement) {
  if (name_2_node_.count(name) == 0 || name_2_node_.count(replaced) == 0 ||
      name_2_node_.count(replacement) == 0) {
    return false;
  }
  DXCHECK_THROW(!IsReachable(find_node(replacement), find_node(name)));

  auto& node = name_2_node_[name];
  auto& replacement_node = name_2_node_[replacement];
  for (int i = 0; i < node->input_size(); ++i) {
    if (node->input(i)->name() == replaced) {
      node->input_[i] = replacement_node.get();
    }
  }
  name_2_output_[replaced].erase(node.get());
  name_2_output_[replacement].insert(node.get());
  return true;
}

bool SimpItem::ReplaceInputOfAllOutputs(const std::string& replaced,
                                        const std::string& replacement) {
  if (name_2_node_.count(replaced) == 0 ||
      name_2_node_.count(replacement) == 0) {
    return false;
  }
  for (auto* output : find_output(replaced)) {
    ReplaceInput(output->name(), replaced, replacement);
  }
  return true;
}

void SimpItem::GetTopologicalSortedNodes(std::vector<GraphNode*>* sorted,
                                         bool reverse) const {
  std::unordered_map<std::string, int> num_ready_input;
  num_ready_input.reserve(node_size());
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second.get();
    num_ready_input.emplace(node->name(), 0);
  }

  auto comparator_by_name = [](const GraphNode* a, const GraphNode* b) {
    return a->name() < b->name();
  };

  sorted->clear();
  sorted->reserve(node_size());
  for (auto& entry : name_2_node_) {
    GraphNode* node = entry.second.get();
    if (node->input_size() == 0) {
      sorted->emplace_back(node);
    }
  }
  std::sort(sorted->begin(), sorted->end(), comparator_by_name);

  std::vector<GraphNode*> ready;
  int front = 0;
  while (front < (int)sorted->size()) {
    GraphNode* node = (*sorted)[front];
    ready.clear();
    for (auto* output : find_output(node->name())) {
      for (auto* input : output->input()) {
        if (input == node) {
          ++num_ready_input[output->name()];
        }
      }
      if (num_ready_input[output->name()] == (int)output->input_size()) {
        ready.emplace_back(output);
      }
    }
    std::sort(ready.begin(), ready.end(), comparator_by_name);
    sorted->insert(sorted->end(), ready.begin(), ready.end());

    ++front;
  }

  if (reverse) {
    std::reverse(sorted->begin(), sorted->end());
  }
}

std::string SimpItem::NewNodeName(const std::string& old_name,
                                  const std::vector<std::string>& append_scopes,
                                  const std::string& prefix,
                                  const std::string& suffix) const noexcept {
  std::string scope, base_name;
  auto pos = old_name.find_last_of("/");
  if (pos != std::string::npos) {
    scope = old_name.substr(0, pos);
    base_name = old_name.substr(pos + 1);
  } else {
    base_name = old_name;
  }
  std::string new_name;
  if (!scope.empty()) {
    new_name.append(scope + "/");
  }
  for (auto& append : append_scopes) {
    new_name.append(append + "/");
  }
  if (!prefix.empty()) {
    new_name.append(prefix + "_");
  }
  new_name.append(base_name);
  if (!suffix.empty()) {
    new_name.append("_" + suffix);
  }
  while (find_node(new_name) != nullptr) {
    new_name.append("_new");
  }
  return new_name;
}

bool SimpItem::FromGraph(const deepx_core::Graph& graph) {
  clear();

  if (!Clone(graph.name_2_node(), &name_2_node_)) {
    return false;
  }

  target_.reserve(graph.target().size());
  for (auto& graph_target : graph.target()) {
    const std::string& name = graph_target.name();
    GraphNode* target = name_2_node_.at(name).get();
    target_.emplace_back(target);
    name_2_target_.emplace(name, target);
  }

  BuildName2Output();
  return true;
}

bool SimpItem::ToGraph(deepx_core::Graph* graph) {
  graph->clear();

  Prune();

  std::unordered_map<std::string, GraphNode*> name_2_node;
  if (!Clone(name_2_node_, &name_2_node)) {
    return false;
  }

  std::vector<GraphNode*> target;
  target.reserve(target_.size());
  for (auto& graph_target : target_) {
    target.emplace_back(name_2_node.at(graph_target->name()));
  }

  return graph->Compile(target, 1);
}

void SimpItem::BuildName2Output() {
  name_2_output_.clear();

  int node_size = (int)name_2_node_.size();
  name_2_output_.reserve(node_size);
  for (auto& entry : name_2_node_) {
    auto& node = entry.second;
    for (auto* input : node->input()) {
      name_2_output_[input->name()].insert(node.get());
    }
  }
}

bool SimpItem::IsReachable(GraphNode* from, GraphNode* to) const {
  node_set_t visited;
  std::vector<GraphNode*> to_visit;
  to_visit.emplace_back(from);
  while (!to_visit.empty()) {
    const GraphNode* top = to_visit.back();
    to_visit.pop_back();
    if (top == to) {
      return true;
    }
    for (auto* input : top->input()) {
      if (visited.count(input) == 0) {
        to_visit.emplace_back(input);
        visited.insert(input);
      }
    }
  }
  return false;
}

bool SimpItem::Clone(GraphNode& from, GraphNode** to) {
  std::string class_name = from.class_name();

  OutputStringStream os;
  from.Write(os);
  if (!os) {
    DXERROR("Failed to write graph node: %s.", class_name.c_str());
    return false;
  }

  *to = GRAPH_NODE_NEW(class_name);
  if (*to == nullptr) {
    DXERROR("Failed to create graph node: %s.", class_name.c_str());
    return false;
  }

  InputStringStream is;
  is.SetView(os.GetData(), os.GetSize());
  (*to)->Read(is);
  if (!is) {
    DXERROR("Failed to read graph node: %s.", class_name.c_str());
    return false;
  }
  return true;
}

bool SimpItem::Clone(const heap_node_map_t& from, node_map_t* to) {
  to->clear();

  for (auto& entry : from) {
    const std::string& name = entry.first;
    GraphNode* node = entry.second.get();
    GraphNode* new_node = nullptr;
    if (!Clone(*node, &new_node)) {
      return false;
    }
    to->emplace(name, new_node);
  }

  for (auto& entry : *to) {
    GraphNode* node = entry.second;
    for (auto& input_name : node->input_name_) {
      node->input_.emplace_back(to->at(input_name));
    }
  }

  return true;
}

bool SimpItem::Clone(const node_map_t& from, heap_node_map_t* to) {
  to->clear();

  for (auto& entry : from) {
    const std::string& name = entry.first;
    GraphNode* node = entry.second;
    GraphNode* new_node = nullptr;
    if (!Clone(*node, &new_node)) {
      return false;
    }
    to->emplace(name, heap_node_t(new_node));
  }

  for (auto& entry : *to) {
    auto& node = entry.second;
    for (auto& input_name : node->input_name_) {
      node->input_.emplace_back(to->at(input_name).get());
    }
  }

  return true;
}

}  // namespace deepx_core
