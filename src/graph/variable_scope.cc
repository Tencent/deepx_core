// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/variable_scope.h>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* VariableScopeManager */
/************************************************************************/
std::string VariableScopeManager::GetVariableName(
    const std::string& name) const {
  std::string variable_name;
  for (size_t i = 0; i < scopes_.size(); ++i) {  // NOLINT
    variable_name += scopes_[i];
    variable_name += "/";
  }
  variable_name += name;
  return variable_name;
}

VariableNode* VariableScopeManager::_GetVariable(
    variable_node_ptr_t& node, variable_node_ptr_t& new_node) {
  if (node) {
    DXCHECK_THROW(node->name() == new_node->name());
    DXCHECK_THROW(node->node_type() == new_node->node_type());
    DXCHECK_THROW(node->tensor_type() == new_node->tensor_type());
    DXCHECK_THROW(node->shape() == new_node->shape());
  } else {
    node = std::move(new_node);
  }
  return node.get();
}

VariableScopeManager& VariableScopeManager::Get() {
  static VariableScopeManager manager;
  return manager;
}

void VariableScopeManager::EnterScope(const std::string& scope) {
  scopes_.emplace_back(scope);
}

void VariableScopeManager::LeaveScope() {
  DXCHECK_THROW(!scopes_.empty());
  scopes_.pop_back();
}

void VariableScopeManager::ClearVariable() { variable_map_.clear(); }

void VariableScopeManager::ReleaseVariable() {
  for (auto& entry : variable_map_) {
    variable_node_ptr_t& node = entry.second;
    (void)node.release();
  }
  variable_map_.clear();
}

VariableNode* VariableScopeManager::GetVariable(const std::string& name,
                                                const Shape& shape,
                                                int tensor_type) {
  std::string variable_name = GetVariableName(name);
  variable_node_ptr_t new_node(
      new VariableNode(variable_name, shape, tensor_type));
  variable_node_ptr_t& node = variable_map_[variable_name];
  return _GetVariable(node, new_node);
}

VariableNode* VariableScopeManager::GetVariable(const std::string& name,
                                                const Shape& shape,
                                                int tensor_type,
                                                int initializer_type,
                                                double initializer_param1,
                                                double initializer_param2) {
  std::string variable_name = GetVariableName(name);
  variable_node_ptr_t new_node(
      new VariableNode(variable_name, shape, tensor_type, initializer_type,
                       initializer_param1, initializer_param2));
  variable_node_ptr_t& node = variable_map_[variable_name];
  return _GetVariable(node, new_node);
}

VariableNode* VariableScopeManager::GetVariable(const std::string& name,
                                                const Shape& shape) {
  std::string variable_name = GetVariableName(name);
  variable_node_ptr_t new_node(new VariableNode(variable_name, shape));
  variable_node_ptr_t& node = variable_map_[variable_name];
  return _GetVariable(node, new_node);
}

VariableNode* VariableScopeManager::GetVariable(const std::string& name,
                                                const Shape& shape,
                                                int initializer_type,
                                                double initializer_param1,
                                                double initializer_param2) {
  std::string variable_name = GetVariableName(name);
  variable_node_ptr_t new_node(
      new VariableNode(variable_name, shape, initializer_type,
                       initializer_param1, initializer_param2));
  variable_node_ptr_t& node = variable_map_[variable_name];
  return _GetVariable(node, new_node);
}

/************************************************************************/
/* VariableScopeEnterer */
/************************************************************************/
VariableScopeEnterer::VariableScopeEnterer(const std::string& scope) {
  VariableScopeManager::Get().EnterScope(scope);
}

VariableScopeEnterer::~VariableScopeEnterer() {
  VariableScopeManager::Get().LeaveScope();
}

/************************************************************************/
/* VariableScope functions */
/************************************************************************/
void EnterScope(const std::string& scope) {
  VariableScopeManager::Get().EnterScope(scope);
}

void LeaveScope() { VariableScopeManager::Get().LeaveScope(); }

void ClearVariable() { VariableScopeManager::Get().ClearVariable(); }

void ReleaseVariable() { VariableScopeManager::Get().ReleaseVariable(); }

VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int tensor_type) {
  return VariableScopeManager::Get().GetVariable(name, shape, tensor_type);
}

VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int tensor_type, int initializer_type,
                          double initializer_param1,
                          double initializer_param2) {
  return VariableScopeManager::Get().GetVariable(
      name, shape, tensor_type, initializer_type, initializer_param1,
      initializer_param2);
}

VariableNode* GetVariable(const std::string& name, const Shape& shape) {
  return VariableScopeManager::Get().GetVariable(name, shape);
}

VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int initializer_type, double initializer_param1,
                          double initializer_param2) {
  return VariableScopeManager::Get().GetVariable(
      name, shape, initializer_type, initializer_param1, initializer_param2);
}

VariableNode* GetVariableZeros(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
}

VariableNode* GetVariableOnes(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_ONES, 0, 0);
}

VariableNode* GetVariableConstant(const std::string& name, const Shape& shape,
                                  double c) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_CONSTANT, c, 0);
}

VariableNode* GetVariableRand(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RAND, 0, 1);
}

VariableNode* GetVariableRand(const std::string& name, const Shape& shape,
                              double _min, double _max) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RAND, _min, _max);
}

VariableNode* GetVariableRandn(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
}

VariableNode* GetVariableRandn(const std::string& name, const Shape& shape,
                               double mean, double stddev) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RANDN, mean, stddev);
}

VariableNode* GetVariableRandLecun(const std::string& name,
                                   const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RAND_LECUN, 0, 0);
}

VariableNode* GetVariableRandnLecun(const std::string& name,
                                    const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RANDN_LECUN, 0, 0);
}

VariableNode* GetVariableRandXavier(const std::string& name,
                                    const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RAND_XAVIER, 0, 0);
}

VariableNode* GetVariableRandnXavier(const std::string& name,
                                     const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RANDN_XAVIER, 0, 0);
}

VariableNode* GetVariableRandHe(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RAND_HE, 0, 0);
}

VariableNode* GetVariableRandnHe(const std::string& name, const Shape& shape) {
  return GetVariable(name, shape, TENSOR_INITIALIZER_TYPE_RANDN_HE, 0, 0);
}

}  // namespace deepx_core
