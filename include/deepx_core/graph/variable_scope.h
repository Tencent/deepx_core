// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph_node.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* VariableScopeManager */
/************************************************************************/
class VariableScopeManager {
 private:
  std::vector<std::string> scopes_;
  using variable_node_ptr_t = std::unique_ptr<VariableNode>;
  std::unordered_map<std::string, variable_node_ptr_t> variable_map_;

 private:
  VariableScopeManager() = default;
  VariableScopeManager(const VariableScopeManager& other) = delete;
  VariableScopeManager& operator=(const VariableScopeManager& other) = delete;

 private:
  std::string GetVariableName(const std::string& name) const;
  VariableNode* _GetVariable(variable_node_ptr_t& node,       // NOLINT
                             variable_node_ptr_t& new_node);  // NOLINT

 public:
  static VariableScopeManager& Get();
  void EnterScope(const std::string& scope);
  void LeaveScope();
  void ClearVariable();
  void ReleaseVariable();
  VariableNode* GetVariable(const std::string& name, const Shape& shape,
                            int tensor_type);
  VariableNode* GetVariable(const std::string& name, const Shape& shape,
                            int tensor_type, int initializer_type,
                            double initializer_param1,
                            double initializer_param2);
  VariableNode* GetVariable(const std::string& name, const Shape& shape);
  VariableNode* GetVariable(const std::string& name, const Shape& shape,
                            int initializer_type, double initializer_param1,
                            double initializer_param2);
};

/************************************************************************/
/* VariableScopeEnterer */
/************************************************************************/
class VariableScopeEnterer {
 public:
  explicit VariableScopeEnterer(const std::string& scope);
  ~VariableScopeEnterer();
  VariableScopeEnterer(const VariableScopeEnterer& other) = delete;
  VariableScopeEnterer& operator=(const VariableScopeEnterer& other) = delete;
};

/************************************************************************/
/* VariableScope functions */
/************************************************************************/
void EnterScope(const std::string& scope);
void LeaveScope();
void ClearVariable();
void ReleaseVariable();
VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int tensor_type);
VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int tensor_type, int initializer_type,
                          double initializer_param1, double initializer_param2);
VariableNode* GetVariable(const std::string& name, const Shape& shape);
VariableNode* GetVariable(const std::string& name, const Shape& shape,
                          int initializer_type, double initializer_param1,
                          double initializer_param2);

VariableNode* GetVariableZeros(const std::string& name, const Shape& shape);
VariableNode* GetVariableOnes(const std::string& name, const Shape& shape);
VariableNode* GetVariableConstant(const std::string& name, const Shape& shape,
                                  double c);
VariableNode* GetVariableRand(const std::string& name, const Shape& shape);
VariableNode* GetVariableRand(const std::string& name, const Shape& shape,
                              double _min, double _max);
VariableNode* GetVariableRandn(const std::string& name, const Shape& shape);
VariableNode* GetVariableRandn(const std::string& name, const Shape& shape,
                               double mean, double stddev);
VariableNode* GetVariableRandLecun(const std::string& name, const Shape& shape);
VariableNode* GetVariableRandnLecun(const std::string& name,
                                    const Shape& shape);
VariableNode* GetVariableRandXavier(const std::string& name,
                                    const Shape& shape);
VariableNode* GetVariableRandnXavier(const std::string& name,
                                     const Shape& shape);
VariableNode* GetVariableRandHe(const std::string& name, const Shape& shape);
VariableNode* GetVariableRandnHe(const std::string& name, const Shape& shape);

}  // namespace deepx_core
