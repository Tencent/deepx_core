// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <string>
#include <utility>  // std::move
#include "simp_item.h"

namespace deepx_core {

/************************************************************************/
/* Simp */
/************************************************************************/
class Simp {
 private:
  const std::string name_;

 public:
  const std::string& name() const noexcept { return name_; }

 public:
  explicit Simp(std::string name) : name_(std::move(name)) {}
  virtual ~Simp() = default;
  virtual bool Simplify(SimpItem* item) const = 0;
};

}  // namespace deepx_core
