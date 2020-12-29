// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/tensor/shape.h>
#include <sstream>

namespace deepx_core {

OutputStream& operator<<(OutputStream& os, const Shape& shape) {
  os << shape.rank_ << shape.total_dim_ << shape.dim_;
  return os;
}

InputStream& operator>>(InputStream& is, Shape& shape) {
  is >> shape.rank_ >> shape.total_dim_ >> shape.dim_;
  return is;
}

Shape::Shape(std::initializer_list<int> il) { assign(il.begin(), il.end()); }

Shape& Shape::operator=(std::initializer_list<int> il) {
  assign(il.begin(), il.end());
  return *this;
}

Shape::Shape(const std::vector<int>& dim) { assign(dim.begin(), dim.end()); }

Shape& Shape::operator=(const std::vector<int>& dim) {
  assign(dim.begin(), dim.end());
  return *this;
}

void Shape::Construct(int dim) noexcept {
  total_dim_ *= dim;
  dim_[rank_++] = dim;
}

bool Shape::real_axis(int* axis) const noexcept {
  if (*axis >= rank_ || *axis < -rank_) {
    return false;
  }

  if (*axis < 0) {
    *axis += rank_;
  }
  return true;
}

int Shape::real_axis(int axis) const noexcept {
  if (axis >= rank_ || axis < -rank_) {
    return SHAPE_INVALID_AXIS;
  }

  if (axis < 0) {
    axis += rank_;
  }
  return axis;
}

bool Shape::do_reshape_nothrow(const Shape& other) noexcept {
  int neg = 0, other_neg = 0;
  for (int i = 0; i < rank_; ++i) {
    if (dim_[i] == SHAPE_DIM_ANY) {
      ++neg;
    }
  }
  for (int i = 0; i < other.rank_; ++i) {
    if (other.dim_[i] == SHAPE_DIM_ANY) {
      ++other_neg;
    }
  }

  do {
    if (neg > 1 || other_neg > 1) {
      break;
    }

    if (neg == 0 && other_neg == 0) {
      if (total_dim_ != other.total_dim_) {
        break;
      }
      *this = other;
      return true;
    } else if (neg == 0) {
      if (total_dim_ == 0 || other.total_dim_ == 0) {
        break;
      }

      int a = total_dim_ / (-other.total_dim_);
      int b = total_dim_ % (-other.total_dim_);
      if (b != 0) {
        break;
      }

      *this = other;
      for (int i = 0; i < rank_; ++i) {
        if (dim_[i] == SHAPE_DIM_ANY) {
          dim_[i] = a;
          total_dim_ *= a / SHAPE_DIM_ANY;
          break;
        }
      }
      return true;
    } else if (other_neg == 0) {
      if (total_dim_ == 0) {
        break;
      }

      int b = other.total_dim_ % (-total_dim_);
      if (b != 0) {
        break;
      }
      *this = other;
      return true;
    } else {
      *this = other;
      return true;
    }
  } while (0);  // NOLINT
  return false;
}

bool Shape::do_expand_dim_nothrow(int axis) noexcept {
  if (rank_ == SHAPE_MAX_RANK) {
    return false;
  }

  ++rank_;
  if (!real_axis(&axis)) {
    return false;
  }

  for (int i = rank_ - 1; i > axis; --i) {
    dim_[i] = dim_[i - 1];
  }
  dim_[axis] = 1;
  return true;
}

bool Shape::do_squeeze_nothrow(int axis) noexcept {
  if (!real_axis(&axis)) {
    return false;
  }

  if (dim_[axis] != 1) {
    return false;
  }

  for (int i = axis; i < rank_ - 1; ++i) {
    dim_[i] = dim_[i + 1];
  }
  dim_[rank_ - 1] = 1;
  --rank_;
  return true;
}

Shape& Shape::reshape(const Shape& other) {
  if (!do_reshape_nothrow(other)) {
    DXTHROW_INVALID_ARGUMENT("Couldn't reshape from %s to %s.",
                             to_string(*this).c_str(),
                             to_string(other).c_str());
  }
  return *this;
}

Shape& Shape::reshape_nothrow(const Shape& other) noexcept {
  if (!do_reshape_nothrow(other)) {
    DXERROR("Couldn't reshape from %s to %s.", to_string(*this).c_str(),
            to_string(other).c_str());
    clear();
  }
  return *this;
}

Shape& Shape::expand_dim(int axis) {
  if (!do_expand_dim_nothrow(axis)) {
    DXTHROW_INVALID_ARGUMENT("Couldn't expand_dim %s by %d.",
                             to_string(*this).c_str(), axis);
  }
  return *this;
}

Shape& Shape::expand_dim_nothrow(int axis) noexcept {
  if (!do_expand_dim_nothrow(axis)) {
    DXERROR("Couldn't expand_dim %s by %d.", to_string(*this).c_str(), axis);
    clear();
  }
  return *this;
}

Shape& Shape::squeeze(int axis) {
  if (!do_squeeze_nothrow(axis)) {
    DXTHROW_INVALID_ARGUMENT("Couldn't squeeze %s by %d.",
                             to_string(*this).c_str(), axis);
  }
  return *this;
}

Shape& Shape::squeeze_nothrow(int axis) noexcept {
  if (!do_squeeze_nothrow(axis)) {
    DXERROR("Couldn't squeeze %s by %d.", to_string(*this).c_str(), axis);
    clear();
  }
  return *this;
}

bool Shape::same_shape(const Shape& other) const noexcept {
  if (rank_ != other.rank_) {
    return false;
  }
  for (int i = 0; i < rank_; ++i) {
    if (dim_[i] != other.dim_[i]) {
      return false;
    }
  }
  return true;
}

std::string to_string(const Shape& shape) {
  std::ostringstream os;
  os << "(";
  for (int i = 0; i < shape.rank(); ++i) {
    os << shape[i];
    if (i != shape.rank() - 1) {
      os << ",";
    }
  }
  os << ")";
  return os.str();
}

}  // namespace deepx_core
