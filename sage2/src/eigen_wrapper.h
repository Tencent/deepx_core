// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// NOTE: rename Eigen to avoid any potential linkage issues.
#define Eigen sage2_eigen
#include <Eigen/Core>
#include <type_traits>

namespace sage2 {

/************************************************************************/
/* array */
/************************************************************************/
using eigen_arrayxf_view_t = Eigen::Map<Eigen::ArrayXf, Eigen::Unaligned>;

template <typename Int>
eigen_arrayxf_view_t make_eigen_arrayxf_view(float* buf, Int n) {
  return eigen_arrayxf_view_t(buf, (eigen_arrayxf_view_t::Index)n);
}

template <typename Int>
const eigen_arrayxf_view_t make_eigen_arrayxf_view(const float* buf, Int n) {
  return eigen_arrayxf_view_t((float*)buf, (eigen_arrayxf_view_t::Index)n);
}

/************************************************************************/
/* matrix */
/************************************************************************/
using eigen_matrix_ld_t = Eigen::OuterStride<Eigen::Dynamic>;
using eigen_matrix_rm_t = std::integral_constant<int, Eigen::RowMajor>;
using eigen_matrix_cm_t = std::integral_constant<int, Eigen::ColMajor>;
template <int OPTIONS>
using eigen_matrixxf_t =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, OPTIONS>;
template <int OPTIONS>
using eigen_matrixxf_view_t =
    Eigen::Map<eigen_matrixxf_t<OPTIONS>, Eigen::Unaligned, eigen_matrix_ld_t>;

template <typename Int, typename RowOrColMajor>
eigen_matrixxf_view_t<RowOrColMajor::value> make_eigen_matrixxf_view(
    float* buf, Int row, Int col, Int ld, RowOrColMajor) {
  static_assert(
      RowOrColMajor::value == Eigen::RowMajor ||
          RowOrColMajor::value == Eigen::ColMajor,
      "RowOrColMajor must be eigen_matrix_rm_t or eigen_matrix_cm_t.");
  using matrix_t = eigen_matrixxf_view_t<RowOrColMajor::value>;
  using index_t = typename matrix_t::Index;
  return matrix_t(buf, (index_t)row, (index_t)col,
                  eigen_matrix_ld_t((index_t)ld));
}

template <typename Int, typename RowOrColMajor>
const eigen_matrixxf_view_t<RowOrColMajor::value> make_eigen_matrixxf_view(
    const float* buf, Int row, Int col, Int ld, RowOrColMajor) {
  static_assert(
      RowOrColMajor::value == Eigen::RowMajor ||
          RowOrColMajor::value == Eigen::ColMajor,
      "RowOrColMajor must be eigen_matrix_rm_t or eigen_matrix_cm_t.");
  using matrix_t = eigen_matrixxf_view_t<RowOrColMajor::value>;
  using index_t = typename matrix_t::Index;
  return matrix_t((float*)buf, (index_t)row, (index_t)col,
                  eigen_matrix_ld_t((index_t)ld));
}

}  // namespace sage2
