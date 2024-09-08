// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions euclidean dot_product and dot_product with a metric

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean dot product of two vectors or one forms
 *
 * \details
 * Returns \f$A^a B^b \delta_{ab}\f$ for input vectors \f$A^a\f$ and \f$B^b\f$
 * or \f$A_a B_b \delta^{ab}\f$ for input one forms \f$A_a\f$ and \f$B_b\f$.
 */
template <typename DataTypeLhs, typename DataTypeRhs, typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(DataTypeLhs() *
                                                             DataTypeRhs()))>
void dot_product(
    const gsl::not_null<Scalar<DataTypeResult>*> dot_product,
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>, index_list<Index>>& vector_b) {
  get(*dot_product) = get<0>(vector_a) * get<0>(vector_b);
  for (size_t d = 1; d < Index::dim; ++d) {
    get(*dot_product) += vector_a.get(d) * vector_b.get(d);
  }
}

template <typename DataTypeLhs, typename DataTypeRhs, typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(DataTypeLhs() *
                                                             DataTypeRhs()))>
Scalar<DataTypeResult> dot_product(
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>, index_list<Index>>& vector_b) {
  Scalar<DataTypeResult> dot_product(get_size(get<0>(vector_a)));
  ::dot_product(make_not_null(&dot_product), vector_a, vector_b);
  return dot_product;
}
/// @}

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Compute the dot product of a vector and a one form
 *
 * \details
 * Returns \f$A^a B_b \delta_{a}^b\f$ for input vector \f$A^a\f$ and
 * input one form \f$B_b\f$
 * or \f$A_a B^b \delta^a_b\f$ for input one form \f$A_a\f$ and
 * input vector \f$B^b\f$.
 */
template <typename DataTypeLhs, typename DataTypeRhs, typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(DataTypeLhs() *
                                                             DataTypeRhs()))>
void dot_product(
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls) false positive
    const gsl::not_null<Scalar<DataTypeResult>*> dot_product,
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>,
                 index_list<change_index_up_lo<Index>>>& vector_b) {
  get(*dot_product) = get<0>(vector_a) * get<0>(vector_b);
  for (size_t d = 1; d < Index::dim; ++d) {
    get(*dot_product) += vector_a.get(d) * vector_b.get(d);
  }
}

template <typename DataTypeLhs, typename DataTypeRhs, typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(DataTypeLhs() *
                                                             DataTypeRhs()))>
Scalar<DataTypeResult> dot_product(
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>,
                 index_list<change_index_up_lo<Index>>>& vector_b) {
  Scalar<DataTypeResult> dot_product(get_size(get<0>(vector_a)));
  ::dot_product(make_not_null(&dot_product), vector_a, vector_b);
  return dot_product;
}
/// @}

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Compute the dot_product of two vectors or one forms
 *
 * \details
 * Returns \f$g_{ab} A^a B^b\f$, where \f$g_{ab}\f$ is the metric,
 * \f$A^a\f$ is vector_a, and \f$B^b\f$ is vector_b.
 * Or, returns \f$g^{ab} A_a B_b\f$ when given one forms \f$A_a\f$
 * and \f$B_b\f$ with an inverse metric \f$g^{ab}\f$.
 */
template <typename DataTypeLhs, typename DataTypeRhs, typename DataTypeMetric,
          typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(
              DataTypeLhs() * DataTypeRhs() * DataTypeMetric()))>
void dot_product(
    const gsl::not_null<Scalar<DataTypeResult>*> dot_product,
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>, index_list<Index>>& vector_b,
    const Tensor<DataTypeMetric, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) {
  if constexpr (Index::dim == 1) {
    get(*dot_product) = get<0>(vector_a) * get<0>(vector_b) * get<0, 0>(metric);
  } else {
    if (&vector_a == &vector_b) {
      get(*dot_product) =
          get<0>(vector_a) * get<1>(vector_a) * get<0, 1>(metric);
      for (size_t j = 2; j < Index::dim; ++j) {
        get(*dot_product) +=
            vector_a.get(0) * vector_a.get(j) * metric.get(0, j);
      }
      for (size_t i = 1; i < Index::dim; ++i) {
        for (size_t j = i + 1; j < Index::dim; ++j) {
          get(*dot_product) +=
              vector_a.get(i) * vector_a.get(j) * metric.get(i, j);
        }
      }
      get(*dot_product) *= 2.0;

      for (size_t i = 0; i < Index::dim; ++i) {
        get(*dot_product) += square(vector_a.get(i)) * metric.get(i, i);
      }
    } else {
      get(*dot_product) =
          get<0>(vector_a) * get<0>(vector_b) * get<0, 0>(metric);
      for (size_t b = 1; b < Index::dim; ++b) {
        get(*dot_product) +=
            get<0>(vector_a) * vector_b.get(b) * metric.get(0, b);
      }

      for (size_t a = 1; a < Index::dim; ++a) {
        for (size_t b = 0; b < Index::dim; ++b) {
          get(*dot_product) +=
              vector_a.get(a) * vector_b.get(b) * metric.get(a, b);
        }
      }
    }
  }
}

template <typename DataTypeLhs, typename DataTypeRhs, typename DataTypeMetric,
          typename Index,
          typename DataTypeResult = decltype(blaze::evaluate(
              DataTypeLhs() * DataTypeRhs() * DataTypeMetric()))>
Scalar<DataTypeResult> dot_product(
    const Tensor<DataTypeLhs, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataTypeRhs, Symmetry<1>, index_list<Index>>& vector_b,
    const Tensor<DataTypeMetric, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) {
  Scalar<DataTypeResult> dot_product(get_size(get<0>(vector_a)));
  ::dot_product(make_not_null(&dot_product), vector_a, vector_b, metric);
  return dot_product;
}
/// @}
