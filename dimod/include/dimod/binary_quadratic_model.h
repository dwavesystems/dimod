// Copyright 2022 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include "dimod/abc.h"
#include "dimod/vartypes.h"

namespace dimod {

/// A binary quadratic model (BQM) is a quadratic polynomial over binary-valued variables.
template <class Bias, class Index = int>
class BinaryQuadraticModel : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// Type of the base class.
    using base_type = abc::QuadraticModelBase<Bias, Index>;

    /// First template parameter (`Bias`).
    using bias_type = Bias;

    /// Second template parameter (`Index`).
    using index_type = Index;

    /// Unsigned integer that can represent non-negative values.
    using size_type = typename base_type::size_type;

    /// Empty constructor. The variable type defaults to `Vartype::BINARY`.
    BinaryQuadraticModel();

    /// Create a BQM of the given `vartype`.
    explicit BinaryQuadraticModel(Vartype vartype);

    /// Create a BQM with `n` variables of the given `vartype`.
    BinaryQuadraticModel(index_type n, Vartype vartype);

    /**
     * Create a BQM from a dense matrix.
     *
     * `dense` must be an array of length `num_variables^2`.
     *
     * Values on the diagonal are treated differently depending on the variable
     * type.
     * If the BQM is SPIN-valued, values on the diagonal are
     * added to the offset.
     * If the BQM is BINARY-valued, values on the diagonal are added
     * as linear biases.
     *
     */
    template <class T>
    BinaryQuadraticModel(const T dense[], index_type num_variables, Vartype vartype);

    using base_type::add_quadratic;

    /**
     * Construct a BQM from COO-formated iterators.
     *
     * A sparse BQM encoded in [COOrdinate] format is specified by three
     * arrays of (row, column, value).
     *
     * [COOrdinate]: https://w.wiki/n$L
     *
     * `row_iterator` must be a random access iterator pointing to the
     * beginning of the row data. `col_iterator` must be a random access
     * iterator pointing to the beginning of the column data. `bias_iterator`
     * must be a random access iterator pointing to the beginning of the bias
     * data. `length` must be the number of (row, column, bias) entries.
     */
    template <class ItRow, class ItCol, class ItBias>
    void add_quadratic(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                       index_type length);

    /// Add one (disconnected) variable to the BQM and return its index.
    index_type add_variable();

    /// Change the variable type of the BQM.
    void change_vartype(Vartype vartype);

    bias_type lower_bound() const;

    /// Return the lower bound on variable ``v``.
    bias_type lower_bound(index_type v) const;

    // Resize the model to contain `n` variables.
    void resize(index_type n);

    bias_type upper_bound() const;

    /// Return the upper bound on variable ``v``.
    bias_type upper_bound(index_type v) const;

    Vartype vartype() const;

    /// Return the variable type of variable ``v``.
    Vartype vartype(index_type v) const;

 private:
    // The vartype of the BQM
    Vartype vartype_;
};

template <class bias_type, class index_type>
BinaryQuadraticModel<bias_type, index_type>::BinaryQuadraticModel()
        : BinaryQuadraticModel(Vartype::BINARY) {}

template <class bias_type, class index_type>
BinaryQuadraticModel<bias_type, index_type>::BinaryQuadraticModel(Vartype vartype)
        : base_type(), vartype_(vartype) {}

template <class bias_type, class index_type>
BinaryQuadraticModel<bias_type, index_type>::BinaryQuadraticModel(index_type n, Vartype vartype)
        : base_type(n), vartype_(vartype) {}

template <class bias_type, class index_type>
template <class T>
BinaryQuadraticModel<bias_type, index_type>::BinaryQuadraticModel(const T dense[],
                                                                  index_type num_variables,
                                                                  Vartype vartype)
        : BinaryQuadraticModel(num_variables, vartype) {
    this->add_quadratic_from_dense(dense, num_variables);
}

template <class bias_type, class index_type>
template <class ItRow, class ItCol, class ItBias>
void BinaryQuadraticModel<bias_type, index_type>::add_quadratic(ItRow row_iterator,
                                                                ItCol col_iterator,
                                                                ItBias bias_iterator,
                                                                index_type length) {
    // we can resize ourself because we know the vartype
    if (length > 0) {
        index_type max_label = std::max(*std::max_element(row_iterator, row_iterator + length),
                                        *std::max_element(col_iterator, col_iterator + length));
        if (max_label >= 0 && static_cast<size_type>(max_label) >= this->num_variables()) {
            this->resize(max_label + 1);
        }
    }

    base_type::add_quadratic(row_iterator, col_iterator, bias_iterator, length);
}

template <class bias_type, class index_type>
index_type BinaryQuadraticModel<bias_type, index_type>::add_variable() {
    return base_type::add_variable();
}

template <class bias_type, class index_type>
void BinaryQuadraticModel<bias_type, index_type>::change_vartype(Vartype vartype) {
    if (vartype_ == vartype) {
        return;
    } else if (vartype == Vartype::SPIN) {
        // binary to spin
        base_type::substitute_variables(.5, .5);
    } else if (vartype == Vartype::BINARY) {
        // spin to binary
        base_type::substitute_variables(2, -1);
    } else {
        throw std::logic_error("unsupported vartype");
    }
    vartype_ = vartype;
}

template <class bias_type, class index_type>
bias_type BinaryQuadraticModel<bias_type, index_type>::lower_bound() const {
    return vartype_info<bias_type>::min(this->vartype_);
}

template <class bias_type, class index_type>
bias_type BinaryQuadraticModel<bias_type, index_type>::lower_bound(index_type v) const {
    return vartype_info<bias_type>::min(this->vartype_);
}

template <class bias_type, class index_type>
void BinaryQuadraticModel<bias_type, index_type>::resize(index_type n) {
    base_type::resize(n);
}

template <class bias_type, class index_type>
bias_type BinaryQuadraticModel<bias_type, index_type>::upper_bound() const {
    return vartype_info<bias_type>::max(this->vartype_);
}

template <class bias_type, class index_type>
bias_type BinaryQuadraticModel<bias_type, index_type>::upper_bound(index_type v) const {
    return vartype_info<bias_type>::max(this->vartype_);
}

template <class bias_type, class index_type>
Vartype BinaryQuadraticModel<bias_type, index_type>::vartype() const {
    return this->vartype_;
}

template <class bias_type, class index_type>
Vartype BinaryQuadraticModel<bias_type, index_type>::vartype(index_type v) const {
    return this->vartype_;
}

}  // namespace dimod
