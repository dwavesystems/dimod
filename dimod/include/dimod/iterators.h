// Copyright 2021 D-Wave Systems Inc.
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

#include <utility>

namespace dimod {

template <class Bias, class Index>
class QuadraticModelBase;

template <class Bias, class Index>
class ConstQuadraticIterator {
 public:
    using bias_type = Bias;
    using index_type = Index;
    using quadratic_model_type = QuadraticModelBase<Bias, Index>;

    struct value_type {
        const index_type u;
        const index_type v;
        const bias_type& bias;
    };
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    // empty constructor is needed for Cython
    ConstQuadraticIterator() : qm_ptr_(nullptr), v_(0), i_(0), term_ptr_(nullptr) {}

    // for the copy constructor, we want to copy the pointer to the quadratic
    // model, but not to the term_ptr_, since that's managed by each iterator
    // separately
    ConstQuadraticIterator(const ConstQuadraticIterator<Bias, Index>& other)
            : qm_ptr_(other.qm_ptr_), v_(other.v_), i_(other.i_), term_ptr_(nullptr) {
        advance();
    }

    // for the move constructor, we want to move everything
    ConstQuadraticIterator(ConstQuadraticIterator<Bias, Index>&& other) noexcept
            : qm_ptr_(nullptr), v_(0), i_(0), term_ptr_(nullptr) {
        swap(other);
    }

    ConstQuadraticIterator(const quadratic_model_type* qm, index_type v)
            : qm_ptr_(qm), v_(v), i_(0), term_ptr_(nullptr) {
        advance();
    }

    ~ConstQuadraticIterator() { delete term_ptr_; }

    ConstQuadraticIterator<Bias, Index>& operator=(
            ConstQuadraticIterator<Bias, Index> const& other) {
        if (this != &other) {
            qm_ptr_ = other.qm_ptr_;
            v_ = other.v_;
            i_ = other.i_;
            advance();  // handles term_ptr_
        }
        return *this;
    }

    ConstQuadraticIterator<Bias, Index>& operator=(
            ConstQuadraticIterator<Bias, Index>&& other) noexcept {
        swap(other);
        return *this;
    }

    const reference operator*() const { return *term_ptr_; }

    const pointer operator->() const { return term_ptr_; }

    ConstQuadraticIterator<Bias, Index>& operator++() {
        ++i_;
        advance();
        return *this;
    }

    ConstQuadraticIterator<Bias, Index> operator++(int) {
        ConstQuadraticIterator<Bias, Index> r(*this);
        ++i_;
        advance();
        return r;
    }

    bool operator==(ConstQuadraticIterator<Bias, Index> const& r) const {
        return v_ == r.v_ && i_ == r.i_;
    }

    bool operator!=(ConstQuadraticIterator<Bias, Index> const& r) const {
        return v_ != r.v_ || i_ != r.i_;
    }

 private:
    const quadratic_model_type* qm_ptr_;
    index_type v_;  // current neighborhood
    index_type i_;  // index in the neighborhood

    value_type* term_ptr_;  // owned by this iterator

    // clear the current term and advance from the current position
    // (inclusive) we traverse the lower triangle (including self-loops)
    void advance() {
        if (term_ptr_) {
            delete term_ptr_;
            term_ptr_ = nullptr;
        }

        while (static_cast<size_t>(v_) < qm_ptr_->num_variables()) {
            auto nit = qm_ptr_->adj_[v_].cbegin() + i_;
            if (nit < qm_ptr_->adj_[v_].cend() && nit->first <= v_) {
                term_ptr_ = new value_type{v_, nit->first, nit->second};
                return;
            }
            ++v_;
            i_ = 0;
        }
    }

    void swap(ConstQuadraticIterator<Bias, Index>& other) noexcept {
        std::swap(qm_ptr_, other.qm_ptr_);
        std::swap(v_, other.v_);
        std::swap(i_, other.i_);
        std::swap(term_ptr_, other.term_ptr_);
    }
};

}  // namespace dimod
