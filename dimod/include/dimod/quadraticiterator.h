// Copyright 2020 D-Wave Systems Inc.
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

#ifndef DIMOD_QUADRATICITERATOR_H_
#define DIMOD_QUADRATICITERATOR_H_

#include <iterator>
#include <type_traits>
#include <utility>

namespace dimod {
namespace utils {
    template <class V, class B>
    struct QuadraticTerm {
        V const u;
        V const v;
        B const bias;

        QuadraticTerm const* operator->() const { return this; }
    };

    template <class InIT, class OutIT>
    class QuadraticIterator {
      public:
        using in_iterator = InIT;
        using out_iterator = OutIT;

        using bias_type = typename std::remove_cv<typename out_iterator::value_type::second_type>::type;
        using variable_type = typename std::remove_cv<typename out_iterator::value_type::first_type>::type;

        using difference_type = typename out_iterator::difference_type;
        using value_type = QuadraticTerm<variable_type, bias_type>;
        using pointer = value_type const*;
        using reference = value_type;
        using iterator_category = std::input_iterator_tag;

        QuadraticIterator() = default;
        QuadraticIterator(in_iterator in_it, in_iterator in_end, variable_type u = 0)
                : u_(u), in_it_(in_it), in_end_(in_end) {
            if (in_it_ != in_end) {
                out_it_ = out_row_begin();

                // Find the first outvar after in_it in the lower triangle
                if (out_it_ == out_row_end() || u_ < (*this)->v) {
                    ++*this;
                }
            }
        }

        reference operator*() const { return {u_, out_it_->first, out_it_->second}; }
        reference operator->() const { return operator*(); }

        // Incrementing this iterates over the lower triangle of the adjacency lists.
        QuadraticIterator& operator++() {
            ++out_it_;

            // If this is the end of the lower triangle in this row, go to the next nonempty row if one exsits.
            while (out_it_ == out_row_end() || u_ < (*this)->v) {
                if (++in_it_ == in_end_) {
                    return *this;
                }
                ++u_;
                out_it_ = out_row_begin();
            }
            return *this;
        }

        // Decrementing this iterates over the lower triangle of the adjacency lists.
        QuadraticIterator& operator--() {
            // If this is past the last row, go to the end of the last row.
            if (in_it_ == in_end_) {
                --in_it_;
                out_it_ = out_row_end();
            }
            // If this is at the beginning of the row, go to the previous nonempty row in the lower triangle.
            do {
                while (out_it_ == out_row_begin()) {
                    --in_it_;
                    --u_;
                    out_it_ = out_row_end();
                }
                --out_it_;
            } while (u_ < (*this)->v);
            return *this;
        }

        bool operator==(QuadraticIterator const& other) const {
            if (in_it_ == in_end_ || other.in_it_ == other.in_end_) {
                return in_it_ == other.in_it_;
            } else {
                return out_it_ == other.out_it_;
            }
        }

        bool operator!=(QuadraticIterator const& other) const { return !(*this == other); }

        QuadraticIterator operator++(int) {
            QuadraticIterator temp{*this};
            ++*this;
            return temp;
        }

        QuadraticIterator operator--(int) {
            QuadraticIterator temp{*this};
            --*this;
            return temp;
        }

      private:
        out_iterator out_row_begin() const { return in_it_->first.begin(); }
        out_iterator out_row_end() const { return in_it_->first.end(); }

        variable_type u_;
        out_iterator out_it_;
        in_iterator in_it_;
        in_iterator in_end_;
    };

    // Since AdjArrayBQM stores indices into the adjacency lists, a different class is needed.
    template <class V, class B, class N>
    class QuadraticArrayIterator {
      public:
        using in_iterator = typename std::vector<std::pair<N, B>>::const_iterator;
        using out_iterator = typename std::vector<std::pair<V, B>>::const_iterator;

        using bias_type = B;
        using variable_type = V;

        using difference_type = typename out_iterator::difference_type;
        using value_type = QuadraticTerm<variable_type, bias_type>;
        using pointer = value_type const*;
        using reference = value_type;
        using iterator_category = std::input_iterator_tag;

        QuadraticArrayIterator() = default;
        QuadraticArrayIterator(in_iterator in_it, in_iterator in_end, out_iterator out_begin, out_iterator out_end,
                               variable_type u = 0)
                : u_(u), out_begin_(out_begin), out_end_(out_end), in_it_(in_it), in_end_(in_end) {
            if (in_it_ != in_end) {
                out_it_ = out_row_begin();

                // Find the first outvar after in_it in the lower triangle
                if (out_it_ == out_row_end() || u_ < (*this)->v) {
                    ++*this;
                }
            }
        }

        reference operator*() const { return {u_, out_it_->first, out_it_->second}; }
        reference operator->() const { return operator*(); }

        // Incrementing this iterates over the lower triangle of the adjacency lists.
        QuadraticArrayIterator& operator++() {
            ++out_it_;

            // If this is the end of the lower triangle in this row, go to the next nonempty row if one exsits.
            while (out_it_ == out_row_end() || u_ < operator*().v) {
                if (++in_it_ == in_end_) {
                    return *this;
                }
                ++u_;
                out_it_ = out_row_begin();
            }
            return *this;
        }

        // Decrementing this iterates over the lower triangle of the adjacency lists.
        QuadraticArrayIterator& operator--() {
            // If this is past the last row, go to the end of the last row.
            if (in_it_ == in_end_) {
                --in_it_;
                out_it_ = out_row_end();
            }

            // If this is at the beginning of the row, go to the previous nonempty row in the lower triangle.
            do {
                while (out_it_ == out_row_begin()) {
                    --in_it_;
                    --u_;
                    out_it_ = out_row_end();
                }
                --out_it_;
            } while (u_ < operator*().v);
            return *this;
        }

        bool operator==(QuadraticArrayIterator const& other) const {
            if (in_it_ == in_end_ || other.in_it_ == other.in_end_) {
                return in_it_ == other.in_it_;
            } else {
                return out_it_ == other.out_it_;
            }
        }

        bool operator!=(QuadraticArrayIterator const& other) const { return !(*this == other); }

        QuadraticArrayIterator operator++(int) {
            QuadraticArrayIterator temp{*this};
            ++*this;
            return temp;
        }

        QuadraticArrayIterator operator--(int) {
            QuadraticArrayIterator temp{*this};
            --*this;
            return temp;
        }

      private:
        out_iterator out_row_begin() const { return in_it_->first + out_begin_; }
        out_iterator out_row_end() const {
            if (next(in_it_) == in_end_) {
                return out_end_;
            } else {
                return out_begin_ + next(in_it_)->first;
            }
        }

        variable_type u_;
        out_iterator out_it_;
        out_iterator out_begin_;
        out_iterator out_end_;
        in_iterator in_it_;
        in_iterator in_end_;
    };
}  // namespace utils
}  // namespace dimod

#endif  // DIMOD_QUADRATICITERATOR_H_