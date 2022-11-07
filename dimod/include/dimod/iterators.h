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

namespace dimod {
namespace iterators {

template <class Iter, class T = typename std::iterator_traits<Iter>::value_type::element_type>
class ConstraintIterator {
    static_assert(std::is_same<std::random_access_iterator_tag,
                               typename std::iterator_traits<Iter>::iterator_category>::value,
                  "template value Iter must be a random access iterator");
    // Also must be a vector of shared_ptr or unique_ptr, but that's hard to static_assert
    // and will raise a compiler error regardless, albiet a hard to read one.

    // Without a whole lot of work this could be generalized to being a full indirect iterator,
    // but it keeps the boilerplate down to be able to make assumptions about the form.

 public:
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = typename std::iterator_traits<Iter>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    explicit ConstraintIterator(Iter& iterator) : iterator_(iterator) {}

    reference operator*() { return *(*iterator_); }
    reference operator[](difference_type n) { return *(*(iterator_ + n)); }
    pointer operator->() { return (*iterator_).get(); }

    ConstraintIterator& operator++() {
        ++iterator_;
        return *this;
    }
    ConstraintIterator operator++(int) {
        ConstraintIterator tmp(*this);
        ++(*this);
        return tmp;
    }
    ConstraintIterator& operator--() {
        --iterator_;
        return *this;
    }
    ConstraintIterator operator--(int) {
        ConstraintIterator tmp(*this);
        --(*this);
        return tmp;
    }

    ConstraintIterator& operator+=(difference_type n) {
        iterator_ += n;
        return *this;
    }
    ConstraintIterator& operator-=(difference_type n) {
        iterator_ -= n;
        return *this;
    }

    ConstraintIterator operator+(difference_type n) {
        ConstraintIterator tmp(*this);
        tmp += n;
        return tmp;
    }
    ConstraintIterator operator-(difference_type n) {
        ConstraintIterator tmp(*this);
        tmp -= n;
        return tmp;
    }

    friend difference_type operator-(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ - b.iterator_;
    }

    friend bool operator==(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ == b.iterator_;
    }
    friend bool operator!=(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ != b.iterator_;
    }
    friend bool operator<=(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ <= b.iterator_;
    }
    friend bool operator>=(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ >= b.iterator_;
    }
    friend bool operator<(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ < b.iterator_;
    }
    friend bool operator>(const ConstraintIterator& a, const ConstraintIterator& b) {
        return a.iterator_ > b.iterator_;
    }

 private:
    Iter iterator_;
};

template <class Iter>
ConstraintIterator<Iter> make_constraint_iterator(Iter iterator) {
    return ConstraintIterator<Iter>(iterator);
}

template <class Iter>
ConstraintIterator<Iter, const typename std::iterator_traits<Iter>::value_type::element_type>
make_const_constraint_iterator(Iter iterator) {
    return ConstraintIterator<Iter,
                              const typename std::iterator_traits<Iter>::value_type::element_type>(
            iterator);
}

}  // namespace iterators
}  // namespace dimod
