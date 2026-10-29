/**
# MIT License
#
# Copyright (c) 2019 Jan Groschaft
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
*/

#ifndef HELPER_DATA_STRUCTURES_HPP_INCLUDED
#define HELPER_DATA_STRUCTURES_HPP_INCLUDED

#include <vector>

// Queue with std::vector as internal container.
template <typename T> class vector_based_queue {
  std::vector<T> _data;
  std::size_t _back{0};
  std::size_t _front{0};

public:
  vector_based_queue() = default;

  explicit vector_based_queue(std::size_t size) {
    _front = 0;
    _back = 0;
    _data.resize(size);
  }

  void push(T val) noexcept { _data[_back++] = val; }

  T pop() noexcept { return _data[_front++]; }

  bool empty() const noexcept { return _front == _back; }

  void reset() noexcept { _front = _back = 0; }

  int size() noexcept { return _back - _front; }
};

// Stack with std::vector as internal container.
template <typename T> class vector_based_stack {
  std::vector<T> _data;
  std::size_t _size{0};

public:
  vector_based_stack() = default;

  explicit vector_based_stack(std::size_t size) {
    _size = 0;
    _data.resize(size);
  }

  void push(T val) noexcept { _data[_size++] = val; }

  T pop() noexcept { return _data[--_size]; }

  T top() noexcept { return _data[_size - 1]; }

  bool empty() const noexcept { return _size == 0; }

  void reset() noexcept { _size = 0; }

  int size() noexcept { return _size; }
};

// Linked list that uses preallocated nodes.
template <typename node> class preallocated_linked_list {
  node _head = node{}, _tail = node{};
  std::size_t _size{0};

public:
  preallocated_linked_list() {
    _head.next = &_tail;
    _tail.prev = &_head;
    _size = 0;
  }

  node *pop() noexcept {
    auto *ret = _head.next;
    _head.next = _head.next->next;
    _head.next->prev = &_head;
    --_size;
    return ret;
  }

  void push(node *n) noexcept {
    n->next = &_tail;
    n->prev = _tail.prev;
    _tail.prev->next = n;
    _tail.prev = n;
    ++_size;
  }

  void push_front(node *n) noexcept {
    n->next = _head.next;
    n->prev = &_head;
    _head.next->prev = n;
    _head.next = n;
    ++_size;
  }

  void erase(node *n) noexcept {
    n->prev->next = n->next;
    n->next->prev = n->prev;
    --_size;
  }

  node *front() const noexcept { return _head.next; }

  node *back() const noexcept { return _tail.prev; }

  bool empty() const noexcept { return _size == 0; }

  void clear() noexcept {
    _head.next = &_tail;
    _tail.prev = &_head;
    _size = 0;
  }

  std::size_t size() const noexcept { return _size; }

  void append_list(preallocated_linked_list &other) noexcept {
    if (other.empty())
      return;
    auto other_head = other.front();
    auto other_tail = other.back();
    this->back()->next = other_head;
    other_head->prev = this->back();
    _tail.prev = other_tail;
    other_tail->next = &_tail;
    _size += other.size();
    other.clear();
  }
};

#endif
