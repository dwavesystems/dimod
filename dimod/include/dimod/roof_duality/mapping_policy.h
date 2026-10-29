// Copyright 2021 D-Wave Systems Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IMPLICATION_NETWORK_MAPPING_POLICY_HPP_INCLUDED
#define IMPLICATION_NETWORK_MAPPING_POLICY_HPP_INCLUDED

// PLEASE DO NOT MAKE AN INTERFACE FOR THE MAPPER CLASS. VIRTUAL DISPATCH WILL
// ADD TO THE ALREADY EXISTING PENALTY OF USING THIS METHODOLOGY. WE CREATED
// THIS CLASS SINCE WE CAN HAVE PERFORMANCE ADVANTAGE WITH ALTERNATE/EVEN_ODD
// MAPPING WHILE IT IS EASIER TO DEBUG WITH SEQUENTIAL MAPPING, BUT IT IS HARD
// TO SHIFT BETWEEN THEM WITHOUT A DEDICATED CLASS.

#define evenOddMapper mapper_t

// These mapper objects control variable to vertex mapping. Please do not
// manually map them even though it may seem easy, since we may change the
// mapping by these functions later. The assertions should be changed if the
// mapping is changed. If there are n variables in the Quobo, the posiform will
// have 2 * ( n + 1 ) variables, each variable with its complement and a root,
// X_0/X_source and its complement. Here we treat variable 0-n-1 as the original
// n Qubo variables and variable n as X_0/X_source.

// In sequential mapping the variables 0-n-1 map to vertices 0-n-1. The source
// variable maps to vertex n. Then the vertices n+1 to 2*n + 1 correspond to the
// complements of vertices 0 - n sequentially. This method of mapping is easy
// for debugging purpose but does not keep the ordering that may be in the qubo
// as regards to the biases which may be sorted based on the connecting
// variables.
class sequentialMapper {
public:
  sequentialMapper(int num_variables) : _num_variables(num_variables) {}

  sequentialMapper() {}

  inline int source() { return _num_variables; }

  inline int sink() { return 2 * _num_variables + 1; }

  inline int num_vertices() { return 2 * (_num_variables + 1); }

  inline int complement(int vertex) {
    assert((vertex <= 2 * _num_variables + 1) && (vertex >= 0));
    return (vertex <= _num_variables) ? (vertex + _num_variables + 1)
                                      : (vertex - _num_variables - 1);
  }

  inline int non_complemented_vertex(int vertex) {
    assert((vertex <= 2 * _num_variables + 1) && (vertex >= 0));
    return (vertex <= _num_variables) ? vertex : (vertex - _num_variables - 1);
  }

  // Should convert only vertices which correspond to non-complemented variables
  // and should not convert the source vertex.
  inline int non_complemented_vertex_to_variable(int vertex) {
    assert((vertex < _num_variables) && (vertex >= 0));
    return vertex;
  }

  inline int variable_to_vertex(int variable) {
    assert((variable < _num_variables) && (variable >= 0));
    return variable;
  }

  inline bool complement_maintains_order() { return false; }

private:
  int _num_variables;
};

// The evenOddMapper maps variable x to vertex 2 * x and its complement is 2 * x
// + 1. Same as before we consider variable n to be the source.
class evenOddMapper {
public:
  evenOddMapper(int num_variables) : _num_variables(num_variables) {}

  evenOddMapper() {}

  inline int source() { return 2 * _num_variables; }

  inline int sink() { return 2 * _num_variables + 1; }

  inline int num_vertices() { return 2 * (_num_variables + 1); }

  inline int complement(int vertex) {
    assert((vertex <= 2 * _num_variables + 1) && (vertex >= 0));
    return (vertex % 2) ? (vertex - 1) : (vertex + 1);
  }

  inline int non_complemented_vertex(int vertex) {
    assert((vertex <= 2 * _num_variables + 1) && (vertex >= 0));
    return (vertex % 2) ? (vertex - 1) : (vertex);
  }

  // Should convert only vertices which correspond to non-complemented variables
  // and should not convert the source vertex.
  inline int non_complemented_vertex_to_variable(int vertex) {
    assert((vertex <= 2 * _num_variables - 1) && (vertex >= 0) &&
           !(vertex % 2));
    return vertex / 2;
  }

  inline int variable_to_vertex(int variable) {
    assert((variable < _num_variables) && (variable >= 0));
    return variable * 2;
  }

  inline bool complement_maintains_order() { return true; }

private:
  int _num_variables;
};

#endif
