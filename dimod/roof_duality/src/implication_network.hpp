/**
# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#
================================================================================================
*/
#ifndef IMPLICATION_NETWORK_HPP_INCLUDED
#define IMPLICATION_NETWORK_HPP_INCLUDED

// Edge type for implication network. An implication network is formed from a
// posiform. If there is a term Coeff * X_i * X_j, we will have two edges in the
// network one X_i to X_j' and another X_j to X_i', the directions will depend
// on the the sign of the coefficient. The index of the symmetric edge for
// X_i connecting to X_j' is the index to the edge in the edge list of X_j that
// connects to X_i'. It is needed for the step where we average the residual
// capacities of the edges to get the symmetric residual graph.
// For more details see : Boros, Endre & Hammer, Peter & Tavares, Gabriel.
// (2006). Preprocessing of unconstrained quadratic binary optimization. RUTCOR
// Research Report.
template <typename capacity_t> class ImplicationEdge {
public:
  typedef capacity_t capacity_type;

  ImplicationEdge(int from_vertex, int to_vertex, capacity_t capacity,
                  capacity_t reverse_capacity)
      : from_vertex(from_vertex), to_vertex(to_vertex), residual(capacity) {
    assert((!capacity || !reverse_capacity) &&
           "Either capacity or reverse edge capacity must be zero.");
    _encoded_capacity = (!capacity) ? -reverse_capacity : capacity;
  }

  void print() {
    std::cout << std::endl;
    std::cout << from_vertex << " --> " << to_vertex << std::endl;
    std::cout << "Capacity : " << getCapacity() << std::endl;
    std::cout << "Residual : " << residual << std::endl;
    std::cout << "Reverse Edge Capaciy : " << getReverseEdgeCapacity()
              << std::endl;
    std::cout << "Reverse Edge Residual : " << getReverseEdgeResidual()
              << std::endl;
  }

  // The from_vertex is redundant, since we use an adjacency list, but we are
  // not wasting any space as of now, since the compiler uses the same amount of
  // storage for padding when residual is of a type that takes 8 bytes and in
  // our implmementation we use long long int (8 byte type).
  int from_vertex;
  int to_vertex;
  int reverse_edge_index;
  int symmetric_edge_index;
  capacity_t residual;

private:
  // The value of capacity is not needed per se to compute a max-flow but is
  // needed for the purpose of verifying it. We use the encoded capacity to
  // store the capacity of an edge, but when it is a residual/reverse edge,
  // instead of saving 0 we save the negative of the original edge's capacity,
  // this way we can both verify max-flow and also return the residual capacity
  // of the edge itself and also its reverse/residual without hopping through
  // memory. This is not the best software engineering practice, but is needed
  // to save memory. Current size is 32 byte when long long int is used for
  // capacity and 2 such edges fit in a typical cache line, adding 8 bytes will
  // not allow that to be possible.
  capacity_t _encoded_capacity;

public:
  inline capacity_t getFlow() {
    return ((_encoded_capacity > 0) ? (_encoded_capacity - residual)
                                    : -residual);
  }
  inline capacity_t getCapacity() {
    return ((_encoded_capacity > 0) ? _encoded_capacity : 0);
  }

  inline capacity_t getReverseEdgeCapacity() {
    return ((_encoded_capacity > 0) ? 0 : -_encoded_capacity);
  }

  inline capacity_t getReverseEdgeResidual() {
    return ((_encoded_capacity > 0) ? (_encoded_capacity - residual)
                                    : (-_encoded_capacity - residual));
  }

  // Needed for the purpose of making residual network symmetric.
  void scaleCapacity(int scale) { _encoded_capacity *= scale; }
};

// The implication graph used in the paper Boros, Endre & Hammer, Peter &
// Tavares, Gabriel. (2006). Preprocessing of unconstrained quadratic binary
// optimization. RUTCOR Research Report. If there are n variables in the Quobo,
// the posiform will have 2 * ( n + 1 ) variables, each variable with its
// complement and a root, X_0 and its complement. Here we treat variable 0-n-1
// as the original n Qubo variables and variable n as X_0. Variable n+1 to 2n+1
// are their complements, thus variable v has its complement as (v + n + 1).
template <class capacity_t> class ImplicationNetwork {
public:
  template <class PosiformInfo> ImplicationNetwork(PosiformInfo &posiform);

  void makeResidualSymmetric();

  void print();

  int getSource() { return _source; }

  int getSink() { return _sink; }

  std::vector<std::vector<ImplicationEdge<capacity_t>>> &getAdjacencyList() {
    return _adjacency_list;
  }

private:
  inline int complement(int v) {
    return (v <= _num_variables) ? (v + _num_variables + 1)
                                 : (v - _num_variables - 1);
  }

  void fillLastOutEdgeReferences(int from_vertex, int to_vertex);
  void createImplicationNetworkEdges(int from_vertex, int to_vertex,
                                     capacity_t capacity);

  int _num_variables;
  int _num_vertices;
  int _source;
  int _sink;
  std::vector<std::vector<ImplicationEdge<capacity_t>>> _adjacency_list;
};

template <class capacity_t>
template <class PosiformInfo>
ImplicationNetwork<capacity_t>::ImplicationNetwork(PosiformInfo &posiform) {
  assert(std::is_integral(capacity_t) && std::is_signed(capacity_t) &&
         "Implication Network must have signed, integral type coefficients");
  assert((std::numeric_limits<capacity_t>::max() >=
          std::numeric_limits<Posiform::coefficient_type> max()) &&
         "Implication Network must have capacity type with larger maximum "
         "value than the type of coefficients in source posiform.");
  _num_variables = posiform.getNumVariables();
  _num_vertices = 2 * _num_variables + 2;
  _source = _num_variables;
  _sink = 2 * _num_variables + 1;
  _adjacency_list.resize(2 * _num_variables + 2);

  // The complement function should only be used after setting the above
  // variables.
  assert(_sink == complement(_source));
  assert(_source == complement(_sink));

  int num_linear = posiform.getNumLinear();
  _adjacency_list[_source].reserve(num_linear);
  _adjacency_list[_sink].reserve(num_linear);

  // For efficiency we preallocate the vectors first.
  // There are reverse edges for each edge created in the implication graph.
  // Depending on the sign of the bias, an edge may start from v or v' but
  // reverse edges makes the number of edges coming out of v and v' the same and
  // are equal to 1/0 + number of quadratic biases in which v contributes. The +
  // 1 is due to the linear term when it is present.
  for (int u = 0; u < _num_variables; u++) {
    int u_complement = complement(u);
    int num_out_edges = posiform.getNumQuadratic(u);
    auto linear = posiform.getLinear(u);
    if (linear) {
      num_out_edges++;
    }
    _adjacency_list[u].reserve(num_out_edges);
    _adjacency_list[u_complement].reserve(num_out_edges);
  }

  for (int u = 0; u < _num_variables; u++) {
    int u_complement = complement(u);
    auto linear = posiform.getLinear(u);
    if (linear > 0) {
      createImplicationNetworkEdges(_source, u_complement, linear);
    } else if (linear < 0) {
      createImplicationNetworkEdges(_source, u, -linear);
    }
    auto quadratic_span = posiform.getQuadratic(u);
    auto it = quadratic_span.first;
    auto it_end = quadratic_span.second;
    for (; it != it_end; it++) {
      // The quadratic iterators, in the posiform, belong  to the original
      // bqm, thus the variables, must be mapped to the posiform variables,
      // and the biases should be ideally converted to the same type the
      // posiform represens them in.
      auto coefficient = posiform.convertToPosiformCoefficient(it->second);
      int v = posiform.getMappedVariable(it->first);
      if (coefficient > 0) {
        createImplicationNetworkEdges(u, complement(v), coefficient);
      } else if (coefficient < 0) {
        createImplicationNetworkEdges(u, v, -coefficient);
      }
    }
  }
}

// Make the residual network symmetric, by summing the residual capacities and
// original capacities of the implicaiton network. Here we multiply the
// capacities by 2, since the symmetric edges have same capacity.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::makeResidualSymmetric() {
  for (int i = 0, i_end = _adjacency_list.size(); i < i_end; i++) {
    auto eit = _adjacency_list[i].begin();
    auto eit_end = _adjacency_list[i].end();
    for (; eit != eit_end; eit++) {
      int from_vertex = i;
      int from_vertex_complement = complement(from_vertex);
      int from_vertex_base = std::min(from_vertex, from_vertex_complement);
      int to_vertex = eit->to_vertex;
      int to_vertex_complement = complement(to_vertex);
      int to_vertex_base = std::min(to_vertex, to_vertex_complement);
      // We don not want to process the symmetric edges twice, we pick the one
      // that starts from the smaller vertex number when complementation is not
      // taken into account.
      if (to_vertex_base > from_vertex_base) {
        int symmetric_edge_idx = eit->reverse_edge_index;
        capacity_t edge_residual = eit->residual;
        capacity_t symmetric_edge_residual =
            _adjacency_list[to_vertex_complement][symmetric_edge_idx].residual;
        // The paper states that we should average the residuals and assign the
        // average, but to avoid underflow we do not divide by two but to keep
        // the flow valid we multiply the capacities by 2. The doubling of
        // capacity is not needed for the later steps in the algorithm, but will
        // help us verify if the symmetric flow is a valid flow or not.
        capacity_t residual_sum = edge_residual + symmetric_edge_residual;
        eit->residual = residual_sum;
        _adjacency_list[to_vertex_complement][symmetric_edge_idx].residual =
            residual_sum;
        eit->scaleCapacity(2);
        _adjacency_list[to_vertex_complement][symmetric_edge_idx].scaleCapacity(
            2);
      }
    }
  }
}

template <class capacity_t> void ImplicationNetwork<capacity_t>::print() {
  std::cout << std::endl;
  std::cout << "Implication Graph Information : " << std::endl;
  std::cout << "Num Variables : " << _num_variables << std::endl;
  std::cout << "Num Vertices : " << _num_vertices << std::endl;
  std::cout << "Source : " << _source << " Sink : " << _sink << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < _adjacency_list.size(); i++) {
    for (int j = 0; j < _adjacency_list[i].size(); j++) {
      auto &node = _adjacency_list[i][j];
      std::cout << "{ " << i << " --> " << node.to_vertex << " "
                << node.residual << " ";
      std::cout << node.reverse_edge_index << " " << node.symmetric_edge_index
                << " } " << std::endl;
    }
    std::cout << endl;
  }
}

template <class capacity_t>
void ImplicationNetwork<capacity_t>::fillLastOutEdgeReferences(int from_vertex,
                                                               int to_vertex) {
  auto &edge = _adjacency_list[from_vertex].back();
  edge.reverse_edge_index = _adjacency_list[to_vertex].size() - 1;
  int symmetric_from_vertex = complement(to_vertex);
  edge.symmetric_edge_index = _adjacency_list[symmetric_from_vertex].size() - 1;
}

// Each term in posiform produces four edges in implication network
// the reverse edges and the symmetric edges.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::createImplicationNetworkEdges(
    int from_vertex, int to_vertex, capacity_t capacity) {
  int from_vertex_complement = complement(from_vertex);
  int to_vertex_complement = complement(to_vertex);
  _adjacency_list[from_vertex].emplace_back(
      ImplicationEdge<capacity_t>(from_vertex, to_vertex, capacity, 0));
  _adjacency_list[to_vertex].emplace_back(
      ImplicationEdge<capacity_t>(to_vertex, from_vertex, 0, capacity));
  _adjacency_list[to_vertex_complement].emplace_back(
      ImplicationEdge<capacity_t>(to_vertex_complement, from_vertex_complement,
                                  capacity, 0));
  _adjacency_list[from_vertex_complement].emplace_back(
      ImplicationEdge<capacity_t>(from_vertex_complement, to_vertex_complement,
                                  0, capacity));
  fillLastOutEdgeReferences(from_vertex, to_vertex);
  fillLastOutEdgeReferences(to_vertex, from_vertex);
  fillLastOutEdgeReferences(to_vertex_complement, from_vertex_complement);
  fillLastOutEdgeReferences(from_vertex_complement, to_vertex_complement);
}

#endif // IMPLICATION_NETWORK_INCLUDED
