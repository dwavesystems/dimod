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
// on the on the sign of the coefficient. The index of the symmetric edge for
// X_i connecting to X_j' is the index to the edge in the edge list of X_j that
// connects to X_i'. It is needed for the step where we average the residual
// capacities of the edges to get the symmetric residual graph.
// For more details see : Boros, Endre & Hammer, Peter & Tavares, Gabriel.
// (2006). Preprocessing of unconstrained quadratic binary optimization. RUTCOR
// Research Report.
template <typename capacity_t> class ImplicationEdge {
public:
  typedef capacity_t capacity_type;
  ImplicationEdge(int toVertex, capacity_t capacity,
                  capacity_t reverse_capacity)
      : toVertex(toVertex), residual(capacity) {
    assert((!capacity || !reverse_capacity) &&
           "Either capacity or reverse edge capacity must be zero.");
    _encoded_capacity = (!capacity) ? -reverse_capacity : capacity;
  }
  int toVertex;
  int revEdgeIdx;
  int symmEdgeIdx;
  capacity_t residual;

private:
  // The value of capacity is not needed per se to compute a max-flow but is
  // needed for the purpose of verifying it. We use the encoded capacity to
  // store the capacity of an edge, but when it is a residual/reverse edge,
  // instead of saving 0 we save the negative of the original edge, this way we
  // can both verify max-flow and also return the residual capacity of the edge
  // itself and also its reverse/residual without hopping through memory.
  // This is not the best software engineering practice, but is needed to save
  // memory. Current size is 32 byte when long long int is used for capacity
  // and 2 such edges fit in a typical cache line, adding 8 bytes will not allow
  // that to be possible.
  capacity_t _encoded_capacity;

public:
  inline capacity_t getFlow() {
    return ((_encoded_capacity > 0) ? (_encoded_capacity - residual)
                                    : -residual);
  }
  inline capacity_t getCapacity() {
    return ((_encoded_capacity > 0) ? _encoded_capacity : 0);
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

  capacity_t getCurrentFlow();

  vector<vector<ImplicationEdge<capacity_t>>> &getAdjacencyList() {
    return _adjacency_list;
  }

private:
  inline int complement(int v) {
    return (v <= _num_variables) ? (v + _num_variables + 1)
                                 : (v - _num_variables - 1);
  }

  void fillLastOutEdgeReferences(int from, int to);
  void createImplicationNetworkEdges(int from, int to, capacity_t capacity);

  int _num_variables;
  int _num_vertices;
  int _source;
  int _sink;
  vector<vector<ImplicationEdge<capacity_t>>> _adjacency_list;

  // TODO : Verify size estimates are correct, for debugging, remove it.
  vector<int> _size_estimates;
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
  _size_estimates.resize(_num_vertices, 0);

  // The complement function should only be used after setting the above
  // variables.
  assert(_sink == complement(_source));
  assert(_source == complement(_sink));

  int num_linear = posiform.getNumLinear();
  _adjacency_list[_source].reserve(num_linear);
  _adjacency_list[_sink].reserve(num_linear);
  _size_estimates[_source] = num_linear;
  _size_estimates[_sink] = num_linear;

  // There are reverse edges for each edge created in the implication graph.
  // Depending on the sign of the bias, an edge may start from v or v' but
  // reverse edges makes the number of edges coming out of v and v' the same and
  // are to 1 + number of quadratic biases in which v contributes. The + 1 is
  // due to the linear term.
  for (int u = 0; u < _num_variables; u++) {
    int u_complement = complement(u);
    int num_out_edges = posiform.getNumQuadratic(u);
    auto linear = posiform.getLinear(u);
    if (linear)
      num_out_edges++;
    _adjacency_list[u].reserve(num_out_edges);
    _adjacency_list[u_complement].reserve(num_out_edges);
    _size_estimates[u] = num_out_edges;
    _size_estimates[u_complement] = num_out_edges;

    if (linear > 0) {
      createImplicationNetworkEdges(_source, u_complement, linear);
    } else if (linear < 0) {
      createImplicationNetworkEdges(_source, u, -linear);
    }

    auto quadratic_span = posiform.getQuadratic(u);
    auto it = quadratic_span.first;
    auto itEnd = quadratic_span.second;
    for (; it != itEnd; it++) {
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
    for (int j = 0, j_end = _adjacency_list[i].size(); j < j_end; j++) {
      int from_vertex = i;
      int from_vertex_complement = complement(from_vertex);
      int from_vertex_base = std::min(from_vertex, from_vertex_complement);
      int to_vertex = _adjacency_list[i][j].toVertex;
      int to_vertex_complement = complement(to_vertex);
      int to_vertex_base = std::min(to_vertex, to_vertex_complement);
      // We don not want to process the symmetric edges twice, we pick the one
      // that starts from the smaller vertex number when complementation is not
      // taken into account.
      if (to_vertex_base > from_vertex_base) {
        int symmetric_edge_idx = _adjacency_list[i][j].revEdgeIdx;
        capacity_t edge_residual = _adjacency_list[i][j].residual;
        capacity_t symmetric_edge_residual =
            _adjacency_list[to_vertex_complement][symmetric_edge_idx].residual;
        // The paper states that we should average the residuals and assign the
        // average, but to avoid underflow we do not divide by two but to keep
        // the flow valid we multiply the capacities by 2. The doubling of
        // capacity is not needed for the later steps in the algorithm, but will
        // help us verify if the symmetric flow is a valid flow or not.
        capacity_t residual_sum = edge_residual + symmetric_edge_residual;
        _adjacency_list[i][j].residual = residual_sum;
        _adjacency_list[to_vertex_complement][symmetric_edge_idx].residual =
            residual_sum;
        _adjacency_list[i][j].scaleCapacity(2);
        _adjacency_list[to_vertex_complement][symmetric_edge_idx].scaleCapacity(
            2);
      }
    }
  }
}

/*
  template <class capacity_t>
  bool ImplicationNetwork::is_residual_graph_valid() {
    bool symmetric_residual_valid = true;
    std::vector<std::pair<int, int>> problem_edges;
    #pragma omp parallel for
    for (int i = 0, i_end = adjList.size(); i < i_end i++) {
      for (int j = 0, j_end = adjList[i].size(); j < j_end j++) {
        int reverse_edge_idx = adjList[i][j].revEdgeIdx;
        capacity_t edge_residual = adjList[i][j].residual;
        capacity_t reverse_edge_residual =
            adjList[j][reverse_edge_idx].residual;
        capacity_t edge_capacity = adjList[i][j].residual;
        capacity_t reverse_edge_capacity =
            adjList[j][reverse_edge_idx].residual;
        bool capacity_valid =
            edge_capacity ? (!reverse_edge_capacity) : reverse_edge_capacity;
        bool flow_valid = ((edge_residual + reverse_edge_residual) ==
                           (edge_capacity + reverse_edge_capacity));
        if (!flow_valid || !capacity_valid) {
          #pragma omp critical
          {
            symmetric_residual_valid = false;
            problem_edges.push_back({i, j});
          }
        }
      }
    }

    for (int i = 0, i_end = problem_edges.size(); i++) {
      u = problem_edges[i].first;
      v = adjList[u][problem_edges[i].second].toVertex;
      std::cout << "Invalid flow/capacity value. "
                << std::endl;
      std::cout << i << " --> " << adjList[i][j].toVertex << " : "
                << edge_residual << std::endl;
      std::cout << j_complement << " --> " << complement(i) << " : "
                << symmetric_edge_residual << std::endl;
      return false;
    }

    return is_symmetric_residual_valid;
  }
*/
template <class capacity_t> void ImplicationNetwork<capacity_t>::print() {
  std::cout << std::endl;
  std::cout << "Implication Graph Information : " << std::endl;
  std::cout << "Num Variables : " << _num_variables << std::endl;
  std::cout << "Num Vertices : " << _num_vertices << std::endl;
  std::cout << "Source : " << _source << " Sink : " << _sink << std::endl;
  std::cout << std::endl;
  for (int i = 0; i < _adjacency_list.size(); i++) {
    if (_adjacency_list[i].size() != _size_estimates[i]) {
      std::cout << "Inaccurate size estimate for out edges for " << i << " "
                << _size_estimates[i] << std::endl;
    }

    for (int j = 0; j < _adjacency_list[i].size(); j++) {
      auto &node = _adjacency_list[i][j];
      std::cout << "{ " << i << " --> " << node.toVertex << " " << node.residual
                << " ";
      std::cout << node.revEdgeIdx << " " << node.symmEdgeIdx << " } "
                << std::endl;
    }
    std::cout << endl;
    assert(_adjacency_list[i].size() == _size_estimates[i]);
  }
}

template <class capacity_t>
capacity_t ImplicationNetwork<capacity_t>::getCurrentFlow() {
  capacity_t source_outflow = 0;
  for (int i = 0; i < _adjacency_list[_source].size(); i++)
    source_outflow += _adjacency_list[_source][i].getCapacity() -
                      _adjacency_list[_source][i].residual;
  return source_outflow;
}

template <class capacity_t>
void ImplicationNetwork<capacity_t>::fillLastOutEdgeReferences(int from,
                                                               int to) {
  auto &edge = _adjacency_list[from].back();
  edge.revEdgeIdx = _adjacency_list[to].size() - 1;
  int symmetricFrom = complement(to);
  edge.symmEdgeIdx = _adjacency_list[symmetricFrom].size() - 1;
}

// Each term in posiform produces four edges in implication network
// the reverse edges and the symmetric edges.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::createImplicationNetworkEdges(
    int from, int to, capacity_t capacity) {
  int from_complement = complement(from);
  int to_complement = complement(to);
  _adjacency_list[from].emplace_back(
      ImplicationEdge<capacity_t>(to, capacity, 0));
  _adjacency_list[to].emplace_back(
      ImplicationEdge<capacity_t>(from, 0, capacity));
  _adjacency_list[to_complement].emplace_back(
      ImplicationEdge<capacity_t>(from_complement, capacity, 0));
  _adjacency_list[from_complement].emplace_back(
      ImplicationEdge<capacity_t>(to_complement, 0, capacity));
  fillLastOutEdgeReferences(from, to);
  fillLastOutEdgeReferences(to, from);
  fillLastOutEdgeReferences(to_complement, from_complement);
  fillLastOutEdgeReferences(from_complement, to_complement);
}

#endif // IMPLICATION_NETWORK_INCLUDED
