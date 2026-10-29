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

#ifndef IMPLICATION_NETWORK_HPP_INCLUDED
#define IMPLICATION_NETWORK_HPP_INCLUDED

#include <assert.h>
#include "helper_graph_algorithms.hpp"
#include "mapping_policy.hpp"
#include "push_relabel.hpp"

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
                  capacity_t reverse_capacity, int reverse_edge_index,
                  int symmetric_edge_index)
      : from_vertex(from_vertex), to_vertex(to_vertex),
        reverse_edge_index(reverse_edge_index),
        symmetric_edge_index(symmetric_edge_index), residual(capacity) {
    assert((!capacity || !reverse_capacity) &&
           "Either capacity or reverse edge capacity must be zero.");
    _encoded_capacity = (!capacity) ? -reverse_capacity : capacity;
  }

  void print() {
    std::cout << std::endl;
    std::cout << from_vertex << " --> " << to_vertex << std::endl;
    std::cout << "Capacity : " << getCapacity() << std::endl;
    std::cout << "Residual : " << residual << std::endl;
    std::cout << "Reverse Edge Capacity : " << getReverseEdgeCapacity()
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

// Needed for binary search when edges are in sorted order.
class ImplicationEdgeComparator {
public:
  template <class capacity_t>
  bool operator()(const ImplicationEdge<capacity_t> &a, const int &vertex) {
    return a.to_vertex < vertex;
  }
};

// Structure for holding various information regarding strongly connected
// components of a implication network.
class stronglyConnectedComponentsInfo {
public:
  stronglyConnectedComponentsInfo(int num_components,
                                  std::vector<int> vertex_to_component_map,
                                  mapper_t &mapper)
      : num_vertices(mapper.num_vertices()),num_components(num_components), 
        vertex_to_component_map(vertex_to_component_map) {
    components.resize(num_components);
    std::vector<int> component_sizes(num_components, 0);
    source_component = vertex_to_component_map[mapper.source()];
    sink_component = vertex_to_component_map[mapper.sink()];

    for (int vertex = 0; vertex < num_vertices; vertex++) {
      component_sizes[vertex_to_component_map[vertex]]++;
    }

    for (int component = 0; component < num_components; component++) {
      components.reserve(component_sizes[component]);
    }

    for (int vertex = 0; vertex < num_vertices; vertex++) {
      components[vertex_to_component_map[vertex]].push_back(vertex);
    }

    complement_map.resize(num_components);

    for (int component = 0; component < num_components; component++) {
      assert(components[component].size() &&
             "Each strongly connected component must have one element.");
      // According to the paper each strongly connected component of the
      // residual graph should have either a set of vertices and its complements
      // in it or there will be another component with the complement vertices.
      // So we can decide the complement of a strongly connected component by
      // using only one vertex.
      complement_map[component] =
          vertex_to_component_map[mapper.complement(components[component][0])];
    }

    // Since the assumption that a strongly connected component should self
    // contain the complements of its vertices, or there will be a complement of
    // the component is a very strong one, we keep a check for that.
    for (int vertex = 0; vertex < num_vertices; vertex++) {
      if (complement_map[vertex_to_component_map[vertex]] !=
          vertex_to_component_map[mapper.complement(vertex)]) {
        std::cout
            << "The assumption that each strongly connected component in the "
               "residual graph containing edges with positive capacities, must "
               "contain vertices and their complements, or there will be "
               "another "
               "component with exactly the complementary vertices, did not "
               "hold."
            << std::endl;
        exit(1);
      }
    }
  }

public:
  int num_vertices;
  int num_components;
  int source_component;
  int sink_component;
  // Component to its complement component.
  std::vector<int> complement_map;
  std::vector<int> vertex_to_component_map;
  std::vector<std::vector<int>> components;
};

// The implication graph used in the paper Boros, Endre & Hammer, Peter &
// Tavares, Gabriel. (2006). Preprocessing of unconstrained quadratic binary
// optimization. RUTCOR Research Report. See the class mappingPolicy to
// understand the mapping between posiform variables and implication network
// vertices.
template <class capacity_t> class ImplicationNetwork {

public:
  template <class PosiformInfo> ImplicationNetwork(PosiformInfo &posiform);

  int getSource() { return _source; }

  int getSink() { return _sink; }

  std::vector<std::vector<ImplicationEdge<capacity_t>>> &getAdjacencyList() {
    checkAdjacencyListValidity();
    return _adjacency_list;
  }

  capacity_t fixVariables(std::vector<std::pair<int, int>> &fixed_variables,
                    bool only_trivially_strong = false);

  void print();

private:
  // We may free the memory of adjacency list after crucial operations have
  // completed. Thus public functions may need to check the validity of the
  // adjacency list before operating.
  void checkAdjacencyListValidity() {
    if (!_adjacency_list_valid) {
      std::cout << std::endl;
      std::cout << "Function requiring adjacency list of implication network "
                   "called after releasing its memory."
                << std::endl;
      exit(1);
    }
  }

  void makeResidualSymmetric();

  void extractResidualNetworkWithoutSourceInSinkOut(
      std::vector<std::vector<int>> &adjacency_list_residual,
      bool free_original_adjacency_list = false);

  capacity_t fixTriviallyStrongVariables(
      std::vector<std::pair<int, int>> &fixed_variables);

  void fixStronglyConnectedComponentVariables(
      int component, stronglyConnectedComponentsInfo &scc_info,
      std::vector<std::vector<int>> &adjacency_list_components_transposed,
      std::vector<int> &out_degrees,
      std::vector<std::pair<int, int>> &fixed_variables,
      vector_based_queue<int> &component_queue, bool enqueue);

  capacity_t
  fixStrongAndWeakVariables(std::vector<std::pair<int, int>> &fixed_variables);

  void createImplicationNetworkEdges(int from_vertex, int to_vertex,
                                     capacity_t capacity);

  int _num_variables;
  int _num_vertices;
  int _source;
  int _sink;
  bool _adjacency_list_valid;
  mapper_t _mapper;
  std::vector<std::vector<ImplicationEdge<capacity_t>>> _adjacency_list;
};

template <class capacity_t>
template <class PosiformInfo>
ImplicationNetwork<capacity_t>::ImplicationNetwork(PosiformInfo &posiform) {
  assert(std::is_integral<capacity_t>::value &&
         std::is_signed<capacity_t>::value &&
         "Implication Network must have signed, integral type coefficients");
  assert(
      (std::numeric_limits<capacity_t>::max() >=
       std::numeric_limits<typename PosiformInfo::coefficient_type>::max()) &&
      "Implication Network must have capacity type with larger maximum "
      "value than the type of coefficients in source posiform.");
  _num_variables = posiform.getNumVariables();
  _mapper = mapper_t(_num_variables);
  _num_vertices = _mapper.num_vertices();
  _source = _mapper.source();
  _sink = _mapper.sink();
  _adjacency_list.resize(2 * _num_variables + 2);

  // For efficiency we preallocate the vectors first.
  int num_linear = posiform.getNumLinear();
  _adjacency_list[_source].reserve(num_linear);
  _adjacency_list[_sink].reserve(num_linear);

  // There are reverse edges for each edge created in the implication graph.
  // Depending on the sign of the bias, an edge may start from v or v' but
  // reverse edges makes the number of edges coming out of v and v' the same and
  // are equal to 1/0 + number of quadratic biases in which v contributes. The +
  // 1 is due to the linear term when it is present.
  for (int variable = 0; variable < _num_variables; variable++) {
    int from_vertex = _mapper.variable_to_vertex(variable);
    int from_vertex_complement = _mapper.complement(from_vertex);
    int num_out_edges = posiform.getNumQuadratic(variable);
    auto linear = posiform.getLinear(variable);
    if (linear) {
      num_out_edges++;
    }
    _adjacency_list[from_vertex].reserve(num_out_edges);
    _adjacency_list[from_vertex_complement].reserve(num_out_edges);
  }

  for (int variable = 0; variable < _num_variables; variable++) {
    int from_vertex = _mapper.variable_to_vertex(variable);
    // int from_vertex_complement = _mapper.complement(from_vertex);
    auto quadratic_span = posiform.getQuadratic(variable);
    auto it = quadratic_span.first;
    auto it_end = quadratic_span.second;
    for (; it != it_end; it++) {
      // The quadratic iterators, in the posiform, belong  to the original
      // bqm, thus the variables, must be mapped to the posiform variables,
      // and the biases should be ideally converted to the same type the
      // posiform represens them in.
      // IMPORTANT NOTE : We skip dividing by 2 when calculating the implication
      // netowrk edge capacities to avoid rounding errors, but when we compute
      // the max flow and convert it back to a lower bound for the bqm, we must
      // take this into account and divide the max flow by 2.
      // See bottom of page 5 after equation 5 of the following paper.
      // Boros, Endre & Hammer, Peter & Tavares, Gabriel. (2006). Preprocessing of
      // unconstrained quadratic binary optimization. RUTCOR Research Report.
      auto coefficient = posiform.convertToPosiformCoefficient(it->bias);
      int variable_2 = posiform.mapVariableQuboToPosiform(it->v);
      int to_vertex = _mapper.variable_to_vertex(variable_2);
      if (coefficient > 0) {
        createImplicationNetworkEdges(
            from_vertex, _mapper.complement(to_vertex), coefficient);
      } else if (coefficient < 0) {
        createImplicationNetworkEdges(from_vertex, to_vertex, -coefficient);
      }
    }
  }

  // We separate out the creation of edges with source and sink, as if the
  // mapping of variables to vertices is ordered such that if variable x <
  // variable y, then vertices corresponding to x and x' both will be less than
  // vertices y and y' then we can keep the order of edges sorted by mapping
  // source and sink to the highest numbered vertices, and creating the
  // corresponding edges last. This ordering lets us optimize the process of
  // making the residual network symmetric.
  for (int variable = 0; variable < _num_variables; variable++) {
    int from_vertex = _mapper.variable_to_vertex(variable);
    int from_vertex_complement = _mapper.complement(from_vertex);
    auto linear = posiform.getLinear(variable);
    if (linear > 0) {
      createImplicationNetworkEdges(_source, from_vertex_complement, linear);
    } else if (linear < 0) {
      createImplicationNetworkEdges(_source, from_vertex, -linear);
    }
  }
  _adjacency_list_valid = true;
}

// Each term in posiform produces four edges in implication network
// the reverse edges and the symmetric edges.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::createImplicationNetworkEdges(
    int from_vertex, int to_vertex, capacity_t capacity) {
  int from_vertex_complement = _mapper.complement(from_vertex);
  int to_vertex_complement = _mapper.complement(to_vertex);
  int from_vertex_edge_index = _adjacency_list[from_vertex].size();
  int to_vertex_edge_index = _adjacency_list[to_vertex].size();
  int from_vertex_complement_edge_index =
      _adjacency_list[from_vertex_complement].size();
  int to_vertex_complement_edge_index =
      _adjacency_list[to_vertex_complement].size();

  // edge
  _adjacency_list[from_vertex].emplace_back(ImplicationEdge<capacity_t>(
      from_vertex, to_vertex, capacity, 0, to_vertex_edge_index,
      to_vertex_complement_edge_index));

  // reverse edge
  _adjacency_list[to_vertex].emplace_back(ImplicationEdge<capacity_t>(
      to_vertex, from_vertex, 0, capacity, from_vertex_edge_index,
      from_vertex_complement_edge_index));

  // symmetric edge
  _adjacency_list[to_vertex_complement].emplace_back(
      ImplicationEdge<capacity_t>(
          to_vertex_complement, from_vertex_complement, capacity, 0,
          from_vertex_complement_edge_index, from_vertex_edge_index));

  // reverse symmetric edge
  _adjacency_list[from_vertex_complement].emplace_back(
      ImplicationEdge<capacity_t>(from_vertex_complement, to_vertex_complement,
                                  0, capacity, to_vertex_complement_edge_index,
                                  to_vertex_edge_index));
}

// Make the residual network symmetric, by summing the residual capacities and
// original capacities of the implicaiton network. Here we multiply the
// capacities by 2, since the symmetric edges have same capacity.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::makeResidualSymmetric() {
  checkAdjacencyListValidity();
  // If the edges are sorted even if we create edges to/from vertices
  // corresponding to variables and their complements.
  bool edges_sorted = _mapper.complement_maintains_order();
#pragma omp parallel for schedule(dynamic)
  for (int vertex = 0; vertex < _num_vertices; vertex++) {
    int from_vertex_base = _mapper.non_complemented_vertex(vertex);
    auto eit = edges_sorted ? std::lower_bound(_adjacency_list[vertex].begin(),
                                               _adjacency_list[vertex].end(),
                                               from_vertex_base,
                                               ImplicationEdgeComparator())
                            : _adjacency_list[vertex].begin();
    auto eit_end = _adjacency_list[vertex].end();
    for (; eit != eit_end; eit++) {
      int to_vertex = eit->to_vertex;
      int to_vertex_complement = _mapper.complement(to_vertex);
      int to_vertex_base = _mapper.non_complemented_vertex(to_vertex);
      // We don not want to process the symmetric edges twice, we pick
      // the one that starts from the smaller vertex number when
      // complementation is not taken into account.
      if (to_vertex_base > from_vertex_base) {
        auto symmetric_eit = _adjacency_list[to_vertex_complement].begin() +
                             eit->symmetric_edge_index;
        capacity_t edge_residual = eit->residual;
        capacity_t symmetric_edge_residual = symmetric_eit->residual;
        // The paper states that we should average the residuals and
        // assign the average, but to avoid underflow we do not divide
        // by two but to keep the flow valid we multiply the
        // capacities by 2. The doubling of capacity is not needed for
        // the later steps in the algorithm, but will help us verify
        // if the symmetric flow is a valid flow or not.
        capacity_t residual_sum = edge_residual + symmetric_edge_residual;
        eit->residual = residual_sum;
        eit->scaleCapacity(2);
        symmetric_eit->residual = residual_sum;
        symmetric_eit->scaleCapacity(2);
      }
    }
  }
}

// We extract the adjacency list that contains only the edges in the
// implication network with positive residual capacity, excluding the one going
// into the source and coming out of the sink. We will run later algorithms on
// the extracted adjacency list. For a single thread this may or may not be
// slower when compared to working with the original network due to the extra
// step but if the graph is large and if we take advantage of parallelism we are
// supposed to get better performance ase there will be less page faults and
// more vertices/edges will fit into the cache due to smaller memory footprint
// of the extracted graph and also due to the fact that we can reduce the cost
// of extraction using multiple threads.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::
    extractResidualNetworkWithoutSourceInSinkOut(
        std::vector<std::vector<int>> &adjacency_list_residual,
        bool free_original_adjacency_list) {
  checkAdjacencyListValidity();
  adjacency_list_residual.resize(_num_vertices);
#pragma omp parallel
  {
    std::vector<int> temp_buffer(_num_vertices);
#pragma omp for
    for (int vertex = 0; vertex < _num_vertices; vertex++) {
      if (vertex != _sink) {
        int num_residual_out_edges = 0;
        auto eit = _adjacency_list[vertex].begin();
        auto eit_end = _adjacency_list[vertex].end();
        for (; eit != eit_end; eit++) {
          if (eit->residual > 0) {
            if (eit->to_vertex != _source) {
              temp_buffer[num_residual_out_edges++] = eit->to_vertex;
            }
          }
        }
        adjacency_list_residual[vertex].assign(
            temp_buffer.begin(), temp_buffer.begin() + num_residual_out_edges);
      }
      if (free_original_adjacency_list) {
        std::vector<ImplicationEdge<capacity_t>>().swap(
            _adjacency_list[vertex]);
      }
    }
  }

  if (free_original_adjacency_list) {
    std::vector<std::vector<ImplicationEdge<capacity_t>>>().swap(
        _adjacency_list);
    _adjacency_list_valid = false;
  }
}

// Fix only variables relevant to a strongly connected component.
template <class capacity_t>
void ImplicationNetwork<capacity_t>::fixStronglyConnectedComponentVariables(
    int component, stronglyConnectedComponentsInfo &scc_info,
    std::vector<std::vector<int>> &adjacency_list_components_transposed,
    std::vector<int> &out_degrees,
    std::vector<std::pair<int, int>> &fixed_variables,
    vector_based_queue<int> &component_queue, bool enqueue) {

  auto &components = scc_info.components;
  auto &complement_map = scc_info.complement_map;

  int complement_component = complement_map[component];
  out_degrees[component] = -1;
  out_degrees[complement_component] = -1;

  auto vit = components[component].begin();
  auto vit_end = components[component].end();
  for (; vit != vit_end; vit++) {
    int vertex = *vit;
    int base_vertex = _mapper.non_complemented_vertex(vertex);
    int variable = _mapper.non_complemented_vertex_to_variable(base_vertex);
    fixed_variables.push_back({variable, (vertex == base_vertex) ? 1 : 0});
  }

  auto eit = adjacency_list_components_transposed[component].begin();
  auto eit_end = adjacency_list_components_transposed[component].end();
  for (; eit != eit_end; eit++) {
    // This is the transpose of the adjacency matrix of components, so edges
    // are coming from the other components.
    int from_component = *eit;
    if (out_degrees[from_component] > 0) {
      out_degrees[from_component]--;
      if (enqueue) {
        component_queue.push(from_component);
      }
    }
  }

  eit = adjacency_list_components_transposed[complement_component].begin();
  eit_end = adjacency_list_components_transposed[complement_component].end();
  for (; eit != eit_end; eit++) {
    int from_component = *eit;
    if (out_degrees[from_component] > 0) {
      out_degrees[from_component]--;
      if (enqueue) {
        component_queue.push(from_component);
      }
    }
  }
}

// Fix both strong and weak pesistencies. This implementation is a bit different
// from the implementation mentioned in the paper : Boros, Endre & Hammer, Peter
// & Tavares, Gabriel. (2006). Preprocessing of unconstrained quadratic binary
// optimization. RUTCOR Research Report. Here after making the residual network
// symmetric and extracting the residual network, we specifically remove the
// edges going into the source and the ones coming out of the sink, this way
// when we generate the graph of strongly connected components, the source and
// sink will belong to different strongly connected components of their own,
// where they will be the sole members. Thus a breadth first search from the
// source will still find the strong persistencies and fix the variables that
// the algorithm from the above mentioned paper was supposed to find, since they
// will be in different components from the source now. We ignore fixing the
// component where the source is, that component should have only source as its
// member. Whenever we fix a component we reduce the outdegree of components
// which had edges into the component just fixed. Whenever a component has zero
// outdegree and is of not self complementing type, that is vertices and their
// complements both do not exist in the same component, it becomes a candidate
// for the fixing process.
template <class capacity_t>
capacity_t ImplicationNetwork<capacity_t>::fixStrongAndWeakVariables(
    std::vector<std::pair<int, int>> &fixed_variables) {

  PushRelabelSolver<ImplicationEdge<capacity_t>> push_relabel_solver(
      _adjacency_list, _source, _sink);
  capacity_t max_flow = push_relabel_solver.computeMaximumFlow(false);
  assert(isMaximumFlow(_adjacency_list, _source, _sink).second &&
         "Maximum flow is not valid.");

  makeResidualSymmetric();
  assert(isMaximumFlow(_adjacency_list, _source, _sink).second &&
         "Maximum flow is not valid.");

  // The removal of certain edges will create a component with the source only
  // and another component with the sink only.
  std::vector<std::vector<int>> adjacency_list_residual;
  extractResidualNetworkWithoutSourceInSinkOut(adjacency_list_residual, true);

  std::vector<int> vertex_to_component_map;
  int num_components = stronglyConnectedComponents(adjacency_list_residual,
                                                   vertex_to_component_map);

  stronglyConnectedComponentsInfo scc_info(num_components,
                                           vertex_to_component_map, _mapper);

  std::vector<std::vector<int>> adjacency_list_components;
  createGraphOfStronglyConnectedComponents(
      scc_info.vertex_to_component_map, scc_info.components,
      adjacency_list_residual, adjacency_list_components);

  std::vector<std::vector<int>> adjacency_list_components_transposed;
  getTransposedAdjacencyList(adjacency_list_components,
                             adjacency_list_components_transposed);

  std::vector<int> bfs_depth_values;
  int UNVISITED = breadthFirstSearch(
      adjacency_list_components, scc_info.source_component, bfs_depth_values);

  auto &complement_map = scc_info.complement_map;
  fixed_variables.reserve(_num_variables);
  vector_based_queue<int> component_queue(num_components);
  std::vector<int> out_degrees(num_components, -1);

  for (int component = 0; component < num_components; component++) {
    if (complement_map[component] != component) {
      out_degrees[component] = adjacency_list_components[component].size();
    }
  }

  out_degrees[scc_info.source_component] = -1;
  out_degrees[scc_info.sink_component] = -1;

  for (int component = 0; component < num_components; component++) {
    // Skipping the source component and the other unvisited ones.
    if ((bfs_depth_values[component] != 0) &&
        (bfs_depth_values[component] != UNVISITED)) {
      fixStronglyConnectedComponentVariables(
          component, scc_info, adjacency_list_components_transposed,
          out_degrees, fixed_variables, component_queue, false);
    }
  }

  for (int component = 0; component < num_components; component++) {
    if (out_degrees[component] == 0) {
      component_queue.push(component);
    }
  }

  while (!component_queue.empty()) {
    int component = component_queue.pop();
    fixStronglyConnectedComponentVariables(
        component, scc_info, adjacency_list_components_transposed, out_degrees,
        fixed_variables, component_queue, true);
  }
  return max_flow;
}

// Fix only the strong variables which can be trivially found. That is fix the
// variables which can be reached from the source in the residual graph. This is
// a subset or say the initial part of the algorithm mentioned in the paper :
// Boros, Endre & Hammer, Peter & Tavares, Gabriel. (2006). Preprocessing of
// unconstrained quadratic binary optimization. RUTCOR Research Report.
template <class capacity_t>
capacity_t ImplicationNetwork<capacity_t>::fixTriviallyStrongVariables(
    std::vector<std::pair<int, int>> &fixed_variables) {

  PushRelabelSolver<ImplicationEdge<capacity_t>> push_relabel_solver(
      _adjacency_list, _source, _sink);
  capacity_t max_flow = push_relabel_solver.computeMaximumFlow(false);
  assert(isMaximumFlow(_adjacency_list, _source, _sink).second &&
         "Maximum flow is not valid.");

  std::vector<int> bfs_depth_values;
  int UNVISITED = breadthFirstSearchResidual(_adjacency_list, _source,
                                             bfs_depth_values, false, false);

  fixed_variables.reserve(_num_variables);
  std::vector<bool> variable_fixed(_num_variables, false);

  for (int vertex = 0; vertex < _num_vertices; vertex++) {
    if (bfs_depth_values[vertex] != UNVISITED) {
      int base_vertex = _mapper.non_complemented_vertex(vertex);
      if (base_vertex == _source) {
        continue;
      }
      int variable = _mapper.non_complemented_vertex_to_variable(base_vertex);
      fixed_variables.push_back({variable, (vertex == base_vertex) ? 1 : 0});
    }
  }
  return max_flow;
}

// Parent function for fixing posiform based variables.
template <class capacity_t>
capacity_t ImplicationNetwork<capacity_t>::fixVariables(
    std::vector<std::pair<int, int>> &fixed_variables,
    bool only_trivially_strong) {
  if (only_trivially_strong) {
    return fixTriviallyStrongVariables(fixed_variables);
  } else {
    return fixStrongAndWeakVariables(fixed_variables);
  }
}

// For debugging.
template <class capacity_t> void ImplicationNetwork<capacity_t>::print() {
  checkAdjacencyListValidity();
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
    std::cout << std::endl;
  }
}
#endif // IMPLICATION_NETWORK_INCLUDED
