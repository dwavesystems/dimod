#ifndef HELPER_GRAPH_ALGORITHM_HPP_INCLUDED
#define HELPER_GRAPH_ALGORITHM_HPP_INCLUDED

#include <algorithm>
#include <iostream>
#include <vector>

#include "helper_data_structures.hpp"

// Check if the flow value in a given graph represented as an adjacency list is
// valid or not.
template <class EdgeType>
std::pair<typename EdgeType::capacity_type, bool>
isFlowValid(std::vector<std::vector<EdgeType>> &adjacency_list, int source,
            int sink) {

  using capacity_t = typename EdgeType::capacity_type;

  bool valid_flow = true;
  std::vector<capacity_t> excess(adjacency_list.size(), 0);

  std::cout << "Validating flow of flow network ..." << std::endl;

  // Since we are validating our algorithms, we will not retrieve the
  // value of residual/capacity of a reverse edge from its counterpart
  // which we generally do for performance reasons. Here we will actually
  // access the data and verify if the flow constraints hold or not.
  for (int i = 0; i < adjacency_list.size(); i++) {
    for (int j = 0; j < adjacency_list[i].size(); j++) {
      int to_vertex = adjacency_list[i][j].to_vertex;
      int reverse_edge_index = adjacency_list[i][j].reverse_edge_index;
      capacity_t edge_capacity = adjacency_list[i][j].getCapacity();
      capacity_t edge_residual = adjacency_list[i][j].residual;
      capacity_t reverse_edge_capacity =
          adjacency_list[to_vertex][reverse_edge_index].getCapacity();
      capacity_t reverse_edge_residual =
          adjacency_list[to_vertex][reverse_edge_index].residual;
      bool valid_edge = (adjacency_list[i][j].getReverseEdgeCapacity() ==
                         reverse_edge_capacity) &&
                        (adjacency_list[i][j].getReverseEdgeResidual() ==
                         reverse_edge_residual) &&
                        (edge_capacity >= 0) && (edge_residual >= 0);
      if (edge_capacity > 0) {
        // Residual edge having capacity 0 is a valid assumption for posiforms,
        // since no term with two variables appear multiple times with different
        // ordering of the variables. This assumption can be maintained with
        // other graphs too.
        valid_edge = valid_edge && (reverse_edge_capacity == 0) &&
                     (edge_residual <= edge_capacity) &&
                     ((edge_residual + reverse_edge_residual) == edge_capacity);

        capacity_t flow = (edge_capacity - edge_residual);
        excess[i] -= flow;
        excess[to_vertex] += flow;
      }
      if (!valid_edge) {
        std::cout << "Invalid Flow due to following edge pair :" << std::endl;
        adjacency_list[i][j].print();
        adjacency_list[to_vertex][reverse_edge_index].print();
      }
      valid_flow = valid_flow && valid_edge;
    }
  }

  for (int i = 0; i < excess.size(); i++) {
    if ((i == source) || (i == sink)) {
      continue;
    }
    if (excess[i]) {
      std::cout << "Excess flow of " << excess[i] << " in vertex : " << i
                << std::endl;
      valid_flow = false;
    }
  }

  if (excess[sink] != -excess[source]) {
    std::cout << "Flow out of source is not equal to flow into sink."
              << std::endl;
    valid_flow = false;
  }

  return {excess[sink], valid_flow};
}

// Perform breadth first search from a certain vertex, a depth equal to  the number of vertices means that vertex could not be reached from the start_vertex, since the maximum depth can be equal to number of vertices -1.
template <class EdgeType>
void breadthFirstSearch(std::vector<std::vector<EdgeType>>& adjacency_list,
                        int start_vertex, std::vector<int>& depth_values,
                        bool reverse = false, bool print_result = false) {
  using capacity_t = typename EdgeType::capacity_type;
  int num_vertices = adjacency_list.size();
  vector_based_queue<int> vertex_queue(num_vertices);
  depth_values.resize(num_vertices);
  std::fill(depth_values.begin(), depth_values.end(), num_vertices);

  depth_values[start_vertex] = 0;
  vertex_queue.push(start_vertex);

  // The check for whether the search should be reverse or not could be done
  // inside the innermost loop, but that would be detrimental to performance.
  if (reverse) {
    while (!vertex_queue.empty()) {
      int v_parent = vertex_queue.pop();
      int current_depth = depth_values[v_parent] + 1;
      auto eit = adjacency_list[v_parent].begin();
      auto eit_end = adjacency_list[v_parent].end();
      for (; eit != eit_end; eit++) {
        int to_vertex = eit->to_vertex;
        if (eit->getReverseEdgeResidual() &&
            depth_values[to_vertex] == num_vertices) {
          depth_values[to_vertex] = current_depth;
          vertex_queue.push(to_vertex);
        }
      }
    }
  } else {
    while (!vertex_queue.empty()) {
      int v_parent = vertex_queue.pop();
      int current_depth = depth_values[v_parent] + 1;
      auto eit = adjacency_list[v_parent].begin();
      auto eit_end = adjacency_list[v_parent].end();
      for (; eit != eit_end; eit++) {
        int to_vertex = eit->to_vertex;
        if (eit->residual && depth_values[to_vertex] == num_vertices) {
          depth_values[to_vertex] = current_depth;
          vertex_queue.push(to_vertex);
        }
      }
    }
  }

  if (print_result) {
    std::vector<int> level_sizes;
    std::vector<std::vector<int>> levels;
    levels.resize(num_vertices + 1);
    level_sizes.resize(num_vertices + 1, 0);
    for (int i = 0; i < depth_values.size(); i++) {
      level_sizes[depth_values[i]]++;
    }
    for (int i = 0; i < level_sizes.size(); i++) {
      levels[i].reserve(level_sizes[i]);
    }
    for (int i = 0; i < depth_values.size(); i++) {
      levels[depth_values[i]].push_back(i);
    }
    std::cout << endl;
    std::cout << "Printing " << (reverse ? "reverse " : "")
              << "breadth first search result starting from vertex : "
              << start_vertex << std::endl;
    std::cout << endl;
    for (int i = 0; i < levels.size(); i++) {
      if (!levels[i].size()) {
        continue;
      }
      std::cout << "Level " << i << " has " << levels[i].size()
                << " vertices : " << std::endl;
      for (int j = 0; j < levels[i].size(); j++) {
        std::cout << levels[i][j] << " ";
      }
      std::cout << endl;
    }
    std::cout << endl;
  }
}

// Check if the flow value in a given graph represented as an adjacency list is
// a valid max-flow or not and also return the flow value.
template <class EdgeType>
std::pair<typename EdgeType::capacity_type, bool>
isMaximumFlow(std::vector<std::vector<EdgeType>> &adjacency_list, int source,
              int sink) {

  // If the flow follows the constraints of network flow.
  auto validity_result = isFlowValid(adjacency_list, source, sink);

  // If the flow is a maximum flow, the source will be unreachable from the sink
  // through a reverse breadth first search, meaning the source cannot reach the
  // sink through any augmenting path.
  std::vector<int> depth_values;
  int num_vertices = adjacency_list.size();
  breadthFirstSearch(adjacency_list, sink, depth_values, true, true);
  return {validity_result.first,
          (validity_result.second && (depth_values[source] == num_vertices))};
}

#endif // HELPER_GRAPH_ALGORITHM_HPP_INCLUDED 
