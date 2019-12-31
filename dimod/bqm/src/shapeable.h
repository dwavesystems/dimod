// Copyright 2019 D-Wave Systems Inc.
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

#ifndef DIMOD_BQM_SRC_SHAPEABLE_H_
#define DIMOD_BQM_SRC_SHAPEABLE_H_

#include <map>
#include <utility>
#include <vector>

namespace dimod {

template<class Neighborhood, typename LinearBias> using
ShapeableBQM = std::vector<std::pair<Neighborhood, LinearBias>>;

template<typename VarIndex, typename Bias> using
MapNeighborhood = typename std::map<VarIndex, Bias>;

template<class VarIndex, class QuadraticBias> using
VectorNeighborhood = std::vector<std::pair<VarIndex, QuadraticBias>>;

template<class VarIndex, class Bias> using
AdjMapBQM = ShapeableBQM<MapNeighborhood<VarIndex, Bias>, Bias>;

template<class VarIndex, class Bias> using
AdjVectorBQM = ShapeableBQM<VectorNeighborhood<VarIndex, Bias>, Bias>;

// Construction

template<typename VarIndex, typename Bias, class BQM>
void copy_bqm(BQM &bqm, AdjMapBQM<VarIndex, Bias> &bqm_copy);

template<typename VarIndex, typename Bias, class BQM>
void copy_bqm(BQM &bqm, AdjVectorBQM<VarIndex, Bias> &bqm_copy);


// Read the BQM

template<class Neighborhood, class Bias>
std::size_t num_variables(const ShapeableBQM<Neighborhood, Bias>&);

template<class Neighborhood, class Bias>
std::size_t num_interactions(const ShapeableBQM<Neighborhood, Bias>&);

template<class Neighborhood, class VarIndex, class Bias>
Bias get_linear(const ShapeableBQM<Neighborhood, Bias>&, VarIndex);

template<class Neighborhood, class VarIndex, class Bias>
std::pair<Bias, bool> get_quadratic(const ShapeableBQM<Neighborhood, Bias>&,
                                    VarIndex, VarIndex);

template<class Neighborhood, class VarIndex, class Bias>
std::size_t degree(const ShapeableBQM<Neighborhood, Bias>&, VarIndex);

// unfortunately cython cannot do delayed template deduction, so we cannot have
// the neighborhood as a template type. Therefore we need to enumerate the two
// different types
template<class VarIndex, class Bias>
std::pair<typename MapNeighborhood<VarIndex, Bias>::iterator,
          typename MapNeighborhood<VarIndex, Bias>::iterator>
neighborhood(AdjMapBQM<VarIndex, Bias>&, VarIndex);

template<class VarIndex, class Bias>
std::pair<typename VectorNeighborhood<VarIndex, Bias>::iterator,
          typename VectorNeighborhood<VarIndex, Bias>::iterator>
neighborhood(AdjVectorBQM<VarIndex, Bias>&, VarIndex);

// Change the values in the BQM

template<class Neighborhood, class VarIndex, class Bias>
void set_linear(ShapeableBQM<Neighborhood, Bias>&, VarIndex, Bias);

template<class VarIndex, class Bias>
void set_quadratic(AdjMapBQM<VarIndex, Bias>&, VarIndex, VarIndex, Bias);

template<class VarIndex, class Bias>
void set_quadratic(AdjVectorBQM<VarIndex, Bias>&, VarIndex, VarIndex, Bias);

// Change the structure of the BQM

// Add one variable and return the index of the new variable added.
template<class Neighborhood, class Bias>
std::size_t add_variable(ShapeableBQM<Neighborhood, Bias>&);

template<class Neighborhood, class Bias>
std::size_t pop_variable(ShapeableBQM<Neighborhood, Bias>&);

template<class VarIndex, class Bias>
bool remove_interaction(AdjMapBQM<VarIndex, Bias>&, VarIndex, VarIndex);

template<class VarIndex, class Bias>
bool remove_interaction(AdjVectorBQM<VarIndex, Bias>&, VarIndex, VarIndex);
}  // namespace dimod

#endif  // DIMOD_BQM_SRC_SHAPEABLE_H_
