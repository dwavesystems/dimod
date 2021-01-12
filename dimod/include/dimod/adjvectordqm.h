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

#ifndef DIMOD_ADJVECTORDQM_H_
#define DIMOD_ADJVECTORDQM_H_

#include <stdio.h>
#include <algorithm>
#include <utility>
#include <vector>

#include "dimod/utils.h"
#include "dimod/adjvectorbqm.h"

namespace dimod {

template <class V, class C, class B>
class AdjVectorDQM {
public:
  AdjVectorBQM _bqm;
  std::vector<C> _case_starts;
  std::vector<std::vector<V>> _adj;
  
 AdjVectorBQM() { _case_starts.push_back(0); }

 void add_variable(size_t num_cases) {
  assert(num_cases > 0);
  auto v = _adj.resize(_adj.size() + 1);
  for(auto n = 0; n < num_cases; n++){
     _bqm.add_variable();
  }
  _case_starts.push_back(_bqm.num_variables());
  return v;
 } 

 // Skipping copy routine since a default copy constructor will work. 
 // No deep copying is needed.

 std::vector<B> energies(std::vector<


}  // namespace dimod

#endif  // DIMOD_ADJVECTORDQM_H_
