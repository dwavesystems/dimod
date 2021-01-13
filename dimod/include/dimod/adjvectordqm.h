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

 int add_variable(size_t num_cases) {
   if(num_cases <=0 ) {
    return -1;
   }
  auto v = _adj.resize(_adj.size() + 1);
  for(auto n = 0; n < num_cases; n++){
     _bqm.add_variable();
  }
  _case_starts.push_back(_bqm.num_variables());
  return v;
 } 

 void get_linear(V v, B* biases) {
    for(int case_v = 0; case_v < num_cases(v); case_v++) {
      biases[case_v] = _bqm.get_linear(_case_starts[v] + case_v);
    } 
 }

 B get_linear_case(V v, C case_v) {
   assert(case_v >= 0 && case_v < num_cases(v));
   return _bqm.get_linear(case_starts[v] + case_v);
 }
 
 int get_quadratic(V u, V v, std::vector<std::vector<B>>& quadratic_biases) {
   assert(u >=0 && u < _adj.size());
   assert(v >=0 && v < _adj.size());
   auto it = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
   if( it == _adj[u].end() || *it != v)  {
     return -1;
   }
   auto num_cases_u = num_cases(u);
   auto num_cases_v = num_cases(v);
   quadratic_biases.resize(num_cases_u);
   for(int i = 0; i < u ; i++){
     quadratic_biases.resize(num_cases_v, 0);
   }  
   
   for(auto case_u = 0; case_u < num_cases_u; case_u++) {
     auto span = _bqm.neighborhood(case_u + _case_starts[u], _case_starts[v]);
     while(span.first != span.second  && *(span.first) < _case_starts[v+1]) {
        case_v = *(span.first) - _case_starts[v];
        quadratic_biases[case_u][case_v] = *(span.first).second; 
        span.first++;
     }
   }   
   return 0;   
 }


 // Skipping copy routine since a default copy constructor will work. 
 // No deep copying is needed.

 void energies(C* p_samples, int num_variables, int num_samples, B* p_energies) {
   assert(num_variables == _bqm.num_variables());
   memset(p_energies, 0, num_variables * sizeof(C));  
   for(auto si = 0; si < num_samples; si++) {
     C* p_curr_sample = samples + si * num_variables;
     for(auto u = 0; u < num_variables; u++) {
       auto case_u = p_curr_sample_es[u];
       assert(case_u < num_cases(u));
       auto cu = _case_starts[u] + case_u;
       p_energies[si] += _bqm.get_linear(cu);
       for(auto vi = 0; vi < _adj[u].size(); vi++) {
         auto v = _adj[u][vi];
         // We only care about lower triangle.
         if( v > u) {
           break;
         }
         auto case_v = p_cur_sample[v];
         auto cv = _case_starts[v] + case_v;
         auto out = _bqm.get_quadratic(cu,cv);
         if(out.second) {
            p_energies[si]+= out.first; 
         }
       } 
     }
   } 
 } 

 }
}  // namespace dimod

#endif  // DIMOD_ADJVECTORDQM_H_
