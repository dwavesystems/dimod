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
    using bias_type = B;
    using case_type = C;
    using variable_type = V;
    using size_type = std::size_t;

  AdjVectorBQM _bqm;
  std::vector<case_type> _case_starts;
  std::vector<std::vector<variable_type>> _adj;
  
 AdjVectorBQM() { _case_starts.push_back(0); }
 
 variable_type num_variables() {
   return _adj.size();
 }

 variable_type num_cases(variable_type v = -1) {
   assert(v < this->num_variables());
   if(v < 0) {
     return _bqm.num_variables();
   } else {
      return (_case_starts[v+1] - _case_starts[v]);
   }
 }

 variable_type add_variable(size_t num_cases) {
  assert(num_cases > 0);
  auto v = _adj.size();
  _adj.resize(v+1);
  for(auto n = 0; n < num_cases; n++){
     _bqm.add_variable();
  }
  _case_starts.push_back(_bqm.num_variables());
  return v;
 } 

 void get_linear(variable_type v, bias_type* biases) {
   assert(v >= 0 && v < this->num_variables());
   for(auto case_v = 0, num_cases_v = this->num_cases(v); case_v < num_cases_v; case_v++) {
     biases[case_v] = _bqm.get_linear(_case_starts[v] + case_v);
   } 
 }

 bias_type get_linear_case(variable_type v, case_type case_v) {
   assert(v >= 0 && v < this->num_variables());
   assert(case_v >= 0 && case_v < num_cases(v));
   return _bqm.get_linear(_case_starts[v] + case_v);
 }

 // Returns false if there is no interaction among the variables. 
 bool get_quadratic(variable_type u, variable_type v, bias_type* quadratic_biases) {
   assert(u >=0 && u < this->num_variables());
   assert(v >=0 && v < this->num_variables());
   auto it = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
   if( it == _adj[u].end() || *it != v)  {
     return false;
   }
   auto num_cases_u = num_cases(u);
   auto num_cases_v = num_cases(v);
   for(auto case_u = 0; case_u < num_cases_u; case_u++) {
     auto span = _bqm.neighborhood(_case_starts[u] + case_u, _case_starts[v]);
     while(span.first != span.second  && *(span.first) < _case_starts[v+1]) {
        case_v = *(span.first) - _case_starts[v];
        quadratic_biases[case_u][case_v] = *(span.first).second; 
        span.first++;
     }
   }   
   return true;   
 }

 bias_type get_quadratic_case(variable_type u, case_type case_u, variable_type v, case_type case_v) {
   assert(u >= 0 && u < this->num_variables());
   assert(case_u >= 0 && case_u < num_cases(v));
   assert(v >= 0 && v < this->num_variables());
   assert(case_v >= 0 && case_v < num_cases(v));
   // should add assert for u != v ?
   auto cu = _case_starts[u] + case_u;
   auto cv = _case_starts[v] + case_v;
   return _bqm.get_quadratic(cu , cv).first;
 }

 size_type num_case_interactions() {
   return _bqm.num_interactions();
 }

 size_type num_variaables_interactions() {
   size_type num = 0;
   for(auto v = 0, vend = this->num_variables(); v < vend; v++) {
     num+= _adj[v].size(); 
   }
   return (num/2);
}
  
 void set_linear(variable_type v, bias_type* p_biases) {
    for(auto case_v = 0, num_cases_v = this->num_cases(v); case_v < num_cases_v; case_v++) {
       _bqm.set_linear(_case_starts[v] + case_v, p_biases[case_v]);
    }
 } 

 void set_linear_case(variable_type v, case_type case_v, bias_type b) {
    assert(case_v >= 0 && case_v < this->num_cases(v));
    _bqm.set_linear(_case_starts[v] + case_v, b); 
 }

 bool set_quadratic(variable_type u, variable_type v, bias_type* p_biases) {
   assert(u >=0 && u < this->num_variables());
   assert(v >=0 && v < this->num_variables());
   assert(u != v);
   num_cases_u = num_cases(u);
   num_cases_v = num_cases(v); 
   auto num_cases_u = num_cases(u);
   auto num_cases_v = num_cases(v);
   for(auto case_u = 0; case_u < num_cases_u; case_u++) {
     cu = _case_starts[u] + case_u;
     for(auto case_v = 0; case_v < num_cases_v; case_v++) {
       cv = _case_starts[v] + case_v;
       auto bias = p_biases[cu * num_cases_v + case_v];
       _bqm.set_quadratic(cu, cv, bias);
     }
   }  
   auto low = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
   if( low == _adj[u].end() || *low != v) {
     _adj[u].insert(low, v);
     _adj[v].insert(std::lower_bound(_adj[v].begin(), _adj[v].end(), u), u);
   }
   return true;
 }

 bool set_quadratic_case(variable_type u, case_type case_u, variable_type v, case_type case_v, bias_type bias) {
   assert(u >= 0 && u < this->num_variables());
   assert(case_u >= 0 && case_u < num_cases(v));
   assert(v >= 0 && v < this->num_variables());
   assert(case_v >= 0 && case_v < num_cases(v));
   auto cu = _case_starts[u] + case_u;
   auto cv = _case_starts[v] + case_v;
   _bqm.set_quadratic(cu, cv, bias);
   auto low = std::lower_bound(_adj[u].begin(), _adj[u].end(), v);
   if( low == _adj[u].end() || *low != v) {
     _adj[u].insert(low, v);
     _adj[v].insert(std::lower_bound(_adj[v].begin(), _adj[v].end(), u), u);
   }
   return true;
 }

 void energies(case_type* p_samples, int num_samples, variable_type num_variables, bias_type* p_energies) {
   assert(num_variables == this->num_variables());
   memset(p_energies, 0, num_samples * sizeof(bias_type));  
   #pragma omp parallel for
   for(auto si = 0; si < num_samples; si++) {
     case_type* p_curr_sample = samples + (si * num_variables);
     bias_type* p_curr_energy = p_energies + si;
     for(auto u = 0; u < num_variables; u++) {
       auto case_u = p_curr_sample[u];
       assert(case_u < num_cases(u));
       auto cu = _case_starts[u] + case_u;
       *p_curr_energy += _bqm.get_linear(cu);
       for(auto vi = 0; vi < _adj[u].size(); vi++) {
         auto v = _adj[u][vi];
         // We only care about lower triangle.
         if(v > u) {
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
