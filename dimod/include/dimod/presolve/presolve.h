// Copyright 2023 D-Wave Systems Inc.
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

#pragma once

#ifndef _WIN32
// MSVC does not support std::experimental
#include <experimental/propagate_const>
#endif

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "dimod/constrained_quadratic_model.h"
#include "dwave/flags.hpp"

namespace dwave::presolve {

template <class Bias, class Index = int, class Assignment = Bias>
class Presolver {
 public:
    using model_type = dimod::ConstrainedQuadraticModel<Bias, Index>;

    using bias_type = Bias;
    using index_type = Index;
    using size_type = typename model_type::size_type;

    using assignment_type = Assignment;

    Presolver();
    ~Presolver();

    /// Construct a presolver from a constrained quadratic model.
    explicit Presolver(model_type model);

    /// Add presolve techniques to be run. This will not affect existing loaded techniques.
    TechniqueFlags add_techniques(TechniqueFlags techniques);

    /// Apply any loaded presolve techniques. Acts on the model() in-place.
    bool apply();

    /// Detach the constrained quadratic model and return it.
    /// This clears the model from the presolver.
    model_type detach_model();

    const Feasibility& feasibility() const;

    /// Return a const reference to the held constrained quadratic model.
    const model_type& model() const;

    /// Normalize the model.
    bool normalize();

    /// Presolve a normalized model.
    bool presolve();
    bool presolve(std::chrono::duration<double> time_limit);

    /// Return a sample of the original CQM from a sample of the reduced CQM.
    std::vector<assignment_type> restore(std::vector<assignment_type> reduced) const;

    /// Set the presolve techniques to be run.
    TechniqueFlags set_techniques(TechniqueFlags techniques);

    /// Return the currently-loaded presolve techniques.
    TechniqueFlags techniques() const;

 protected:
    class PresolverImpl_;

#ifndef _WIN32
    std::experimental::propagate_const<std::unique_ptr<PresolverImpl_>> impl_;
#else
    // MSVC does not support std::experimental
    std::unique_ptr<PresolverImpl_> impl_;
#endif
};

}  // namespace dwave::presolve
