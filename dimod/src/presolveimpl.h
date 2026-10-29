// Copyright 2022 D-Wave Systems Inc.
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dimod/constrained_quadratic_model.h"
#include "dwave/exceptions.hpp"
#include "dwave/flags.hpp"

namespace dwave {
namespace presolve {

template <class Bias, class Index = int, class Assignment = double>
class PresolverImpl {
 public:
    using model_type = dimod::ConstrainedQuadraticModel<Bias, Index>;
    using constraint_type = dimod::Constraint<Bias, Index>;
    using expression_type = dimod::Expression<Bias, Index>;
    using expression_base_type = dimod::abc::QuadraticModelBase<Bias, Index>;

    using bias_type = Bias;
    using index_type = Index;
    using size_type = typename model_type::size_type;

    using assignment_type = Assignment;

    static constexpr double FEASIBILITY_TOLERANCE = 1.0e-6;
    static constexpr double INF = 1.0e30;

    /// Default constructor.
    PresolverImpl() = default;

    /// Default destructor.
    ~PresolverImpl() = default;

    /// Construct a presolver from a constrained quadratic model.
    explicit PresolverImpl(model_type model) : model_(std::move(model)) {}

    /// Apply any loaded presolve techniques. Acts on the model() in-place.
    bool apply() {
        // Use | rather than || because we don't want to short-circuit
        return normalize() | presolve();
    }

    /// Detach the constrained quadratic model and return it.
    /// This clears the model from the presolver.
    model_type detach_model() {
        detached_ = true;
        return model_.detach_model();
    }

    const Feasibility& feasibility() const { return model_.feasibility; }

    // Get the maximal activity constributed by variable v
    static bias_type maximal_activity(const constraint_type& constraint, index_type v) {
        assert(constraint.is_linear());  // O(n) check; todo: support quadratic

        bias_type a = constraint.linear(v);
        if (a > 0) {
            return a * constraint.upper_bound(v);
        } else {
            return a * constraint.lower_bound(v);
        }
    }

    // Get the maximal activity of a constraint
    static bias_type maximal_activity(const constraint_type& constraint) {
        assert(constraint.is_linear());  // O(n) check; todo: support quadratic

        bias_type activity = constraint.offset();  // should be 0, but just in case
        for (const auto& v : constraint.variables()) {
            activity += maximal_activity(constraint, v);
        }
        return activity;
    }

    // Get the minimal activity constributed by variable v
    static bias_type minimal_activity(const constraint_type& constraint, index_type v) {
        assert(constraint.is_linear());  // O(n) check; todo: support quadratic

        bias_type a = constraint.linear(v);
        if (a > 0) {
            return a * constraint.lower_bound(v);
        } else {
            return a * constraint.upper_bound(v);
        }
    }

    // Get the minimal activity of a constraint
    static bias_type minimal_activity(const constraint_type& constraint) {
        assert(constraint.is_linear());  // O(n) check; todo: support quadratic

        bias_type activity = constraint.offset();  // should be 0, but just in case
        for (const auto& v : constraint.variables()) {
            activity += minimal_activity(constraint, v);
        }
        return activity;
    }

    /// Return a const reference to the held constrained quadratic model.
    const model_type& model() const { return model_.model(); }

    bool normalize() {
        if (detached_) {
            throw std::logic_error(
                    "model has been detached, so there is no model to apply presolve() to");
        }

        bool changes = false;

        changes |= normalization_check_nan();
        changes |= normalization_replace_inf();
        changes |= normalization_spin_to_binary();
        changes |= normalization_remove_offsets();
        changes |= normalization_remove_self_loops();
        changes |= normalization_flip_constraints();
        changes |= normalization_remove_invalid_markers();
        changes |= normalization_fix_bounds();

        normalized_ = true;

        return changes;
    }

    /// Normalize the variable bounds by vartype and check that lb <= ub
    bool normalization_fix_bounds() {
        bool changes = false;
        for (size_type v = 0; v < model_.num_variables(); ++v) {
            // tighten bounds based on the vartype
            switch (model_.vartype(v)) {
                case dimod::Vartype::SPIN:
                    // we could handle this case, but because normalization should
                    // take care of it we just throw an error for simplicity
                    throw std::logic_error(
                            "normalization_fix_bounds() must be run after "
                            "normalization_spin_to_binary()");
                case dimod::Vartype::BINARY:
                    changes |= model_.set_upper_bound(v, 1);
                    changes |= model_.set_lower_bound(v, 0);

                    // we carry on into INTEGER to handle fractional

                case dimod::Vartype::INTEGER:
                    if (std::ceil(model_.lower_bound(v)) > std::floor(model_.upper_bound(v))) {
                        throw InvalidModelError(
                                "variable lower bound must be less than or equal to upper bound");
                    }

                    changes |= model_.set_upper_bound(v, std::floor(model_.upper_bound(v)));
                    changes |= model_.set_lower_bound(v, std::ceil(model_.lower_bound(v)));
                    break;
                case dimod::Vartype::REAL:
                    // For REAL variables there's nothing to do
                    break;
            }

            // check that the bounds are valid
            // Dev note: should we consider using FEASIBILITY_TOLERANCE here?
            if (model_.lower_bound(v) > model_.upper_bound(v)) {
                throw InvalidModelError(
                        "variable lower bound must be less than or equal to upper bound");
            }
        }
        return changes;
    }

    bool normalization_check_nan() const {
        bool changes = normalization_check_nan(model_.objective());
        for (auto& constraint : model_.constraints()) {
            changes |= normalization_check_nan(constraint);
        }

        // test the bounds
        for (size_type v = 0, last = model_.num_variables(); v < last; ++v) {
            if (std::isnan(model_.lower_bound(v)) || std::isnan(model_.upper_bound(v))) {
                throw InvalidModelError("bounds cannot be NAN");
            }
        }

        return changes;
    }

    static bool normalization_check_nan(const constraint_type& constraint) {
        // test the rhs
        if (std::isnan(constraint.rhs())) {
            throw InvalidModelError("constraint rhs cannot be NAN");
        }

        // test the weight
        if (std::isnan(constraint.weight())) {
            throw InvalidModelError("constraint weight cannot be NAN");
        }

        // test the biases and offset
        return normalization_check_nan(static_cast<const expression_type&>(constraint));
    }

    static bool normalization_check_nan(const expression_type& expression) {
        // We only care about the biases, so let's just cast to the base type for speed
        const dimod::abc::QuadraticModelBase<bias_type, index_type>& base = expression;

        for (auto it = base.cbegin_quadratic(), end = base.cend_quadratic(); it != end; ++it) {
            if (std::isnan(it->bias)) throw InvalidModelError("biases cannot be NAN");
        }

        for (size_type v = 0; v < base.num_variables(); ++v) {
            if (std::isnan(base.linear(v))) throw InvalidModelError("biases cannot be NAN");
        }

        if (std::isnan(base.offset())) {
            throw InvalidModelError("offset cannot be NAN");
        }

        return false;  // no changes made
    }

    /// Convert any >= constraints into <=.
    bool normalization_flip_constraints() {
        bool changes = false;
        for (auto& constraint : model_.constraints()) {
            changes |= normalization_flip_constraint(constraint);
        }
        return changes;
    }

    /// Convert a >= constraint into <=.
    static bool normalization_flip_constraint(constraint_type& constraint) {
        if (constraint.sense() == dimod::Sense::GE) {
            constraint.scale(-1);
            return true;   // we made changes
        }

        return false; // no changes made
    }

    bool normalization_remove_invalid_markers() {
        bool changes = false;

        // find all of the discrete constraints
        std::vector<index_type> discrete;
        for (size_type c = 0; c < model_.num_constraints(); ++c) {
            auto& constraint = model_.constraint_ref(c);

            if (!constraint.marked_discrete()) {
                continue;
            }

            // we can check if it's well formed
            if (constraint.is_onehot()) {
                discrete.push_back(c);
            } else {
                constraint.mark_discrete(false);  // if it's not one-hot, it's not discrete
                changes = true;
            }
        }

        // if there are no discrete constraints (remaining) then we can just return now
        if (!discrete.size()) {
            return changes;
        }

        // Make sure the constraints don't overlap by tracking which variables are already
        // used in a discrete constraint.
        std::vector<bool> used(model_.num_variables(), false);
        for (const auto& c : discrete) {
            auto& constraint = model_.constraint_ref(c);

            for (const auto& v : constraint.variables()) {
                if (used[v]) {
                    // there is a variable already used by another discrete constraint
                    constraint.mark_discrete(false);
                    changes = true;
                    break;
                }
            }

            if (!constraint.marked_discrete()) continue;  // it was overlapping

            for (const auto& v : constraint.variables()) {
                used[v] = true;
            }
        }

        return changes;
    }

    /// Remove the offsets from all constraints in the model.
    bool normalization_remove_offsets() {
        bool changes = false;
        for (auto& constraint : model_.constraints()) {
            changes |= normalization_remove_offset(constraint);
        }
        return changes;
    }

    /// Remove the offset from a cosntraint. E.g. `x + 1 <= 2` becomes `x <= 1`.
    static bool normalization_remove_offset(constraint_type& constraint) {
        if (constraint.offset()) {
            constraint.set_rhs(constraint.rhs() - constraint.offset());
            constraint.set_offset(0);
            return true;   // we made changes
        }

        return false; // no changes made
    }

    /// Remove any self-loops (e.g. x^2) by adding a new variable and an equality constraint.
    bool normalization_remove_self_loops() {
        std::unordered_map<index_type, index_type> mapping;

        // Function to go through the objective/constraints and for each variable in a self-loop,
        // we add a new variable it can interact with.
        // We could break this out into a standalone method, but because all of  those methods
        // would need to share a common mapping, there isn't really an opportunity for
        // parallization. So let's keep it contained for now.
        auto substitute = [&](expression_type& expression) {
            std::size_t current_num_variables = expression.num_variables();
            for (std::size_t i = 0; i < current_num_variables; ++i) {
                index_type v = expression.variables()[i];

                if (!expression.has_interaction(v, v)) {
                    // no self-loop
                    continue;
                }

                auto it = mapping.find(v);
                if (it == mapping.end()) {
                    // we haven't seen this variable before
                    it = mapping.emplace_hint(
                            it, v,
                            model_.add_variable(model_.vartype(v), model_.lower_bound(v),
                                                model_.upper_bound(v)));
                }

                assert(it != mapping.end());

                // Set the bias between v and the new variable
                // We could spread the linear bias out between the two variables, but in that case
                // we would probably want to do that in every expression for consistency and this
                // is a lot easier. But it's something we could test in the future
                expression.add_quadratic(v, it->second, expression.quadratic(v, v));
                expression.remove_interaction(v, v);
            }
        };

        // Actually apply the method
        substitute(model_.objective());
        for (auto& constraint : model_.constraints()) {
            substitute(constraint);
        }

        // We add the new constraints last, because otherwise we would cause reallocation
        for (auto& uv : mapping) {
            model_.add_linear_constraint({uv.first, uv.second}, {+1, -1}, dimod::Sense::EQ, 0);
        }

        return mapping.size();
    }

    /// Replace any inf with 1e30
    bool normalization_replace_inf() {
        bool changes = normalization_replace_inf(model_.objective());
        for (auto& constraint : model_.constraints()) {
            changes |= normalization_replace_inf(constraint);
        }

        // Bounds will all be tightened by fix_bounds later, but for
        // maintainability and symmetry, let's go ahead and replace those too
        for (size_type v = 0, last = model_.num_variables(); v < last; ++v) {
            if (std::isinf(model_.lower_bound(v))) {
                model_.set_lower_bound(v, (model_.lower_bound(v) > 0) ? INF : -INF);
                changes = true;
            }

            if (std::isinf(model_.upper_bound(v))) {
                model_.set_upper_bound(v, (model_.upper_bound(v) > 0) ? INF : -INF);
                changes = true;
            }
        }

        return changes;
    }

    static bool normalization_replace_inf(constraint_type& constraint) {
        // test the biases and offset
        bool changes = normalization_replace_inf(static_cast<expression_type&>(constraint));

        // test the rhs
        if (std::isinf(constraint.rhs())) {
            constraint.set_rhs((constraint.rhs() > 0) ? INF : -INF);
            changes = true;
        }

        // infinite constraint.weight() is just fine, it's the default!

        return changes;
    }

    static bool normalization_replace_inf(expression_type& expression) {
        bool changes = false;

        // We only care about the biases, so let's just cast to the base type for speed
        dimod::abc::QuadraticModelBase<bias_type, index_type>& base = expression;

        // Replace a bias with +/- INF based on the sign
        auto replace = [&](bias_type bias) { return (bias > 0) ? INF : -INF; };

        for (auto it = base.cbegin_quadratic(), end = base.cend_quadratic(); it != end; ++it) {
            if (std::isinf(it->bias)) {
                base.set_quadratic(it->u, it->v, replace(it->bias));  // does not reallocate
                changes = true;
            }
        }

        for (size_type v = 0, last = base.num_variables(); v < last; ++v) {
            if (std::isinf(base.linear(v))) {
                base.set_linear(v, replace(base.linear(v)));
                changes = true;
            }
        }

        if (std::isinf(base.offset())) {
            base.set_offset(replace(base.offset()));
            changes = true;
        }

        return changes;
    }

    /// Convert any SPIN variables to BINARY variables
    bool normalization_spin_to_binary() {
        bool changes = false;
        for (size_type v = 0; v < model_.num_variables(); ++v) {
            if (model_.vartype(v) == dimod::Vartype::SPIN) {
                model_.change_vartype(dimod::Vartype::BINARY, v);
                changes = true;
            }
        }
        return changes;
    }

    bool presolve(std::chrono::duration<double> time_limit =
                          std::chrono::duration<double>(std::numeric_limits<double>::infinity())) {
        if (detached_) {
            throw std::logic_error(
                    "model has been detached, so there is no model to apply presolve() to");
        }
        if (!normalized_) {
            throw std::logic_error("model must be normalized before presolve() is applied");
        }

        // If no techniques have been loaded, return early.
        if (!techniques) {
            return false;
        }

        bool changes = false;

        // We have many exit criteria. We could put them all in the for-loop
        // declaration but that makes it heard to read so we put them into
        // separate if/break statements.
        const auto start_time = std::chrono::steady_clock::now();
        for (index_type num_rounds = 0; num_rounds < max_num_rounds; ++num_rounds) {
            // No point doing presolve if we're infeasible
            if (feasibility() == Feasibility::Infeasible) break;

            // If we've exceeded the time_limit then don't proceed
            if (std::chrono::steady_clock::now() - start_time >= time_limit) break;

            bool loop_changes = false;

            // Objective
            {
                if (techniques & TechniqueFlags::RemoveSmallBiases) {
                    loop_changes |= technique_remove_small_biases(model_.objective());
                }

                if (std::chrono::steady_clock::now() - start_time >= time_limit) break;
            }

            // Constraints
            for (auto& constraint : model_.constraints()) {
                if (techniques & TechniqueFlags::RemoveSmallBiases) {
                    loop_changes |= technique_remove_small_biases(constraint);
                }

                if (techniques & TechniqueFlags::DomainPropagation) {
                    loop_changes |= technique_domain_propagation(constraint);
                }

                if (techniques & TechniqueFlags::RemoveRedundantConstraints) {
                    loop_changes |= technique_clear_redundant_constraint(constraint);
                }

                // this will ultimately give us a double break because we'll test again in the main
                // loop
                if (std::chrono::steady_clock::now() - start_time >= time_limit) break;
            }

            // If we didn't make any changes, then doing more loops won't help
            // so we exit out
            if (!loop_changes) break;

            changes |= loop_changes;
        }

        // Cleanup. We do these steps even in the infeasible case.

        // These steps are not broken out into methods because we only want to do them here.
        // Otherwise we may reallocate vectors we don't want to reallocate.

        /// Remove variables that are fixed by their bounds
        {
            std::vector<index_type> variables;
            std::vector<assignment_type> values;

            for (size_type v = 0; v < model_.num_variables(); ++v) {
                if (model_.lower_bound(v) == model_.upper_bound(v)) {
                    variables.emplace_back(v);
                    values.emplace_back(model_.lower_bound(v));
                }
            }

            model_.fix_variables(variables, values);

            changes |= variables.size();

            // we may have introduced offsets here, so let's fix them
            changes |= normalization_remove_offsets();
        }

        // Remove any constraints without any variables, their contribution to
        // infeasibility or whatever is baked in so no point carrying them around
        {
            const size_type num_constraints = model_.num_constraints();
            model_.remove_constraints_if(
                    [](const constraint_type& c) { return !c.num_variables(); });
            changes |= num_constraints > model_.num_constraints();  // if we removed any
        }

        // There are a few normalization steps we want to re-run to clean up the model
        // e.g. discrete variable markers may not be valid anymore.
        // Todo: we could replace this with a step where we can also add discrete markers
        changes |= normalization_remove_invalid_markers();

        // Sanity check that we didn't break normalization
        assert(!normalize());

        return changes;
    }

    /// Return a sample of the original CQM from a sample of the reduced CQM.
    template<class T>
    std::vector<T> restore(std::vector<T> reduced) const {
        return model_.restore(std::move(reduced));
    }

    /// Clear redundant constraints by turning them into 0 == 0 constraints.
    /// We don't actually remove them (yet) because we don't want to reallocate
    /// our constraint vector.
    bool technique_clear_redundant_constraint(constraint_type& constraint) {
        // This method is not (currently, todo) implemented for quadratic constraints
        if (!constraint.is_linear()) {
            return false;
        }

        // Skip the constraints that have already been cleared
        if (!constraint.num_variables() && !constraint.offset() && !constraint.rhs()) {
            // the constraint is already empty, nothing to do
            return false;
        }

        bias_type minac = minimal_activity(constraint);
        bias_type maxac = maximal_activity(constraint);

        // Test if the constraint is trivially infeasible
        if (constraint.sense() == dimod::Sense::LE || constraint.sense() == dimod::Sense::EQ) {
            if (minac > constraint.rhs() + FEASIBILITY_TOLERANCE) {
                // Soft constraints don't cause the model to be infeasible.
                if (!constraint.is_soft()) {
                    model_.feasibility = Feasibility::Infeasible;
                }

                return false;
            }
        }
        if (constraint.sense() == dimod::Sense::GE || constraint.sense() == dimod::Sense::EQ) {
            if (maxac < constraint.rhs() - FEASIBILITY_TOLERANCE) {
                // Soft constraints don't cause the model to be infeasible.
                if (!constraint.is_soft()) {
                    model_.feasibility = Feasibility::Infeasible;
                }

                return false;
            }
        }

        // Test if the constraint is always satisfied
        // If the constraint is soft, it can just be remove it because it will
        // contribute 0 to the energy of the problem.
        switch (constraint.sense()) {
            case dimod::Sense::LE:
                if (maxac <= constraint.rhs() + FEASIBILITY_TOLERANCE) {
                    constraint.clear();
                    return true;
                }
                break;
            case dimod::Sense::GE:
                if (minac >= constraint.rhs() - FEASIBILITY_TOLERANCE) {
                    constraint.clear();
                    return true;
                }
                break;
            case dimod::Sense::EQ:
                if (minac == constraint.rhs() && maxac == constraint.rhs()) {
                    constraint.clear();
                    return true;
                }
                break;
        }

        return false;
    }

    /// Tighten bounds based on constraints.
    /// See Achterberg et al., section 3.2.
    bool technique_domain_propagation(const constraint_type& constraint) {
        bool changes = false;

        if (constraint.sense() == dimod::Sense::LE || constraint.sense() == dimod::Sense::EQ) {
            changes |= technique_domain_propagation<dimod::Sense::LE>(constraint);
        }

        // We don't need to handle GE, but it makes testing easier so may as well
        if (constraint.sense() == dimod::Sense::GE || constraint.sense() == dimod::Sense::EQ) {
            changes |= technique_domain_propagation<dimod::Sense::GE>(constraint);
        }

        return changes;
    }

    /// Remove small biases from an expression.
    /// See Achterberg et al., section 3.1.
    static bool technique_remove_small_biases(expression_type& expression) {
        bool changes = false;

        // For general Expressions (as opposed to constraints) we remove
        // biases that are smaller than 1e-10
        static constexpr double UNCONDITIONAL_REMOVAL_BIAS_LIMIT = 1.0e-10;

        // We'll want to traverse the expression in index order, so let's
        // work with its base class when doing so
        expression_base_type& base = expression;

        // We remove quadratic interactions with biases smaller than 1e-10
        changes |= expression.remove_interactions([](index_type, index_type, bias_type bias) {
            return std::abs(bias) < UNCONDITIONAL_REMOVAL_BIAS_LIMIT;
        });

        // We zero linear biases that are smaller than 1e-10 and we remove the
        // variable from the expression if its linear bias is small and it
        // has no quadratic interactions
        std::vector<index_type> variables;
        for (size_type vi = 0; vi < base.num_variables(); ++vi) {
            if (std::abs(base.linear(vi)) < UNCONDITIONAL_REMOVAL_BIAS_LIMIT) {
                if (base.num_interactions(vi)) {
                    // it has at least one quadratic interaction
                    base.set_linear(vi, 0);
                    changes = true;
                } else {
                    // it's linear-only, so we can just remove it
                    variables.emplace_back(expression.variables()[vi]);
                }
            }
        }
        expression.remove_variables(variables.begin(), variables.end());
        changes |= variables.size();

        return changes;
    }

    /// Remove small biases from a constraint.
    /// See Achterberg et al., section 3.1.
    static bool technique_remove_small_biases(constraint_type& constraint) {
        // First we do the unconditional removals (valid for objective and constraints)
        bool changes = technique_remove_small_biases(static_cast<expression_type&>(constraint));

        // For constraints, we can do conditional removals
        static constexpr double CONDITIONAL_REMOVAL_BIAS_LIMIT = 1.0e-3;
        static constexpr double CONDITIONAL_REMOVAL_LIMIT = 1.0e-2 * FEASIBILITY_TOLERANCE;

        // We'll want to traverse the expression in index order, so let's
        // work with its base class when doing so
        expression_base_type& base = constraint;

        std::vector<index_type> variables;
        for (size_type vi = 0; vi < base.num_variables(); ++vi) {
            // skip the variables that have quadratic interactions
            if (base.num_interactions(vi)) {
                continue;
            }

            const bias_type a = base.linear(vi);
            const bias_type lb = base.lower_bound(vi);
            const bias_type ub = base.upper_bound(vi);

            if (std::abs(a) < CONDITIONAL_REMOVAL_BIAS_LIMIT &&
                std::abs(a) * (ub - lb) * base.num_variables() < CONDITIONAL_REMOVAL_LIMIT) {
                variables.emplace_back(constraint.variables()[vi]);
                constraint.set_rhs(constraint.rhs() - a * lb);
            }
        }
        constraint.remove_variables(variables.begin(), variables.end());

        // NB: Achterberg et al. does an additional step where they scan the row
        // and remove coefficients as long as the sum stays below a certain value.
        // However, we don't want to do that here because this method gets called
        // multiple times.

        return changes || variables.size();
    }

    /// The maximum number of rounds of presolving
    int max_num_rounds = 100;

    TechniqueFlags techniques = TechniqueFlags::Default;

 private:
    // We want to control access to the model in order to track changes,
    // so we create a rump version of the model.
    class ModelView : private model_type {
     public:
        ModelView() = default;
        ~ModelView() = default;

        explicit ModelView(model_type&& model) : model_type(model) {}

        // Const methods are all safe to expose
        using model_type::num_constraints;
        using model_type::num_variables;
        using model_type::lower_bound;
        using model_type::upper_bound;
        using model_type::vartype;

        // We can expose the expression access methods because changes to
        // the individual expressions don't get tracked
        using model_type::constraint_ref;
        using model_type::constraints;

        // Adding/removing constraints isn't tracked
        using model_type::add_linear_constraint;
        using model_type::remove_constraints_if;

        // The bound changes don't get tracked, but we do want to maintain normalization
        // and test for feasibility
        bool set_lower_bound(index_type v, bias_type lb) {
            // handle the discrete vartypes
            switch (vartype(v)) {
                case dimod::Vartype::BINARY:
                case dimod::Vartype::SPIN:
                case dimod::Vartype::INTEGER:
                    lb = std::ceil(lb);
                case dimod::Vartype::REAL:
                    break;
            }

            if (lb > upper_bound(v)) {
                // infeasible, set flag but don't set.
                feasibility = Feasibility::Infeasible;
                return false;
            }

            if (lb > lower_bound(v)) {
                model_type::set_lower_bound(v, lb);
                return true;
            }

            return false;
        }
        bool set_upper_bound(index_type v, bias_type ub) {
            // handle the discrete vartypes
            switch (vartype(v)) {
                case dimod::Vartype::BINARY:
                case dimod::Vartype::SPIN:
                case dimod::Vartype::INTEGER:
                    ub = std::floor(ub);
                case dimod::Vartype::REAL:
                    break;
            }

            if (lower_bound(v) > ub) {
                // infeasible, set flag but don't set.
                feasibility = Feasibility::Infeasible;
                return false;
            }

            if (ub < upper_bound(v)) {
                model_type::set_upper_bound(v, ub);
                return true;
            }

            return false;
        }

        // Expose the objective. Changes don't get tracked
        auto& objective() { return static_cast<model_type*>(this)->objective; }
        const auto& objective() const { return static_cast<const model_type*>(this)->objective; }

        // Expose the model as a const reference
        const model_type& model() const { return *this; }

        // Clear the model, but not the transforms queue.
        model_type detach_model() {
            using std::swap;  // ADL, though doubt it makes a difference
            auto cqm = dimod::ConstrainedQuadraticModel<bias_type, index_type>();
            swap(*this, cqm);
            return cqm;
        }

        // Track when we add a variable to the model.
        index_type add_variable(dimod::Vartype vartype, bias_type lb, bias_type ub) {
            index_type v = model_type::add_variable(vartype, lb, ub);
            transforms_.emplace_back(TransformKind::ADD);
            transforms_.back().v = v;
            return v;
        }

        // Track when we change the vartype of a variable
        void change_vartype(dimod::Vartype vartype, index_type v) {
            if ((model_type::vartype(v) == dimod::Vartype::SPIN) &&
                (vartype == dimod::Vartype::BINARY)) {
                // SPIN->BINARY
                transforms_.emplace_back(TransformKind::SUBSTITUTE);
                transforms_.back().v = v;
                transforms_.back().multiplier = 2;
                transforms_.back().offset = -1;
            } else {
                // We currently only need SPIN->BINARY, but can add more as needed
                throw std::logic_error("unsupported vartype change");
            }
            model_type::change_vartype(vartype, v);
        }

        // Track the variables that are fixed.
        // Require that the variables are in sorted order.
        void fix_variables(const std::vector<index_type>& variables,
                           const std::vector<assignment_type>& assignments) {
            assert(variables.size() == assignments.size());

            // short circuit in the case there's nothing to fix, thereby avoiding
            // an expensive copy.
            if (variables.empty()) {
                return;
            }

            // track the changes as if we had applied them one-by-one, but do it
            // in reverse order so that when we're restoring we're pushing onto the end
            assert(std::is_sorted(variables.begin(), variables.end()));
            auto vit = variables.rbegin();
            const auto last = variables.rend();
            auto ait = assignments.rbegin();
            for (; vit != last; ++vit, ++ait) {
                transforms_.emplace_back(TransformKind::FIX);
                transforms_.back().v = (*vit);
                transforms_.back().value = *ait;
            }

            /// fix the variables in-place.
            using std::swap;  // ADL, though doubt it makes a difference
            auto cqm = model_type::fix_variables(variables.begin(), variables.end(),
                                                 assignments.begin());
            swap(*this, cqm);
        }

        // Restore a sample by undoing all of the transforms
        template <class T>
        std::vector<T> restore(std::vector<T> sample) const {
            // all that we have to do is undo the transforms back to front.
            for (auto it = transforms_.crbegin(); it != transforms_.crend(); ++it) {
                switch (it->kind) {
                    case TransformKind::FIX:
                        sample.insert(sample.begin() + it->v, it->value);
                        break;
                    case TransformKind::SUBSTITUTE:
                        sample[it->v] *= it->multiplier;
                        sample[it->v] += it->offset;
                        break;
                    case TransformKind::ADD:
                        sample.erase(sample.begin() + it->v);
                        break;
                }
            }
            return sample;
        }

        Feasibility feasibility = Feasibility::Unknown;

     private:
        // we want to track what changes were made
        enum TransformKind { FIX, SUBSTITUTE, ADD };

        // We could get fancy with pointers and templates to save a bit of space here
        struct Transform {
            TransformKind kind;
            index_type v;           // what variable it was applied to
            assignment_type value;  // if it was fixed what it was fixed to
            bias_type multiplier;
            bias_type offset;

            explicit Transform(TransformKind kind)
                    : kind(kind), v(-1), value(NAN), multiplier(NAN), offset(NAN) {}
        };

        std::vector<Transform> transforms_;
    };

    template <dimod::Sense Sense>
    bool technique_domain_propagation(const constraint_type& constraint) {
        // Dev note: if-branche(s) below rely on this static assert
        static_assert(Sense == dimod::Sense::GE || Sense == dimod::Sense::LE,
                      "Sense must be <= or >=; equality constraints should call both");

        assert(constraint.sense() == dimod::Sense::EQ || constraint.sense() == Sense);

        // todo: extend to quadratic
        if (!constraint.is_linear()) {
            return false;  // no changes
        }

        // Cannot use soft constraints to strengthen bounds.
        if (constraint.is_soft()) {
            return false;  // no changes
        }

        // In order to not fall into zeno's paradox here, we don't shrink bounds when
        // the reduction is small.
        static constexpr double BOUND_CHANGE_MINIMUM = 1.0e-3 * FEASIBILITY_TOLERANCE;

        // Very large activities will create numeric issues. So we don't do domain
        // propagation on those
        static constexpr double MAX_ACTIVITY = 1.0e10;

        // In the following code there are three cases we want to account for:
        // - No large activities: we can do domain propagation on all of the variables
        // - Exactly one large activity: we can do domain propagation on the one variable with
        //   large activity
        // - 2+ large activities: we can't do domain propagation

        // Precalculate the activities of each of the variables
        std::vector<bias_type> activities;
        std::vector<index_type> large_activities;
        for (const auto& v : constraint.variables()) {
            if constexpr (Sense == dimod::Sense::LE) {
                activities.emplace_back(minimal_activity(constraint, v));
            } else {  // Sense == dimod::Sense::GE, enforced by static_assert above
                activities.emplace_back(maximal_activity(constraint, v));
            }

            if (std::abs(activities.back()) > MAX_ACTIVITY) {
                large_activities.emplace_back(v);

                // Early exit test: in the case that we have more than one large
                // activity, we cannot do domain propagation at all
                // Everything will work without this test, but it can save some time
                if (large_activities.size() > 1) {
                    return false;  // no changes
                }

                // Since we've marked it as large, let's zero it out to make
                // some of the sums later a bit safer
                activities.back() = 0;
            }
        }

        // Get the total activity of everything (except the large activity variables)
        const bias_type activity =
                std::reduce(activities.begin(), activities.end()) + constraint.offset();

        // If, even excluding large activities, we get a large activity we can't do anything
        if (std::abs(activity) > INF) {
            return false;  // no changes
        }

        // Create a function to do domain propagation given the activity of the rest
        // of the constraint for a single variable
        auto propagate = [&](index_type v, bias_type activity_excluding_v) {
            bias_type a = constraint.linear(v);

            // get the new bound
            const bias_type bound = (constraint.rhs() - activity_excluding_v) / a;

            // For >=, we need to flip the sign of a to determine which bound we affect
            if constexpr (Sense == dimod::Sense::GE) {
                a *= -1;
            }

            bool changes = false;
            if (a > 0 && model_.upper_bound(v) - bound > BOUND_CHANGE_MINIMUM) {
                changes |= model_.set_upper_bound(v, bound);  // handles vartype and feasibility
            } else if (a < 0 && bound - model_.lower_bound(v) > BOUND_CHANGE_MINIMUM) {
                changes |= model_.set_lower_bound(v, bound);  // handles vartype and feasibility
            }
            return changes;
        };

        // Alright, we're finally able to actually do domain propagation

        bool changes = false;

        if (large_activities.size() == 0) {
            // we can do domain propagation on every variable
            auto it = activities.begin();
            for (const auto& v : constraint.variables()) {
                // Just subtract out the activity of v
                const bias_type activity_excluding_v = activity - *it;

                changes |= propagate(v, activity_excluding_v);
                ++it;
            }
        } else if (large_activities.size() == 1) {
            // We can only do domain propagation on the variable with large bias
            const auto& v = large_activities.back();

            // we already zeroed out the activity associated with v, so this
            // is just the "activity"
            const bias_type activity_excluding_v = activity;

            changes |= propagate(v, activity_excluding_v);
        }  // else large_activities.size() > 1, do nothing. Also we should have already exited early

        return changes;
    }

    ModelView model_;

    bool detached_ = false;
    bool normalized_ = false;
};
}  // namespace presolve
}  // namespace dwave
