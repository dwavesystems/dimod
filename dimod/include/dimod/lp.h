// The LP-File parsing code (under the dimod::lp::reader namespace)
// is adapted from HiGHS https://github.com/ERGO-Code/HiGHS/tree/501b30a
// See the license at the beginning of that namespace.
//
// Any modifications to that code, or any code not under the dimod::lp::reader
// namespace is licensed under:
//
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

#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dimod/quadratic_model.h"

namespace dimod {
namespace lp {

template <class Bias>
struct Variable {
    Vartype vartype;
    std::string name;
    Bias lower_bound;
    Bias upper_bound;

    Variable(Vartype vartype_, std::string name_) : vartype(vartype_), name(name_) {
        this->lower_bound = vartype_info<Bias>::default_min(vartype_);
        this->upper_bound = vartype_info<Bias>::default_max(vartype_);
    }
};

template <class Bias, class Index>
struct Expression {
    dimod::QuadraticModel<Bias, Index> model;
    std::unordered_map<std::string, Index> labels;
    std::string name = "";

    // add the variable and get back the index
    Index add_variable(const Variable<Bias>& variable) {
        auto out = this->labels.emplace(variable.name, this->labels.size());

        if (out.second) {
            this->model.add_variable(variable.vartype, variable.lower_bound, variable.upper_bound);
        }

        // todo: make assert
        if (this->model.num_variables() != this->labels.size()) {
            throw std::logic_error("something went wrong");
        }

        return out.first->second;
    }
};

template <class Bias, class Index>
struct Constraint {
    Expression<Bias, Index> lhs;
    std::string sense;
    Bias rhs;
};

template <class Bias, class Index>
struct LPModel {
    Expression<Bias, Index> objective;
    std::vector<Constraint<Bias, Index>> constraints;
    bool minimize = true;
};

namespace reader {
// The LP-File parsing code (under the dimod::lp::reader namespace)
// is adapted from HiGHS https://github.com/ERGO-Code/HiGHS/tree/501b30a
// under the following license:
//
// Copyright (c) 2020 Michael Feldmeier
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

void inline lpassert(bool condition) {
    if (!condition) {
        throw std::invalid_argument("File not existant or illegal file format.");
    }
}

const unsigned int LP_MAX_NAME_LENGTH = 255;
const unsigned int LP_MAX_LINE_LENGTH = 560;

const std::string LP_KEYWORD_MIN[] = {"minimize", "min", "minimum"};
const std::string LP_KEYWORD_MAX[] = {"maximize", "max", "maximum"};
const std::string LP_KEYWORD_ST[] = {"subject to", "such that", "st", "s.t."};
const std::string LP_KEYWORD_BOUNDS[] = {"bounds", "bound"};
const std::string LP_KEYWORD_INF[] = {"infinity", "inf"};
const std::string LP_KEYWORD_FREE[] = {"free"};
const std::string LP_KEYWORD_GEN[] = {"general", "generals", "gen"};
const std::string LP_KEYWORD_BIN[] = {"binary", "binaries", "bin"};
const std::string LP_KEYWORD_SEMI[] = {"semi-continuous", "semi", "semis"};
const std::string LP_KEYWORD_SOS[] = {"sos"};
const std::string LP_KEYWORD_END[] = {"end"};

const unsigned int LP_KEYWORD_MIN_N = 3;
const unsigned int LP_KEYWORD_MAX_N = 3;
const unsigned int LP_KEYWORD_ST_N = 4;
const unsigned int LP_KEYWORD_BOUNDS_N = 2;
const unsigned int LP_KEYWORD_INF_N = 2;
const unsigned int LP_KEYWORD_FREE_N = 1;
const unsigned int LP_KEYWORD_GEN_N = 3;
const unsigned int LP_KEYWORD_BIN_N = 3;
const unsigned int LP_KEYWORD_SEMI_N = 3;
const unsigned int LP_KEYWORD_SOS_N = 1;
const unsigned int LP_KEYWORD_END_N = 1;

enum class RawTokenType {
    NONE,
    STR,
    CONS,
    LESS,
    GREATER,
    EQUAL,
    COLON,
    LNEND,
    FLEND,
    BRKOP,
    BRKCL,
    PLUS,
    MINUS,
    HAT,
    SLASH,
    ASTERISK
};

struct RawToken {
    RawTokenType type;
    inline bool istype(RawTokenType t) { return this->type == t; }
    RawToken(RawTokenType t) : type(t){};
};

struct RawStringToken : RawToken {
    std::string value;
    RawStringToken(std::string v) : RawToken(RawTokenType::STR), value(v){};
};

struct RawConstantToken : RawToken {
    double value;
    RawConstantToken(double v) : RawToken(RawTokenType::CONS), value(v){};
};

enum class ProcessedTokenType {
    NONE,
    SECID,
    VARID,
    CONID,
    CONST,
    FREE,
    BRKOP,
    BRKCL,
    COMP,
    LNEND,
    SLASH,
    ASTERISK,
    HAT
};

enum class LpSectionKeyword { NONE, OBJ, CON, BOUNDS, GEN, BIN, SEMI, SOS, END };

enum class LpObjectiveSectionKeywordType { NONE, MIN, MAX };

enum class LpComparisonType { LEQ, L, EQ, G, GEQ };

struct ProcessedToken {
    ProcessedTokenType type;
    ProcessedToken(ProcessedTokenType t) : type(t){};
};

struct ProcessedTokenSectionKeyword : ProcessedToken {
    LpSectionKeyword keyword;
    ProcessedTokenSectionKeyword(LpSectionKeyword k)
            : ProcessedToken(ProcessedTokenType::SECID), keyword(k){};
};

struct ProcessedTokenObjectiveSectionKeyword : ProcessedTokenSectionKeyword {
    LpObjectiveSectionKeywordType objsense;
    ProcessedTokenObjectiveSectionKeyword(LpObjectiveSectionKeywordType os)
            : ProcessedTokenSectionKeyword(LpSectionKeyword::OBJ), objsense(os){};
};

struct ProcessedConsIdToken : ProcessedToken {
    std::string name;
    ProcessedConsIdToken(std::string n) : ProcessedToken(ProcessedTokenType::CONID), name(n){};
};

struct ProcessedVarIdToken : ProcessedToken {
    std::string name;
    ProcessedVarIdToken(std::string n) : ProcessedToken(ProcessedTokenType::VARID), name(n){};
};

struct ProcessedConstantToken : ProcessedToken {
    double value;
    ProcessedConstantToken(double v) : ProcessedToken(ProcessedTokenType::CONST), value(v){};
};

struct ProcessedComparisonToken : ProcessedToken {
    LpComparisonType dir;
    ProcessedComparisonToken(LpComparisonType d)
            : ProcessedToken(ProcessedTokenType::COMP), dir(d){};
};

bool isstrequalnocase(const std::string str1, const std::string str2) {
    size_t len = str1.size();
    if (str2.size() != len) return false;
    for (size_t i = 0; i < len; ++i)
        if (tolower(str1[i]) != tolower(str2[i])) return false;
    return true;
}

bool iskeyword(const std::string str, const std::string* keywords, const int nkeywords) {
    for (int i = 0; i < nkeywords; i++) {
        if (isstrequalnocase(str, keywords[i])) {
            return true;
        }
    }
    return false;
}

LpObjectiveSectionKeywordType parseobjectivesectionkeyword(const std::string str) {
    if (iskeyword(str, LP_KEYWORD_MIN, LP_KEYWORD_MIN_N)) {
        return LpObjectiveSectionKeywordType::MIN;
    }

    if (iskeyword(str, LP_KEYWORD_MAX, LP_KEYWORD_MAX_N)) {
        return LpObjectiveSectionKeywordType::MAX;
    }

    return LpObjectiveSectionKeywordType::NONE;
}

LpSectionKeyword parsesectionkeyword(const std::string& str) {
    if (parseobjectivesectionkeyword(str) != LpObjectiveSectionKeywordType::NONE) {
        return LpSectionKeyword::OBJ;
    }

    if (iskeyword(str, LP_KEYWORD_ST, LP_KEYWORD_ST_N)) {
        return LpSectionKeyword::CON;
    }

    if (iskeyword(str, LP_KEYWORD_BOUNDS, LP_KEYWORD_BOUNDS_N)) {
        return LpSectionKeyword::BOUNDS;
    }

    if (iskeyword(str, LP_KEYWORD_BIN, LP_KEYWORD_BIN_N)) {
        return LpSectionKeyword::BIN;
    }

    if (iskeyword(str, LP_KEYWORD_GEN, LP_KEYWORD_GEN_N)) {
        return LpSectionKeyword::GEN;
    }

    if (iskeyword(str, LP_KEYWORD_SEMI, LP_KEYWORD_SEMI_N)) {
        return LpSectionKeyword::SEMI;
    }

    if (iskeyword(str, LP_KEYWORD_SOS, LP_KEYWORD_SOS_N)) {
        return LpSectionKeyword::SOS;
    }

    if (iskeyword(str, LP_KEYWORD_END, LP_KEYWORD_END_N)) {
        return LpSectionKeyword::END;
    }

    return LpSectionKeyword::NONE;
}

template <class Bias, class Index>
class Reader {
 public:
    using bias_type = Bias;
    using index_type = Index;

 private:
    FILE* file;
    std::vector<std::unique_ptr<RawToken>> rawtokens;
    std::vector<std::unique_ptr<ProcessedToken>> processedtokens;
    std::map<LpSectionKeyword, std::vector<std::unique_ptr<ProcessedToken>>> sectiontokens;

    char linebuffer[LP_MAX_LINE_LENGTH + 1];
    char* linebufferpos;
    bool linefullyread;
    bool newline_encountered;

    LPModel<bias_type, index_type> model_;
    std::unordered_map<std::string, Variable<bias_type>> variables_;

 public:
    Reader(const std::string filename) : file(fopen(filename.c_str(), "r")) {
        if (file == nullptr) {
            throw std::invalid_argument("given file cannot be opened");
        };
    };

    ~Reader() { fclose(file); }

    LPModel<bias_type, index_type> read() {
        tokenize();
        processtokens();
        splittokens();
        processsections();

        return this->model_;
    }

 private:
    Variable<bias_type>& variable(const std::string name) {
        auto out = this->variables_.emplace(name, Variable<bias_type>(Vartype::REAL, name));
        return out.first->second;
    }

    void processnonesec() {
        if (!sectiontokens[LpSectionKeyword::NONE].empty()) {
            throw std::logic_error("dimod does not support None sections");
        }
    }

    void parseexpression(std::vector<std::unique_ptr<ProcessedToken>>& tokens,
                         Expression<bias_type, index_type>& expr, unsigned int& i, bool isobj) {
        if (tokens.size() - i >= 1 && tokens[0]->type == ProcessedTokenType::CONID) {
            expr.name = ((ProcessedConsIdToken*)tokens[i].get())->name;
            i++;
        }

        while (i < tokens.size()) {
            // const var
            if (tokens.size() - i >= 2 && tokens[i]->type == ProcessedTokenType::CONST &&
                tokens[i + 1]->type == ProcessedTokenType::VARID) {
                std::string name = ((ProcessedVarIdToken*)tokens[i + 1].get())->name;
                bias_type coef = ((ProcessedConstantToken*)tokens[i].get())->value;

                index_type vi = expr.add_variable(this->variable(name));
                expr.model.linear(vi) += coef;

                i += 2;
                continue;
            }

            // const
            if (tokens.size() - i >= 1 && tokens[i]->type == ProcessedTokenType::CONST) {
                expr.model.offset() += ((ProcessedConstantToken*)tokens[i].get())->value;
                i++;
                continue;
            }

            // var
            if (tokens.size() - i >= 1 && tokens[i]->type == ProcessedTokenType::VARID) {
                std::string name = ((ProcessedVarIdToken*)tokens[i].get())->name;

                index_type vi = expr.add_variable(this->variable(name));
                expr.model.linear(vi) += 1;

                i++;
                continue;
            }

            // quadratic expression
            if (tokens.size() - i >= 2 && tokens[i]->type == ProcessedTokenType::BRKOP) {
                i++;
                while (i < tokens.size() && tokens[i]->type != ProcessedTokenType::BRKCL) {
                    // const var hat const
                    if (tokens.size() - i >= 4 && tokens[i]->type == ProcessedTokenType::CONST &&
                        tokens[i + 1]->type == ProcessedTokenType::VARID &&
                        tokens[i + 2]->type == ProcessedTokenType::HAT &&
                        tokens[i + 3]->type == ProcessedTokenType::CONST) {
                        std::string name = ((ProcessedVarIdToken*)tokens[i + 1].get())->name;
                        bias_type coef = ((ProcessedConstantToken*)tokens[i].get())->value;

                        // in the objective the quadratic terms are all in [ ... ] / 2
                        if (isobj) {
                            coef /= 2;
                        }

                        lpassert(((ProcessedConstantToken*)tokens[i + 3].get())->value == 2.0);

                        index_type vi = expr.add_variable(this->variable(name));
                        expr.model.add_quadratic(vi, vi, coef);

                        i += 4;
                        continue;
                    }

                    // var hat const
                    if (tokens.size() - i >= 3 && tokens[i]->type == ProcessedTokenType::VARID &&
                        tokens[i + 1]->type == ProcessedTokenType::HAT &&
                        tokens[i + 2]->type == ProcessedTokenType::CONST) {
                        std::string name = ((ProcessedVarIdToken*)tokens[i].get())->name;
                        bias_type coef = 1;

                        // in the objective the quadratic terms are all in [ ... ] / 2
                        if (isobj) {
                            coef /= 2;
                        }

                        lpassert(((ProcessedConstantToken*)tokens[i + 2].get())->value == 2.0);

                        index_type vi = expr.add_variable(this->variable(name));
                        expr.model.add_quadratic(vi, vi, coef);

                        i += 3;
                        continue;
                    }

                    // const var asterisk var
                    if (tokens.size() - i >= 4 && tokens[i]->type == ProcessedTokenType::CONST &&
                        tokens[i + 1]->type == ProcessedTokenType::VARID &&
                        tokens[i + 2]->type == ProcessedTokenType::ASTERISK &&
                        tokens[i + 3]->type == ProcessedTokenType::VARID) {
                        std::string name1 = ((ProcessedVarIdToken*)tokens[i + 1].get())->name;
                        std::string name2 = ((ProcessedVarIdToken*)tokens[i + 3].get())->name;
                        bias_type coef = ((ProcessedConstantToken*)tokens[i].get())->value;

                        // in the objective the quadratic terms are all in [ ... ] / 2
                        if (isobj) {
                            coef /= 2;
                        }

                        index_type ui = expr.add_variable(this->variable(name1));
                        index_type vi = expr.add_variable(this->variable(name2));
                        expr.model.add_quadratic(ui, vi, coef);

                        i += 4;
                        continue;
                    }

                    // var asterisk var
                    if (tokens.size() - i >= 3 && tokens[i]->type == ProcessedTokenType::VARID &&
                        tokens[i + 1]->type == ProcessedTokenType::ASTERISK &&
                        tokens[i + 2]->type == ProcessedTokenType::VARID) {
                        std::string name1 = ((ProcessedVarIdToken*)tokens[i].get())->name;
                        std::string name2 = ((ProcessedVarIdToken*)tokens[i + 2].get())->name;
                        bias_type coef = 1;

                        // in the objective the quadratic terms are all in [ ... ] / 2
                        if (isobj) {
                            coef /= 2;
                        }

                        index_type ui = expr.add_variable(this->variable(name1));
                        index_type vi = expr.add_variable(this->variable(name2));
                        expr.model.add_quadratic(ui, vi, coef);

                        i += 3;
                        continue;
                    }
                    break;
                }
                if (isobj) {
                    // only in the objective function, a quadratic term is followed by "/2.0"
                    lpassert(tokens.size() - i >= 3);
                    lpassert(tokens[i]->type == ProcessedTokenType::BRKCL);
                    lpassert(tokens[i + 1]->type == ProcessedTokenType::SLASH);
                    lpassert(tokens[i + 2]->type == ProcessedTokenType::CONST);
                    lpassert(((ProcessedConstantToken*)tokens[i + 2].get())->value == 2.0);
                    i += 3;
                } else {
                    lpassert(tokens.size() - i >= 1);
                    lpassert(tokens[i]->type == ProcessedTokenType::BRKCL);
                    i += 1;
                }
                continue;
            }

            break;
        }
    }

    void processobjsec() {
        // builder.model.objective = std::shared_ptr<Expression>(new Expression);
        unsigned int i = 0;
        parseexpression(sectiontokens[LpSectionKeyword::OBJ], model_.objective, i, true);
        lpassert(i == sectiontokens[LpSectionKeyword::OBJ].size());
    }

    void processconsec() {
        unsigned int i = 0;
        while (i < sectiontokens[LpSectionKeyword::CON].size()) {
            Constraint<bias_type, index_type> constraint;

            parseexpression(sectiontokens[LpSectionKeyword::CON], constraint.lhs, i, false);

            lpassert(sectiontokens[LpSectionKeyword::CON].size() - i >= 2);
            lpassert(sectiontokens[LpSectionKeyword::CON][i]->type == ProcessedTokenType::COMP);
            lpassert(sectiontokens[LpSectionKeyword::CON][i + 1]->type ==
                     ProcessedTokenType::CONST);

            constraint.rhs =
                    ((ProcessedConstantToken*)sectiontokens[LpSectionKeyword::CON][i + 1].get())
                            ->value;

            switch (((ProcessedComparisonToken*)sectiontokens[LpSectionKeyword::CON][i].get())
                            ->dir) {
                case LpComparisonType::EQ:
                    constraint.sense = "==";
                    break;
                case LpComparisonType::LEQ:
                    constraint.sense = "<=";
                    break;
                case LpComparisonType::GEQ:
                    constraint.sense = ">=";
                    break;
                default:
                    assert(false);
            }
            i += 2;

            this->model_.constraints.push_back(std::move(constraint));
        }
    }

    void processboundssec() {
        unsigned int i = 0;
        while (i < sectiontokens[LpSectionKeyword::BOUNDS].size()) {
            // VAR free
            if (sectiontokens[LpSectionKeyword::BOUNDS].size() - i >= 2 &&
                sectiontokens[LpSectionKeyword::BOUNDS][i]->type == ProcessedTokenType::VARID &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 1]->type == ProcessedTokenType::FREE) {
                std::string name =
                        ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::BOUNDS][i].get())
                                ->name;

                this->variables_.emplace(name, Variable<bias_type>(Vartype::REAL, name));

                i += 2;
                continue;
            }

            // CONST COMP VAR COMP CONST
            if (sectiontokens[LpSectionKeyword::BOUNDS].size() - i >= 5 &&
                sectiontokens[LpSectionKeyword::BOUNDS][i]->type == ProcessedTokenType::CONST &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 1]->type == ProcessedTokenType::COMP &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 2]->type == ProcessedTokenType::VARID &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 3]->type == ProcessedTokenType::COMP &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 4]->type == ProcessedTokenType::CONST) {
                lpassert(((ProcessedComparisonToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 1]
                                  .get())
                                 ->dir == LpComparisonType::LEQ);
                lpassert(((ProcessedComparisonToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 3]
                                  .get())
                                 ->dir == LpComparisonType::LEQ);

                double lb =
                        ((ProcessedConstantToken*)sectiontokens[LpSectionKeyword::BOUNDS][i].get())
                                ->value;
                double ub = ((ProcessedConstantToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 4]
                                     .get())
                                    ->value;

                std::string name =
                        ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 2].get())
                                ->name;

                // add as REAL or get the existing value
                auto out = this->variables_.emplace(name, Variable<bias_type>(Vartype::REAL, name));
                out.first->second.lower_bound = lb;
                out.first->second.upper_bound = ub;

                i += 5;
                continue;
            }

            // CONST COMP VAR
            if (sectiontokens[LpSectionKeyword::BOUNDS].size() - i >= 3 &&
                sectiontokens[LpSectionKeyword::BOUNDS][i]->type == ProcessedTokenType::CONST &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 1]->type == ProcessedTokenType::COMP &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 2]->type == ProcessedTokenType::VARID) {
                double value =
                        ((ProcessedConstantToken*)sectiontokens[LpSectionKeyword::BOUNDS][i].get())
                                ->value;
                std::string name =
                        ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 2].get())
                                ->name;

                // add as REAL or get the existing value
                auto out = this->variables_.emplace(name, Variable<bias_type>(Vartype::REAL, name));

                LpComparisonType dir =
                        ((ProcessedComparisonToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 1]
                                 .get())
                                ->dir;

                lpassert(dir != LpComparisonType::L && dir != LpComparisonType::G);

                switch (dir) {
                    case LpComparisonType::LEQ:
                        out.first->second.lower_bound = value;
                        break;
                    case LpComparisonType::GEQ:
                        out.first->second.upper_bound = value;
                        break;
                    case LpComparisonType::EQ:
                        out.first->second.lower_bound = out.first->second.upper_bound = value;
                        break;
                    default:
                        assert(false);
                }
                i += 3;
                continue;
            }

            // VAR COMP CONST
            if (sectiontokens[LpSectionKeyword::BOUNDS].size() - i >= 3 &&
                sectiontokens[LpSectionKeyword::BOUNDS][i]->type == ProcessedTokenType::VARID &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 1]->type == ProcessedTokenType::COMP &&
                sectiontokens[LpSectionKeyword::BOUNDS][i + 2]->type == ProcessedTokenType::CONST) {
                double value =
                        ((ProcessedConstantToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 2]
                                 .get())
                                ->value;
                std::string name =
                        ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::BOUNDS][i].get())
                                ->name;

                // add as REAL or get the existing value
                auto out = this->variables_.emplace(name, Variable<bias_type>(Vartype::REAL, name));

                LpComparisonType dir =
                        ((ProcessedComparisonToken*)sectiontokens[LpSectionKeyword::BOUNDS][i + 1]
                                 .get())
                                ->dir;

                lpassert(dir != LpComparisonType::L && dir != LpComparisonType::G);

                switch (dir) {
                    case LpComparisonType::LEQ:
                        out.first->second.upper_bound = value;
                        break;
                    case LpComparisonType::GEQ:
                        out.first->second.lower_bound = value;
                        break;
                    case LpComparisonType::EQ:
                        out.first->second.lower_bound = out.first->second.upper_bound = value;
                        break;
                    default:
                        assert(false);
                }
                i += 3;
                continue;
            }

            assert(false);
        }
    }

    void processbinsec() {
        for (unsigned int i = 0; i < sectiontokens[LpSectionKeyword::BIN].size(); i++) {
            lpassert(sectiontokens[LpSectionKeyword::BIN][i]->type == ProcessedTokenType::VARID);
            std::string name =
                    ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::BIN][i].get())->name;

            this->variables_.emplace(name, Variable<bias_type>(Vartype::BINARY, name));
        }
    }

    void processgensec() {
        for (unsigned int i = 0; i < sectiontokens[LpSectionKeyword::GEN].size(); i++) {
            lpassert(sectiontokens[LpSectionKeyword::GEN][i]->type == ProcessedTokenType::VARID);
            std::string name =
                    ((ProcessedVarIdToken*)sectiontokens[LpSectionKeyword::GEN][i].get())->name;

            this->variables_.emplace(name, Variable<bias_type>(Vartype::INTEGER, name));
        }
    }

    void processsemisec() {
        if (!sectiontokens[LpSectionKeyword::SEMI].empty()) {
            throw std::logic_error("dimod does not support semi-continuous variables");
        }
    }

    void processsossec() {
        if (sectiontokens[LpSectionKeyword::SOS].size()) {
            throw std::logic_error("dimod does not support SOS");
        }
    }

    void processendsec() {
        if (!sectiontokens[LpSectionKeyword::END].empty()) {
            throw std::invalid_argument("expected END section to be empty");
        }
    }

    void processsections() {
        // the stuff that's not handled by dimod
        processsemisec();
        processsossec();
        processnonesec();

        // note the binary variables the bounds
        processbinsec();
        processgensec();
        processboundssec();

        // now the objective and constraints
        processobjsec();
        processconsec();

        // END section
        processendsec();
    }

    void splittokens() {
        LpSectionKeyword currentsection = LpSectionKeyword::NONE;

        for (unsigned int i = 0; i < processedtokens.size(); ++i) {
            if (processedtokens[i]->type == ProcessedTokenType::SECID) {
                currentsection = ((ProcessedTokenSectionKeyword*)processedtokens[i].get())->keyword;

                if (currentsection == LpSectionKeyword::OBJ) {
                    switch (((ProcessedTokenObjectiveSectionKeyword*)processedtokens[i].get())
                                    ->objsense) {
                        case LpObjectiveSectionKeywordType::MIN:
                            this->model_.minimize = true;
                            break;
                        case LpObjectiveSectionKeywordType::MAX:
                            this->model_.minimize = false;
                            break;
                        default:
                            lpassert(false);
                    }
                }

                // make sure this section did not yet occur
                lpassert(sectiontokens[currentsection].empty());
            } else {
                sectiontokens[currentsection].push_back(std::move(processedtokens[i]));
            }
        }
    }

    void processtokens() {
        unsigned int i = 0;

        while (i < this->rawtokens.size()) {
            fflush(stdout);

            // long section keyword semi-continuous
            if (rawtokens.size() - i >= 3 && rawtokens[i]->istype(RawTokenType::STR) &&
                rawtokens[i + 1]->istype(RawTokenType::MINUS) &&
                rawtokens[i + 2]->istype(RawTokenType::STR)) {
                std::string temp = ((RawStringToken*)rawtokens[i].get())->value + "-" +
                                   ((RawStringToken*)rawtokens[i + 2].get())->value;
                LpSectionKeyword keyword = parsesectionkeyword(temp);
                if (keyword != LpSectionKeyword::NONE) {
                    processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                            new ProcessedTokenSectionKeyword(keyword)));
                    i += 3;
                    continue;
                }
            }

            // long section keyword subject to/such that
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::STR) &&
                rawtokens[i + 1]->istype(RawTokenType::STR)) {
                std::string temp = ((RawStringToken*)rawtokens[i].get())->value + " " +
                                   ((RawStringToken*)rawtokens[i + 1].get())->value;
                LpSectionKeyword keyword = parsesectionkeyword(temp);
                if (keyword != LpSectionKeyword::NONE) {
                    processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                            new ProcessedTokenSectionKeyword(keyword)));
                    i += 2;
                    continue;
                }
            }

            // other section keyword
            if (rawtokens[i]->istype(RawTokenType::STR)) {
                LpSectionKeyword keyword =
                        parsesectionkeyword(((RawStringToken*)rawtokens[i].get())->value);
                if (keyword != LpSectionKeyword::NONE) {
                    if (keyword == LpSectionKeyword::OBJ) {
                        LpObjectiveSectionKeywordType kw = parseobjectivesectionkeyword(
                                ((RawStringToken*)rawtokens[i].get())->value);
                        processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                                new ProcessedTokenObjectiveSectionKeyword(kw)));
                    } else {
                        processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                                new ProcessedTokenSectionKeyword(keyword)));
                    }
                    i++;
                    continue;
                }
            }

            // constraint identifier?
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::STR) &&
                rawtokens[i + 1]->istype(RawTokenType::COLON)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedConsIdToken(((RawStringToken*)rawtokens[i].get())->value)));
                i += 2;
                continue;
            }

            // check if free
            if (rawtokens[i]->istype(RawTokenType::STR) &&
                iskeyword(((RawStringToken*)rawtokens[i].get())->value, LP_KEYWORD_FREE,
                          LP_KEYWORD_FREE_N)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::FREE)));
                i++;
                continue;
            }

            // check if infinty
            if (rawtokens[i]->istype(RawTokenType::STR) &&
                iskeyword(((RawStringToken*)rawtokens[i].get())->value, LP_KEYWORD_INF,
                          LP_KEYWORD_INF_N)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedConstantToken(std::numeric_limits<double>::infinity())));
                i++;
                continue;
            }

            // assume var identifier
            if (rawtokens[i]->istype(RawTokenType::STR)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedVarIdToken(((RawStringToken*)rawtokens[i].get())->value)));
                i++;
                continue;
            }

            // + Constant
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::PLUS) &&
                rawtokens[i + 1]->istype(RawTokenType::CONS)) {
                processedtokens.push_back(
                        std::unique_ptr<ProcessedToken>(new ProcessedConstantToken(
                                ((RawConstantToken*)rawtokens[i + 1].get())->value)));
                i += 2;
                continue;
            }

            // - constant
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::MINUS) &&
                rawtokens[i + 1]->istype(RawTokenType::CONS)) {
                processedtokens.push_back(
                        std::unique_ptr<ProcessedToken>(new ProcessedConstantToken(
                                -((RawConstantToken*)rawtokens[i + 1].get())->value)));
                i += 2;
                continue;
            }

            // + [
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::PLUS) &&
                rawtokens[i + 1]->istype(RawTokenType::BRKOP)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::BRKOP)));
                i += 2;
                continue;
            }

            // +
            if (rawtokens[i]->istype(RawTokenType::PLUS)) {
                processedtokens.push_back(
                        std::unique_ptr<ProcessedToken>(new ProcessedConstantToken(1.0)));
                i++;
                continue;
            }

            // -
            if (rawtokens[i]->istype(RawTokenType::MINUS)) {
                processedtokens.push_back(
                        std::unique_ptr<ProcessedToken>(new ProcessedConstantToken(-1.0)));
                i++;
                continue;
            }

            // constant
            if (rawtokens[i]->istype(RawTokenType::CONS)) {
                processedtokens.push_back(
                        std::unique_ptr<ProcessedToken>(new ProcessedConstantToken(
                                ((RawConstantToken*)rawtokens[i].get())->value)));
                i++;
                continue;
            }

            // [
            if (rawtokens[i]->istype(RawTokenType::BRKOP)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::BRKOP)));
                i++;
                continue;
            }

            // ]
            if (rawtokens[i]->istype(RawTokenType::BRKCL)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::BRKCL)));
                i++;
                continue;
            }

            // /
            if (rawtokens[i]->istype(RawTokenType::SLASH)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::SLASH)));
                i++;
                continue;
            }

            // *
            if (rawtokens[i]->istype(RawTokenType::ASTERISK)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::ASTERISK)));
                i++;
                continue;
            }

            // ^
            if (rawtokens[i]->istype(RawTokenType::HAT)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedToken(ProcessedTokenType::HAT)));
                i++;
                continue;
            }

            // <=
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::LESS) &&
                rawtokens[i + 1]->istype(RawTokenType::EQUAL)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedComparisonToken(LpComparisonType::LEQ)));
                i += 2;
                continue;
            }

            // <
            if (rawtokens[i]->istype(RawTokenType::LESS)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedComparisonToken(LpComparisonType::L)));
                i++;
                continue;
            }

            // >=
            if (rawtokens.size() - i >= 2 && rawtokens[i]->istype(RawTokenType::GREATER) &&
                rawtokens[i + 1]->istype(RawTokenType::EQUAL)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedComparisonToken(LpComparisonType::GEQ)));
                i += 2;
                continue;
            }

            // >
            if (rawtokens[i]->istype(RawTokenType::GREATER)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedComparisonToken(LpComparisonType::G)));
                i++;
                continue;
            }

            // =
            if (rawtokens[i]->istype(RawTokenType::EQUAL)) {
                processedtokens.push_back(std::unique_ptr<ProcessedToken>(
                        new ProcessedComparisonToken(LpComparisonType::EQ)));
                i++;
                continue;
            }

            // FILEEND
            if (rawtokens[i]->istype(RawTokenType::FLEND)) {
                i++;
                continue;
            }

            // catch all unknown symbols
            lpassert(false);
            break;
        }
    }

    // reads the entire file and separates
    void tokenize() {
        this->linefullyread = true;
        this->newline_encountered = true;
        this->linebufferpos = this->linebuffer;
        bool done = false;
        while (true) {
            this->readnexttoken(done);
            if (this->rawtokens.size() >= 1 &&
                this->rawtokens.back()->type == RawTokenType::FLEND) {
                break;
            }
        }
    }

    void readnexttoken(bool& done) {
        done = false;
        if (!this->linefullyread) {
            // fill up line

            // how many do we need to read?
            unsigned int num_already_read = this->linebufferpos - this->linebuffer;

            // shift buffer
            for (unsigned int i = num_already_read; i < LP_MAX_LINE_LENGTH + 1; i++) {
                this->linebuffer[i - num_already_read] = this->linebuffer[i];
            }

            char* write_start = &this->linebuffer[LP_MAX_LINE_LENGTH - num_already_read];

            // read more values
            char* eof = fgets(write_start, num_already_read + 1, this->file);
            unsigned int linelength;
            for (linelength = 0; linelength < LP_MAX_LINE_LENGTH; linelength++) {
                if (this->linebuffer[linelength] == '\r') {
                    this->linebuffer[linelength] = '\n';
                }
                if (this->linebuffer[linelength] == '\n') {
                    break;
                }
            }

            if (this->linebuffer[linelength] == '\n') {
                this->linefullyread = true;
            } else {
                this->linefullyread = false;
            }

            // fgets returns nullptr if end of file reached (EOF following a \n)
            if (eof == nullptr) {
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::FLEND)));
                done = true;
                return;
            }
            this->linebufferpos = this->linebuffer;
        } else if (newline_encountered) {
            newline_encountered = false;
            char* eof = fgets(this->linebuffer, LP_MAX_LINE_LENGTH + 1, this->file);
            this->linebufferpos = this->linebuffer;

            unsigned int linelength;
            for (linelength = 0; linelength < LP_MAX_LINE_LENGTH; linelength++) {
                if (this->linebuffer[linelength] == '\r') {
                    this->linebuffer[linelength] = '\n';
                }
                if (this->linebuffer[linelength] == '\n') {
                    break;
                }
            }
            if (this->linebuffer[linelength] == '\n') {
                this->linefullyread = true;
            } else {
                this->linefullyread = false;
            }

            // fgets returns nullptr if end of file reached (EOF following a \n)
            if (eof == nullptr) {
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::FLEND)));
                done = true;
                return;
            }
        }

        // check single character tokens
        char nextchar = *this->linebufferpos;

        switch (nextchar) {
            // check for comment
            case '\\':
                this->newline_encountered = true;
                this->linefullyread = true;
                return;

            // check for bracket opening
            case '[':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::BRKOP)));
                this->linebufferpos++;
                return;

            // check for bracket closing
            case ']':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::BRKCL)));
                this->linebufferpos++;
                return;

            // check for less sign
            case '<':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::LESS)));
                this->linebufferpos++;
                return;

            // check for greater sign
            case '>':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::GREATER)));
                this->linebufferpos++;
                return;

            // check for equal sign
            case '=':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::EQUAL)));
                this->linebufferpos++;
                return;

            // check for colon
            case ':':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::COLON)));
                this->linebufferpos++;
                return;

            // check for plus
            case '+':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::PLUS)));
                this->linebufferpos++;
                return;

            // check for hat
            case '^':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::HAT)));
                this->linebufferpos++;
                return;

            // check for hat
            case '/':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::SLASH)));
                this->linebufferpos++;
                return;

            // check for asterisk
            case '*':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::ASTERISK)));
                this->linebufferpos++;
                return;

            // check for minus
            case '-':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::MINUS)));
                this->linebufferpos++;
                return;

            // check for whitespace
            case ' ':
            case '\t':
                this->linebufferpos++;
                return;

            // check for line end
            case ';':
            case '\n':
                this->newline_encountered = true;
                this->linefullyread = true;
                return;

            // check for file end (EOF at end of some line)
            case '\0':
                this->rawtokens.push_back(
                        std::unique_ptr<RawToken>(new RawToken(RawTokenType::FLEND)));
                done = true;
                return;
        }

        // check for double value
        double constant;
        int ncharconsumed;
        int nread = sscanf(this->linebufferpos, "%lf%n", &constant, &ncharconsumed);
        if (nread == 1) {
            this->rawtokens.push_back(std::unique_ptr<RawToken>(new RawConstantToken(constant)));
            this->linebufferpos += ncharconsumed;
            return;
        }

        // todo: scientific notation

        // assume it's an (section/variable/constraint) idenifier
        char stringbuffer[LP_MAX_NAME_LENGTH + 1];
        nread = sscanf(this->linebufferpos, "%[^][\t\n\\:+<>^= /-*]%n", stringbuffer,
                       &ncharconsumed);
        if (nread == 1) {
            this->rawtokens.push_back(std::unique_ptr<RawToken>(new RawStringToken(stringbuffer)));
            this->linebufferpos += ncharconsumed;
            return;
        }

        lpassert(false);
    }
};

}  // namespace reader

template <class Bias, class Index = int>
LPModel<Bias, Index> read(const std::string filename) {
    // auto r = reader::Reader<Bias, Index>(filename);
    return reader::Reader<Bias, Index>(filename).read();
}

}  // namespace lp
}  // namespace dimod
