// Copyright 2021 D-Wave Systems Inc.
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

#include <cstdio>

#include "../Catch2/single_include/catch2/catch.hpp"
#include "dimod/lp.h"

namespace dimod {


TEST_CASE("LP tests") {
    GIVEN("A simple LP file as a string") {
        std::FILE* tmpf = std::tmpfile();
        std::fputs("minimize\nx0 - 2 x1\nsubject to\nx0 + x1 = 1\nBinary\nx0\nEnd\n", tmpf);
        std::rewind(tmpf);
        auto model = lp::read<double>(tmpf);

        // dev note: this is really just a smoke test
        // we'll need to fill this out once we change the LP file reading to
        // return a proper CQM
    }
}
}  // namespace dimod

