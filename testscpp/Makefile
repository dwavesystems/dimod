ROOT := ..
SRC := $(ROOT)/dimod/include/
CATCH2 := $(ROOT)/testscpp/Catch2/single_include/

all: catch2 test_main tests

coverage:
	$(CXX) -std=c++11 -Wall -c test_main.cpp -I $(CATCH2) --coverage -fno-inline -fno-inline-small-functions -fno-default-inline
	$(CXX) -std=c++11 -Wall test_main.o tests/*.cpp -o test_main -I $(CATCH2) -I $(SRC) --coverage -fno-inline -fno-inline-small-functions -fno-default-inline
	lcov -c -i -b ${ROOT} -d . -o baseline.info
	./test_main
	lcov -c -d . -b ${ROOT} -o test.info
	lcov -a baseline.info -a test.info -o coverage.info
	rm baseline.info test.info
	lcov --remove coverage.info '/usr/*' --output-file coverage.info

tests: test_main.out
	./test_main 

test_main: test_main.cpp
	$(CXX) -std=c++11 -Wall -Werror -c test_main.cpp -I $(CATCH2) 
	$(CXX) -std=c++11 -Wall -Werror test_main.o tests/*.cpp -o test_main -I $(SRC) -I $(CATCH2) 

catch2:
	git submodule init
	git submodule update
