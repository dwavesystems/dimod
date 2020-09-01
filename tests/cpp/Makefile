all: catch2 test_main tests

tests: test_main.out
	./test_main 

test_main: test_main.cpp
	g++ -std=c++11 -Wall -c test_main.cpp
	g++ -std=c++11 -Wall test_main.o tests/*.cpp ../../dimod/roof_duality/src/fix_variables.cpp -o test_main

catch2:
	git submodule init
	git submodule update
