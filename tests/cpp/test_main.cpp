#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "Catch2/single_include/catch2/catch.hpp"

/*
The purpose of this file is to include Catch's main(). Tests can be found inside tests directory.

To see all tests, use `./test_main -l`

Some examples:

eg) Run all tests
>>> make
>>> ./test_main

eg) Run all test cases with the tag [bqm] 
>>> ./test_main [bqm]

eg) Run all tests from file 'test_bqm.cpp'
>>> ./test_main -# [test_bqm]

For more command line options, see: https://github.com/catchorg/Catch2/blob/master/docs/command-line.md#run-section

*/
