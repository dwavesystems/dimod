#ifndef FIX_VARIABLES_HPP_INCLUDED
#define FIX_VARIABLES_HPP_INCLUDED

#include <vector>
#include <utility>

#include "compressed_matrix.hpp"

namespace fix_variables_
{

struct FixVariablesResult
{
	std::vector<std::pair<int,  int> > fixedVars; //1-based
	compressed_matrix::CompressedMatrix<double> newQ; //0-based
	double offset;
};

class FixVariablesException
{
public:
	FixVariablesException(const std::string& m = "fix variables exception") : message(m) {}
	const std::string& what() const {return message;}
private:
	std::string message;
};
	
FixVariablesResult fixQuboVariables(const compressed_matrix::CompressedMatrix<double>& Q, int method);

} // namespace fix_variables_

#endif // FIX_VARIABLES_HPP_INCLUDED



