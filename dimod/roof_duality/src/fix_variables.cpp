/**
# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
*/
#include "fix_variables.hpp"
#include "compressed_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/strong_components.hpp>

//for debugging only
#include <iostream>
#include <iomanip>
#include <fstream>
#include <time.h>

namespace
{

class compClass
{
public:
	bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b)
	{
		if (a.second != b.second)
			return !(a.second < b.second);
		else
			return a.first < b.first;
	}
};

bool compareAbs(double a, double b)
{
	return std::abs(a) < std::abs(b);
}


//the index is 1-based, according to the paper
struct Posiform
{
	std::vector<std::pair<std::pair<int, int>, long long int> > quadratic;
	std::vector<std::pair<int, long long int> > linear;
	long long int cst;
	int numVars;
};

struct SC
{
	std::vector<int> original;
	std::vector<int> positive;
	std::vector<int> negative;
};

struct SCRet
{
	std::vector<SC> S;
	compressed_matrix::CompressedMatrix<long long int> G;
};

Posiform BQPToPosiform(const compressed_matrix::CompressedMatrix<long long int>& Q, bool makeRandom = false)
{
	int numVariables = Q.numRows();
	Posiform ret;
	ret.numVars = numVariables;
	std::vector<long long int> q(numVariables);

	//q = diag(Q);
	for (int i = 0; i < numVariables; i++)
		q[i] = Q.get(i, i);

	//tmpQ = triu(Q,1);
	std::vector<int> tmpQRowOffsets(numVariables + 1);
	std::vector<int> tmpQColIndices; tmpQColIndices.reserve(Q.nnz());
	std::vector<long long int> tmpQValues; tmpQValues.reserve(Q.nnz());
	int index = 0;
	for (int i = 0; i < Q.numRows(); i++)
	{
		int start = Q.rowOffsets()[i];
		int end = Q.rowOffsets()[i + 1];
		tmpQRowOffsets[i] = index;
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = Q.colIndices()[j];
			if (r != c)
			{
				tmpQColIndices.push_back(c);
				tmpQValues.push_back(Q.values()[j]); //Q.values()[j] is Q(r, c)
				++index;
			}
		}
	}
	tmpQRowOffsets.back() = index; //number of non-zero elements

	compressed_matrix::CompressedMatrix<long long int> tmpQ(numVariables, numVariables, tmpQRowOffsets, tmpQColIndices, tmpQValues);


	if (!makeRandom) //make a deterministic posiform from Q
	{
		//cPrime = q + sum(tmpQ.*(tmpQ<0),2); and quadratic term
		std::vector<long long int> cPrime = q;
		for (int i = 0; i < tmpQ.numRows(); i++)
		{
			int start = tmpQ.rowOffsets()[i];
			int end = tmpQ.rowOffsets()[i + 1];
			for (int j = start; j < end; j++)
			{
				int r = i;
				int c = tmpQ.colIndices()[j];
				if (tmpQ.values()[j] < 0)
				{
					cPrime[i] += tmpQ.values()[j];
					ret.quadratic.push_back(std::make_pair(std::make_pair(r + 1, -(c + 1)), -tmpQ.values()[j])); //+1 makes it 1-based
				}
				else if (tmpQ.values()[j] > 0)
					ret.quadratic.push_back(std::make_pair(std::make_pair(r + 1, c + 1), tmpQ.values()[j])); //+1 makes it 1-based
			}
		}

		//constant term and linear term
		ret.cst = 0;
		for (int i = 0; i < numVariables; i++)
		{
			if (cPrime[i] < 0)
			{
				ret.cst += cPrime[i]; //constant term
				ret.linear.push_back(std::make_pair(-(i + 1), -cPrime[i])); //+1 makes it 1-based
			}
			else if (cPrime[i] > 0)
				ret.linear.push_back(std::make_pair(i + 1, cPrime[i])); //+1 makes it 1-based
		}
	}
	else
	{
		std::vector<long long int> linear = q;
		std::set<std::pair<int, int> > negPairs;
		for (int i = 0; i < tmpQ.numRows(); i++)
		{
			int start = tmpQ.rowOffsets()[i];
			int end = tmpQ.rowOffsets()[i + 1];
			for (int j = start; j < end; j++)
			{
				int r = i;
				int c = tmpQ.colIndices()[j];
				if (tmpQ.values()[j] < 0)
				{
					if (rand() % 2 == 0)
						negPairs.insert(std::make_pair(-(r + 1), c + 1)); //+1 makes it 1-based
					else
						negPairs.insert(std::make_pair(r + 1, -(c + 1))); //+1 makes it 1-based
				}
			}
		}

		for (int i = 0; i < tmpQ.numRows(); i++)
		{
			int start = tmpQ.rowOffsets()[i];
			int end = tmpQ.rowOffsets()[i + 1];
			for (int j = start; j < end; j++)
			{
				int r = i;
				int c = tmpQ.colIndices()[j];
				if (tmpQ.values()[j] > 0)
					ret.quadratic.push_back(std::make_pair(std::make_pair(r + 1, c + 1), tmpQ.values()[j]));
				else
				{
					if (negPairs.find(std::make_pair(r + 1, -(c + 1)))!=negPairs.end())
						ret.quadratic.push_back(std::make_pair(std::make_pair(r + 1, -(c + 1)), -tmpQ.values()[j]));
					else
						ret.quadratic.push_back(std::make_pair(std::make_pair(-(r + 1), c + 1), -tmpQ.values()[j]));
				}
			}
		}

		for (std::set<std::pair<int, int> >::iterator it = negPairs.begin(); it != negPairs.end(); ++it)
		{
			if (it->first > 0)
				linear[it->first - 1] += tmpQ(it->first - 1, -it->second - 1);
			else
				linear[it->second - 1] += tmpQ(-it->first - 1, it->second - 1);
		}

		ret.cst = 0;

		for (int i = 0; i < numVariables; i++)
		{
			if (linear[i] < 0)
			{
				ret.cst += linear[i];
				ret.linear.push_back(std::make_pair(-(i + 1), -linear[i]));
			}
			else if (linear[i] > 0)
				ret.linear.push_back(std::make_pair(i + 1, linear[i]));
		}
	}

	return ret;
}

compressed_matrix::CompressedMatrix<long long int> posiformToImplicationNetwork_1(const Posiform& p)
{
	int n = p.numVars;
	int numVertices = 2 * n + 2;
	int source = 0;
	int sink = n + 1;

	//n = p.numVars
	//0: source;
	//1 to n: x_1 to x_n
	//n+1: sink
	//n+2 to 2*n+1: \overline_x_1 to \overline_x_n
	std::map<std::pair<int, int>, long long int> m;
	for (int i = 0; i < p.linear.size(); i++)
	{
		int v = p.linear[i].first;

		long long int capacity = p.linear[i].second; // originally p.linear[i].second/2
		if (v > 0)
		{
			m[std::make_pair(source, v + n + 1)] = capacity;
			m[std::make_pair(v, sink)] = capacity;
		}
		else
		{
			v = std::abs(v);
			m[std::make_pair(source, v)] = capacity;
			m[std::make_pair(v + n + 1, sink)] = capacity;
		}
	}

	for (int i = 0; i < p.quadratic.size(); i++)
	{
		int v_1 = p.quadratic[i].first.first;
		int v_2 = p.quadratic[i].first.second;

		long long int capacity = p.quadratic[i].second; // originally p.quadratic[i].second/2
		if (v_1 < 0)
		{
			v_1 = std::abs(v_1);
			m[std::make_pair(v_1 + n + 1, v_2 + n + 1)] = capacity;
			m[std::make_pair(v_2, v_1)] = capacity;
		}
		else if (v_2 < 0)
		{
			v_2 = std::abs(v_2);
			m[std::make_pair(v_1, v_2)] = capacity;
			m[std::make_pair(v_2 + n + 1, v_1 + n + 1)] = capacity;
		}
		else
		{
			m[std::make_pair(v_1, v_2 + n + 1)] = capacity;
			m[std::make_pair(v_2, v_1 + n + 1)] = capacity;
		}
	}

	return compressed_matrix::CompressedMatrix<long long int>(numVertices, numVertices, m);
}

compressed_matrix::CompressedMatrix<long long int> posiformToImplicationNetwork_2(const Posiform& p)
{
	int n = p.numVars;
	int numVertices = 2 * n + 2;
	int source = n;
	int sink = 2 * n + 1;

	//n = p.numVars
	//0 to n-1: x_1 to x_n
	//n: source;
	//n+1 to 2*n: \overline_x_1 to \overline_x_n
	//2*n+1: sink
	std::map<std::pair<int, int>, long long int> m;
	for (int i = 0; i < p.linear.size(); i++)
	{
		int v = p.linear[i].first;
		long long int capacity = p.linear[i].second; // originally p.linear[i].second/2
		if (v > 0)
		{
			m[std::make_pair(source, v + n)] = capacity;
			m[std::make_pair(v - 1, sink)] = capacity;
		}
		else
		{
			v = std::abs(v);
			m[std::make_pair(source, v - 1)] = capacity;
			m[std::make_pair(v + n, sink)] = capacity;
		}
	}

	for (int i = 0; i < p.quadratic.size(); i++)
	{
		int v_1 = p.quadratic[i].first.first;
		int v_2 = p.quadratic[i].first.second;
		long long int capacity = p.quadratic[i].second; // originally p.quadratic[i].second/2
		if (v_1 < 0)
		{
			v_1 = std::abs(v_1);
			m[std::make_pair(v_1 + n, v_2 + n)] = capacity;
			m[std::make_pair(v_2 - 1, v_1 - 1)] = capacity;
		}
		else if (v_2 < 0)
		{
			v_2 = std::abs(v_2);
			m[std::make_pair(v_1 - 1, v_2 - 1)] = capacity;
			m[std::make_pair(v_2 + n, v_1 + n)] = capacity;
		}
		else
		{
			m[std::make_pair(v_1 - 1, v_2 + n)] = capacity;
			m[std::make_pair(v_2 - 1, v_1 + n)] = capacity;
		}
	}

	return compressed_matrix::CompressedMatrix<long long int>(numVertices, numVertices, m);
}

//this version returns R instead of F
compressed_matrix::CompressedMatrix<long long int> maxFlow(const compressed_matrix::CompressedMatrix<long long int>& A)
{
	int numVertices = A.numRows();
	int numVariables = numVertices / 2 - 1;

	//clock_t curr_1 = clock();
	//clock_t curr_2;

	using namespace boost;

	typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
	typedef adjacency_list<vecS, vecS, directedS, property<vertex_name_t, std::string>, property<edge_capacity_t, long long int, property<edge_residual_capacity_t, long long int, property<edge_reverse_t, Traits::edge_descriptor> > > > Graph; //for edge capacity is long long int
	//typedef adjacency_list<vecS, vecS, directedS, property<vertex_name_t, std::string>, property<edge_capacity_t, double, property<edge_residual_capacity_t, double, property<edge_reverse_t, Traits::edge_descriptor> > > > Graph; //for edge capacity is double

	Graph g;

	property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
	property_map<Graph, edge_reverse_t>::type reverse_edge = get(edge_reverse, g);
	property_map<Graph, edge_residual_capacity_t>::type residual_capacity = get(edge_residual_capacity, g);

	std::vector<Traits::vertex_descriptor> verts(numVertices);
	for (int i = 0; i < numVertices; ++i)
		verts[i] = add_vertex(g);

	Traits::vertex_descriptor s = verts[0];
	Traits::vertex_descriptor t = verts[numVariables + 1];

	for (int i = 0; i < A.numRows(); i++)
	{
		int start = A.rowOffsets()[i];
		int end = A.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = A.colIndices()[j];
			long long int cap = A.values()[j];

			Traits::edge_descriptor e1, e2;
			bool in1, in2;
			boost::tie(e1, in1) = add_edge(verts[r], verts[c], g);
			boost::tie(e2, in2) = add_edge(verts[c], verts[r], g);

			capacity[e1] = cap;
			capacity[e2] = 0;
			reverse_edge[e1] = e2;
			reverse_edge[e2] = e1;
		}
	}


	//curr_2 = clock();
	//mexPrintf("inside maxFlow int version: Time elapsed_constructing_graph_for_max_flow: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	push_relabel_max_flow(g, s, t);

	//curr_2 = clock();
	//mexPrintf("inside maxFlow int version: Time elapsed_for_boost_max_flow: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	std::vector<int> RRowOffsets(numVertices+1);
	std::vector<int> RColIndices;
	std::vector<long long int> RValues;
	int offset = 0;
	int currRow = 0;
	graph_traits<Graph>::vertex_iterator u_iter, u_end;
	graph_traits<Graph>::out_edge_iterator ei, e_end;
	for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
		for (boost::tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
		{
			if (capacity[*ei] > 0)
			{
				if (currRow <= *u_iter)
				{
					for (int i = currRow; i <= *u_iter; i++)
						RRowOffsets[i] = offset;
					currRow = static_cast<int>((*u_iter) + 1);
				}

				RColIndices.push_back(static_cast<int>(target(*ei, g)));
				RValues.push_back(residual_capacity[*ei]);
				++offset;
			}
		}
	for (int i = currRow; i < RRowOffsets.size(); i++)
		RRowOffsets[i] = offset;

	compressed_matrix::CompressedMatrix<long long int> R(numVertices, numVertices, RRowOffsets, RColIndices, RValues); //residual

	//curr_2 = clock();
	//mexPrintf("inside maxFlow int version: Time elapsed_get_R: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	std::map<std::pair<int, int>, long long int> rm = compressed_matrix::compressedMatrixToMap(R);
	std::map<std::pair<int, int>, long long int> rmMissing;

	//curr_2 = clock();
	//mexPrintf("inside maxFlow int version: Time elapsed_copy_R_to_map: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	for (int i = 0; i < R.numRows(); i++)
	{
		int start = R.rowOffsets()[i];
		int end = R.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = R.colIndices()[j];
			long long int Arc = A.get(r, c);
			long long int Rcr = R.get(c, r);
			if (Arc != 0 && R.values()[j] + Rcr - Arc != 0) //it->second is: R(r, c); here it means R(r, c)+R(c, r)!=A(r, c), so R(c, r) is missing
				rmMissing.insert(std::make_pair(std::make_pair(c, r), Arc - R.values()[j]));
		}
	}

	for (std::map<std::pair<int, int>, long long int>::iterator it = rmMissing.begin(), end = rmMissing.end(); it != end; ++it)
		rm.insert(*it);

	//curr_2 = clock();
	//mexPrintf("inside maxFlow int version: Time elapsed_get_missing_elements_in_R: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	return compressed_matrix::CompressedMatrix<long long int>(numVertices, numVertices, rm);
}

std::vector<int> bfs_for_method_2(const compressed_matrix::CompressedMatrix<long long int>& g, int s)
{
	std::queue<int> q;
	q.push(s);
	int numVertices = g.numRows();
	std::vector<int> visited(numVertices, -1);
	visited[s] = 1;

	while (!q.empty())
	{
		int curr = q.front(); q.pop();
		int start = g.rowOffsets()[curr];
		int end = g.rowOffsets()[curr + 1];
		for (int j = start; j < end; j++)
		{
			int c = g.colIndices()[j];
			if (g.values()[j] != 0 && visited[c] == -1)
			{
				visited[c] = 1;
				q.push(c);
			}
		}
	}

	return visited;
}

compressed_matrix::CompressedMatrix<long long int> makeResidualSymmetric(const compressed_matrix::CompressedMatrix<long long int>& R)
{
	//clock_t curr_1 = clock();
	//clock_t curr_2;

	int numVertices = R.numRows();
	int numVariables = numVertices / 2 - 1;
	std::map<std::pair<int, int>, long long int> MsymR;
	for (int i = 0; i < R.numRows(); i++)
	{
		int start = R.rowOffsets()[i];
		int end = R.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = R.colIndices()[j];
			if (r != c)
			{
				MsymR[std::make_pair(r, c)] += R.values()[j];
				int compR;
				if (r <= numVariables)
					compR = r + (numVariables + 1);
				else
					compR = r - (numVariables + 1);
				int compC;
				if (c <= numVariables)
					compC = c + (numVariables + 1);
				else
					compC = c - (numVariables + 1);
				MsymR[std::make_pair(compC, compR)] += R.values()[j];
			}
		}
	}

	//curr_2 = clock();
	//mexPrintf("inside makeResidualSym: Time elapsed_get_MsymR: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	//remove extra 0s in MsymR, required because there will possibly be 0 in MsymR and the 0 will add non-exist edges in stronglyConnectedComponents()
	//cause the connected components results incorrct !!!
	std::map<std::pair<int, int>, long long int> MsymRWithoutZero;
	for (std::map<std::pair<int, int>, long long int>::const_iterator it = MsymR.begin(), end = MsymR.end(); it != end; ++it)
	{
		if (it->second != 0 && it->first.first != numVariables + 1 && it->first.second != 0) //add clearing R here !!!
			MsymRWithoutZero.insert(*it);
	}

	//curr_2 = clock();
	//mexPrintf("inside makeResidualSym: Time elapsed_remove_zero_in_MsymR: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	return compressed_matrix::CompressedMatrix<long long int>(numVertices, numVertices, MsymRWithoutZero);
}

SCRet stronglyConnectedComponents(const compressed_matrix::CompressedMatrix<long long int>& R)
{
	//clock_t curr_1 = clock();
	//clock_t curr_2;

	int numVertices = R.numRows();
	int numVariables = numVertices / 2 - 1;

	using namespace boost;

	typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
	typedef adjacency_list < vecS, vecS, directedS,
		property < vertex_name_t, std::string,
		property < vertex_index_t, long,
		property < vertex_color_t, boost::default_color_type,
		property < vertex_distance_t, long,
		property < vertex_predecessor_t, Traits::edge_descriptor > > > > >
	> Graph;

	typedef graph_traits<Graph>::vertex_descriptor Vertex;

	Graph G;

	std::vector<Traits::vertex_descriptor> root(numVertices);
	for (int i = 0; i < numVertices; ++i)
		root[i] = add_vertex(G);

	std::vector<int> component(num_vertices(G)), discover_time(num_vertices(G));
	std::vector<default_color_type> color(num_vertices(G));

	for (int i = 0; i < R.numRows(); i++)
	{
		int start = R.rowOffsets()[i];
		int end = R.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = R.colIndices()[j];
			add_edge(root[r], root[c], G);
		}
	}

	//curr_2 = clock();
	//mexPrintf("Time elapsed_constructing_graph_for_strongly_connected_components_graph: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	//int num = strong_components(G, &component[0], root_map(&root[0]).color_map(&color[0]).discover_time_map(&discover_time[0]));
	int num = strong_components(G, boost::make_iterator_property_map(component.begin(), boost::get(boost::vertex_index, G)),
			                                root_map(boost::make_iterator_property_map(root.begin(), boost::get(boost::vertex_index, G))).color_map(boost::make_iterator_property_map(color.begin(), boost::get(boost::vertex_index, G))).discover_time_map(boost::make_iterator_property_map(discover_time.begin(), boost::get(boost::vertex_index, G))));

	//curr_2 = clock();
	//mexPrintf("Time elapsed_boost_strongly_connected_components_graph: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	//here original's content is from 0 to 2*n+1, 0: source, 1 to n, variables, n+1: sink, n+2 to 2*n+1: bar variables
	//positive: from 1 to n (1-based variables index)
	//negative: from 1 to n (1-based variables index)
	std::vector<SC> vsc(num);
	for (int i = 0; i < component.size(); i++)
	{
		int whichComponent = component[i];
		vsc[whichComponent].original.push_back(i);
		if (vsc[whichComponent].original.back() <= numVariables)
			vsc[whichComponent].positive.push_back(vsc[whichComponent].original.back());
		else
			vsc[whichComponent].negative.push_back(vsc[whichComponent].original.back() - (numVariables + 1));
	}

	std::map<std::pair<int, int>, long long int> M;

	//speed up the program a lot !!!
	//build the graph G for strongly connected components
	for (int i = 0; i < R.numRows(); i++)
	{
		int start = R.rowOffsets()[i];
		int end = R.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = R.colIndices()[j];
			int sc1 = component[r];
			int sc2 = component[c];
			if (sc1 != sc2 && R.values()[j] > 0 && M.find(std::make_pair(sc1, sc2)) == M.end())
				M.insert(std::make_pair(std::make_pair(sc1, sc2), 1));
		}
	}

	//curr_2 = clock();
	//mexPrintf("Time elapsed_after_boost_scc: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	SCRet ret;
	ret.S = vsc;
	ret.G = compressed_matrix::CompressedMatrix<long long int>(num, num, M);

	return ret;
}

std::vector<int> classifyStronglyConnectedComponents(const std::vector<SC>& S)
{
	std::vector<int> ret(S.size()); //0: self-complement, 1: non-self-complement

	for (int i = 0; i < S.size(); i++)
	{
		bool flag = true;
		std::set<int> pos(S[i].positive.begin(), S[i].positive.end());
		for (int j = 0; j < S[i].negative.size(); j++)
		{
			if (pos.find(S[i].negative[j]) != pos.end())
			{
				flag = false;
				break;
			}
		}

		if (flag)
			ret[i] = 1;
		else
			ret[i] = 0;
	}

	return ret;
}

void shortestPath(int src, const compressed_matrix::CompressedMatrix<long long int>& g, std::vector<int>& visited)
{
	std::queue<int> q;
	q.push(src);
	visited.resize(g.numRows(), 0);
	visited[src] = 1;

	while (!q.empty())
	{
		int curr = q.front(); q.pop();

		int start = g.rowOffsets()[curr];
		int end = g.rowOffsets()[curr + 1];
		for (int j = start; j < end; j++)
		{
			int c = g.colIndices()[j];
			if (g.values()[j] != 0 && !visited[c])
			{
				visited[c] = 2;
				q.push(c);
			}
		}
	}
}

std::vector<std::pair<int, int> > fixVariables(const std::vector<int>& classifiedSC, const SCRet& scRet, int numVariables)
{
	std::vector<std::pair<int, int> > ret;

	int x0Location;
	int x0BarLocation;
	std::vector<int> I;

	for (int i = 0; i < classifiedSC.size(); i++)
	{
		if (classifiedSC[i] == 1) //1: non self-complement
		{
			if (scRet.S[i].original[0] == 0)
				x0Location = i;
			else if (scRet.S[i].original[0] == numVariables + 1)
				x0BarLocation = i;
			else
				I.push_back(i);
		}
	}

	//calculate if there is a path from x0 to other nodes
	std::vector<int> visited;
	shortestPath(x0Location, scRet.G, visited); //speed up the program

	//find complement pairs
	std::vector<int> positive(numVariables + 1, -1);
	std::vector<int> negative(numVariables + 1, -1);
	for (int i = 0; i < I.size(); i++)
	{
		for (int k = 0; k < scRet.S[I[i]].positive.size(); k++)
			positive[scRet.S[I[i]].positive[k]] = I[i];
		for (int k = 0; k < scRet.S[I[i]].negative.size(); k++)
			negative[scRet.S[I[i]].negative[k]] = I[i];
	}

	std::map<int, int> complementPairs;
	for (int i = 0; i < positive.size(); i++)
	{
		if (positive[i] != -1 && negative[i] != -1)
		{
			complementPairs[positive[i]] = negative[i];
			complementPairs[negative[i]] = positive[i];
		}
	}

	std::queue<int> q;

	std::map<std::pair<int, int>, long long int> GTransMap;
	for (int i = 0; i < scRet.G.numRows(); i++)
	{
		int start = scRet.G.rowOffsets()[i];
		int end = scRet.G.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
			GTransMap.insert(std::make_pair(std::make_pair(scRet.G.colIndices()[j], i), scRet.G.values()[j]));
	}
	compressed_matrix::CompressedMatrix<long long int> GTrans(scRet.G.numRows(), scRet.G.numRows(), GTransMap);

	std::vector<int> outDegrees(scRet.G.numRows(), -1);
	for (int i = 0; i < classifiedSC.size(); i++)
	{
		if (classifiedSC[i] == 1) //non self-complement components
		{
			outDegrees[i] = scRet.G.rowOffsets()[i + 1] - scRet.G.rowOffsets()[i];
			if (outDegrees[i] == 0 && i != x0Location && i != x0BarLocation) //need to push the original outdegree 0 nodes into q !!!
				q.push(i);
		}
	}
	outDegrees[x0Location] = -1;
	outDegrees[x0BarLocation] = -1;

	for (int i = 0; i < visited.size(); i++)
	{
		if (visited[i] == 2) //exclude x0
		{
			for (int k = 0; k < scRet.S[i].positive.size(); k++)
				ret.push_back(std::make_pair(scRet.S[i].positive[k], 1));

			for (int k = 0; k < scRet.S[i].negative.size(); k++)
				ret.push_back(std::make_pair(scRet.S[i].negative[k], 0));

			int complement = complementPairs[i];

			outDegrees[i] = -1;
			outDegrees[complement] = -1;
		}
	}


	//decrease the outdegrees of node which has outgoing edges to i and to complement
	//push node which has 0 outdegrees into the queue
	for (int i = 0; i < visited.size(); i++)
	{
		if (visited[i] == 2) //exclude x0
		{
			int start = GTrans.rowOffsets()[i];
			int end = GTrans.rowOffsets()[i + 1];
			for (int k = start; k < end; k++)
			{
				if (outDegrees[GTrans.colIndices()[k]] > 0)
				{
					--outDegrees[GTrans.colIndices()[k]];
					if (outDegrees[GTrans.colIndices()[k]] == 0)
						q.push(GTrans.colIndices()[k]);
				}
			}

			int complement = complementPairs[i];
			start = GTrans.rowOffsets()[complement];
			end = GTrans.rowOffsets()[complement + 1];
			for (int k = start; k < end; k++)
			{
				if (outDegrees[GTrans.colIndices()[k]] > 0)
				{
					--outDegrees[GTrans.colIndices()[k]];
					if (outDegrees[GTrans.colIndices()[k]] == 0)
						q.push(GTrans.colIndices()[k]);
				}
			}
		}
	}


	while (!q.empty())
	{
		int curr = q.front(); q.pop();
		if (outDegrees[curr] == 0)
		{
			outDegrees[curr] = -1;
			int complement = complementPairs[curr];
			outDegrees[complement] = -1;

			//fixed all variables in component curr
			for (int k = 0; k < scRet.S[curr].positive.size(); k++)
				ret.push_back(std::make_pair(scRet.S[curr].positive[k], 1));

			for (int k = 0; k < scRet.S[curr].negative.size(); k++)
				ret.push_back(std::make_pair(scRet.S[curr].negative[k], 0));

			//decrease the outdegrees of node which has outgoing edges to i and to complement
			//push node which has 0 outdegrees into the queue
			int start = GTrans.rowOffsets()[curr];
			int end = GTrans.rowOffsets()[curr + 1];
			for (int k = start; k < end; k++)
			{
				if (outDegrees[GTrans.colIndices()[k]] > 0)
				{
					--outDegrees[GTrans.colIndices()[k]];
					if (outDegrees[GTrans.colIndices()[k]] == 0)
						q.push(GTrans.colIndices()[k]);
				}
			}

			start = GTrans.rowOffsets()[complement];
			end = GTrans.rowOffsets()[complement + 1];
			for (int k = start; k < end; k++)
			{
				if (outDegrees[GTrans.colIndices()[k]] > 0)
				{
					--outDegrees[GTrans.colIndices()[k]];
					if (outDegrees[GTrans.colIndices()[k]] == 0)
						q.push(GTrans.colIndices()[k]);
				}
			}
		}
	}

	std::sort(ret.begin(), ret.end(), compClass());

	return ret;
}

compressed_matrix::CompressedMatrix<double> computeNewQAndOffset(const compressed_matrix::CompressedMatrix<double>& Q, const std::vector<std::pair<int, int> >& fixed, double& offset)
{
	//fixed now is 1-based
	if (!fixed.empty())
	{
		int numVariables = Q.numRows();

		std::map<int, int> mI;
		for (int i = 0; i < fixed.size(); i++)
			mI[fixed[i].first - 1] = i; //-1 to make it 0-based

		std::vector<int> J;
		std::map<int, int> mJ;
		int cnt = 0;
		for (int i = 0; i < numVariables; i++)
		{
			if (mI.find(i) == mI.end())
			{
				J.push_back(i);
				mJ[i] = cnt++;
			}
		}

		std::map<std::pair<int, int>, double> QII;
		std::map<std::pair<int, int>, double> QJJ;
		std::map<std::pair<int, int>, double> QIJ;
		std::map<std::pair<int, int>, double> QJI;

		for (int i = 0; i < Q.numRows(); i++)
		{
			int start = Q.rowOffsets()[i];
			int end = Q.rowOffsets()[i + 1];
			for (int j = start; j < end; j++)
			{
				int r = i;
				int c = Q.colIndices()[j];
				if (mI.find(r) != mI.end() && mI.find(c) != mI.end())
					QII[std::make_pair(mI[r], mI[c])] = Q.values()[j];
				if (mJ.find(r) != mJ.end() && mJ.find(c) != mJ.end())
					QJJ[std::make_pair(mJ[r], mJ[c])] = Q.values()[j];
				if (mI.find(r) != mI.end() && mJ.find(c) != mJ.end())
					QIJ[std::make_pair(mI[r], mJ[c])] = Q.values()[j];
				if (mJ.find(r) != mJ.end() && mI.find(c) != mI.end())
					QJI[std::make_pair(mJ[r], mI[c])] = Q.values()[j];
			}
		}

		//off_set = x0'*Q_II*x0;
		offset = 0;
		for (std::map<std::pair<int, int>, double>::const_iterator it = QII.begin(), end = QII.end(); it != end; ++it)
		{
			int r = it->first.first;
			int c = it->first.second;
			offset += fixed[r].second * it->second * fixed[c].second;
		}

		//x0'*Q_IJ
		std::vector<double> tmp2(numVariables - fixed.size(), 0);
		for (std::map<std::pair<int, int>, double>::const_iterator it = QIJ.begin(), end = QIJ.end(); it != end; ++it)
		{
			int r = it->first.first;
			int c = it->first.second;
			tmp2[c] += it->second * fixed[r].second;
		}

		//Q_JI*x0
		std::vector<double> tmp3(numVariables - fixed.size(), 0);
		for (std::map<std::pair<int, int>, double>::const_iterator it = QJI.begin(), end = QJI.end(); it != end; ++it)
		{
			int r = it->first.first;
			int c = it->first.second;
			tmp3[r] += it->second * fixed[c].second;
		}

		for (int i = 0; i < (int)tmp3.size(); i++)
			tmp2[i] += tmp3[i];

		cnt = 0;
		for (int i = 0; i < (int)mJ.size() * (int)mJ.size(); i += numVariables - static_cast<int>(fixed.size()) + 1)
		{
			int r = i / static_cast<int>(mJ.size());
			int c = i % mJ.size();
			QJJ[std::make_pair(r, c)] += tmp2[cnt++];
		}

		std::map<std::pair<int, int>, double> mfixedQ;
		for (std::map<std::pair<int, int>, double>::const_iterator it = QJJ.begin(), end = QJJ.end(); it != end; ++it)
		{
			int r = it->first.first;
			int c = it->first.second;
			if (it->second != 0)
				mfixedQ[std::make_pair(J[r], J[c])] = it->second;
		}

		return compressed_matrix::CompressedMatrix<double>(numVariables, numVariables, mfixedQ);
	}
	else
	{
		offset = 0;
		return Q;
	}
}

std::vector<std::pair<int, int> > applyImplication(const compressed_matrix::CompressedMatrix<long long int>& A)
{
	int numVertices = A.numRows();
	int numVariables = numVertices / 2 - 1;

	//debuging only
	//clock_t curr_1 = clock();
	//clock_t curr_2;

	using namespace boost;

	typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
	typedef adjacency_list<vecS, vecS, directedS, property<vertex_name_t, std::string>, property<edge_capacity_t, long long int, property<edge_residual_capacity_t, long long int, property<edge_reverse_t, Traits::edge_descriptor> > > > Graph; //for edge capacity is long long int
	//typedef adjacency_list<vecS, vecS, directedS, property<vertex_name_t, std::string>, property<edge_capacity_t, double, property<edge_residual_capacity_t, double, property<edge_reverse_t, Traits::edge_descriptor> > > > Graph; //for edge capacity is double

	Graph g;

	property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
	property_map<Graph, edge_reverse_t>::type reverse_edge = get(edge_reverse, g);
	property_map<Graph, edge_residual_capacity_t>::type residual_capacity = get(edge_residual_capacity, g);

	std::vector<Traits::vertex_descriptor> verts(numVertices);
	for (int i = 0; i < numVertices; ++i)
		verts[i] = add_vertex(g);

	Traits::vertex_descriptor s = verts[numVariables];
	Traits::vertex_descriptor t = verts[2 * numVariables + 1];

	for (int i = 0; i < A.numRows(); i++)
	{
		int start = A.rowOffsets()[i];
		int end = A.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = A.colIndices()[j];
			long long int cap = A.values()[j];

			Traits::edge_descriptor e1, e2;
			bool in1, in2;
			boost::tie(e1, in1) = add_edge(verts[r], verts[c], g);
			boost::tie(e2, in2) = add_edge(verts[c], verts[r], g);

			capacity[e1] = cap;
			capacity[e2] = 0;
			reverse_edge[e1] = e2;
			reverse_edge[e2] = e1;
		}
	}

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_building_graph: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	push_relabel_max_flow(g, s, t);

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_max_flow: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	//residual
	std::vector<int> RRowOffsets(numVertices+1);
	std::vector<int> RColIndices;
	std::vector<long long int> RValues;
	//flow
	std::vector<int> FRowOffsets(numVertices+1);
	std::vector<int> FColIndices;
	std::vector<long long int> FValues;

	int offset = 0;
	int currRow = 0;
	graph_traits<Graph>::vertex_iterator u_iter, u_end;
	graph_traits<Graph>::out_edge_iterator ei, e_end;
	for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
		for (boost::tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
		{
			if (capacity[*ei] > 0)
			{
				if (currRow <= *u_iter)
				{
					for (int i = currRow; i <= *u_iter; i++)
					{
						RRowOffsets[i] = offset;
						FRowOffsets[i] = offset;
					}
					currRow = static_cast<int>((*u_iter) + 1);
				}
				RColIndices.push_back(static_cast<int>(target(*ei, g)));
				RValues.push_back(residual_capacity[*ei]);
				FColIndices.push_back(static_cast<int>(target(*ei, g)));
				FValues.push_back(capacity[*ei] - residual_capacity[*ei]);
				++offset;
			}
		}
	for (int i = currRow; i < RRowOffsets.size(); i++)
		RRowOffsets[i] = offset;
	for (int i = currRow; i < FRowOffsets.size(); i++)
		FRowOffsets[i] = offset;

	compressed_matrix::CompressedMatrix<long long int> R(numVertices, numVertices, RRowOffsets, RColIndices, RValues); //residual
	compressed_matrix::CompressedMatrix<long long int> F(numVertices, numVertices, FRowOffsets, FColIndices, FValues); //flow

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_get_RF: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	//resid = (A>0).*R + F'.*(A'>0);
	std::map<std::pair<int, int>, long long int> residM;
	for (int i = 0; i < A.numRows(); i++)
	{
		int start = A.rowOffsets()[i];
		int end = A.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = A.colIndices()[j];
			long long int RValue = R.get(r, c);
			if (A.values()[j] > 0 && RValue != 0 && r != 2 * numVariables + 1 && c != numVariables)
				residM[std::make_pair(r, c)] += RValue;

			long long int FTValue = F.get(r, c); //not F(c, r) !!!
			if (A.values()[j] > 0 && FTValue != 0 && c != 2 * numVariables + 1 && r != numVariables)
				residM[std::make_pair(c, r)] += FTValue;
		}
	}

	compressed_matrix::CompressedMatrix<long long int> resid(numVertices, numVertices, residM);

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_finally_get_resid: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	std::vector<int> forced = bfs_for_method_2(resid, numVariables);

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_bfs_for_method_2: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	std::vector<std::pair<int, int> > fixed;
	for (int i = 0; i < forced.size(); i++)
	{
		if (forced[i] > 0)
		{
			if (i <= numVariables - 1)
				fixed.push_back(std::make_pair(i + 1, 1)); //1-based
			else if (i >= numVariables + 1 && i <= 2 * numVariables)
				fixed.push_back(std::make_pair(i - numVariables, 0)); //1-based
		}
	}

	std::sort(fixed.begin(), fixed.end(), compClass());

	//curr_2 = clock();
	//mexPrintf("inside applyImplication int version: Time elapsed_fix_vars_and_sorting: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	return fixed;
}

} // anonymous namespace

namespace fix_variables_
{

FixVariablesResult fixQuboVariables(const compressed_matrix::CompressedMatrix<double>& Q, int method)
{
	//Q needs to be a square matrix
	if (Q.numRows() != Q.numCols())
        throw std::invalid_argument("Q's size is not correct.");

	if (!(method == 1 || method == 2))
        throw std::invalid_argument("method must be an integer of 1 or 2.");

	FixVariablesResult ret;

	//check if Q is empty
	if (Q.numRows() == 0 || Q.numCols() == 0)
	{
		ret.offset = 0;
		return ret;
	}

	int numVariables = Q.numRows();

	//clock_t curr_1 = clock();
	//clock_t curr_2;

	//uTriQ = triu(Q) + tril(Q,-1)'; //make upper triangular
	std::map<std::pair<int, int>, double> uTriQMap;
	std::set<int> usedVariables;
	for (int i = 0; i < Q.numRows(); i++)
	{
		int start = Q.rowOffsets()[i];
		int end = Q.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
		{
			int r = i;
			int c = Q.colIndices()[j];
			uTriQMap[std::make_pair(std::min(r, c), std::max(r, c))] += Q.values()[j];
			usedVariables.insert(r);
			usedVariables.insert(c);
		}
	}

	compressed_matrix::CompressedMatrix<double> uTriQ(numVariables, numVariables, uTriQMap);

	double maxAbsValue = 0;

	if (!uTriQ.values().empty())
		maxAbsValue = std::fabs(*std::max_element(uTriQ.values().begin(), uTriQ.values().end(), compareAbs));

	double ratio = 1;

	if (maxAbsValue != 0)
		ratio = static_cast<double>(std::numeric_limits<long long int>::max()) / maxAbsValue;

	ratio /= static_cast<double>(1LL << 10);

	if (ratio < 1)
		ratio = 1;

	std::map<std::pair<int, int>, long long int> uTriQMapLLI;

	for (std::map<std::pair<int, int>, double>::iterator it = uTriQMap.begin(), end = uTriQMap.end(); it != end; ++it)
		uTriQMapLLI[it->first] = static_cast<long long int>(it->second * ratio);

	compressed_matrix::CompressedMatrix<long long int> uTriQLLI(numVariables, numVariables, uTriQMapLLI);

	//curr_2 = clock();
	//mexPrintf("Time elapsed_make upper triangular: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	Posiform p = BQPToPosiform(uTriQLLI);

	//curr_2 = clock();
	//mexPrintf("Time elapsed_BQPToPosiform: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	if (method == 1)
	{
		compressed_matrix::CompressedMatrix<long long int> A = posiformToImplicationNetwork_1(p);

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_posiformToImplicationNetwork_1: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		compressed_matrix::CompressedMatrix<long long int> R = maxFlow(A);  //use this

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_maxFlow: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		compressed_matrix::CompressedMatrix<long long int> symR = makeResidualSymmetric(R); //use this

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_makeRSym: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		//add clearing R in makeResidualSym, so here just use symR directly !!!
		SCRet scRet = stronglyConnectedComponents(symR);


		//curr_2 = clock();
	    //mexPrintf("Time elapsed_stronglyConnectedComponents: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		std::vector<int> classifiedSC = classifyStronglyConnectedComponents(scRet.S);

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_classifyStrongComponents: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		ret.fixedVars = fixVariables(classifiedSC, scRet, numVariables);

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_fixVarsUsingOutDegree: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;
	}
	else if (method == 2)
	{
		compressed_matrix::CompressedMatrix<long long int> A = posiformToImplicationNetwork_2(p);

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_posiformToImplicationNetwork_2: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;

		ret.fixedVars = applyImplication(A);

		//curr_2 = clock();
	    //mexPrintf("Time elapsed_applyImplication: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	    //curr_1 = curr_2;
	}

	if (ret.fixedVars.size() > numVariables)
        throw std::logic_error("ret.fixedVars has wrong size.");

	// remove unused variables from ret.fixedVars
	std::vector<std::pair<int, int> > updatedFixedVars;
	for (int i = 0; i < ret.fixedVars.size(); ++i)
	{
		if (usedVariables.find(ret.fixedVars[i].first - 1) != usedVariables.end()) // -1 to make it 0-based since usedVariables is 0-based
			updatedFixedVars.push_back(ret.fixedVars[i]);
	}

	ret.fixedVars = updatedFixedVars;

	ret.newQ = computeNewQAndOffset(uTriQ, ret.fixedVars, ret.offset);

	//curr_2 = clock();
	//mexPrintf("Time elapsed_computeNewQAndOffset: %f\n", ((double)curr_2 - curr_1) / CLOCKS_PER_SEC);
	//curr_1 = curr_2;

	return ret;
}


std::vector<std::pair<int,  int> > fixQuboVariablesMap(std::map<std::pair<int, int>, double> QMap, int QSize, int mtd)
{
    compressed_matrix::CompressedMatrix<double> QInput(QSize, QSize, QMap);

    FixVariablesResult ret = fixQuboVariables(QInput, mtd);

    return ret.fixedVars;
}


} // namespace fix_variables_
