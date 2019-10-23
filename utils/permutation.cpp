#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;



//template <typename T> void swap(T& i, T& j)
//{
//		T k(std::move(i));
//		i = std::move(j);
//		j = std::move(k);
//}

void swap(int& i, int& j)
{
		int tmp = i;
		i = j;
		j = tmp;
}

using VInt = std::vector<int>;
using VVInt = std::vector<std::vector<int> >;


void permute(VInt& in, int n, VVInt& out)
{
		if ( n == 1)
		{
				out.push_back(in);
				return;
		}
		for (int i = 0; i < (n - 1); ++i)
		{
				permute(in, n - 1, out);
				if (n % 2 == 0)
				{
						swap(in[n - 1], in[i]);	
				}
				else
				{
						swap(in[n - 1], in[0]);
				}
		}
		permute(in, n - 1, out);

}

auto REF = py::return_value_policy::reference;
auto CPY = py::return_value_policy::copy;

PYBIND11_MAKE_OPAQUE(VInt);
PYBIND11_MAKE_OPAQUE(VVInt);

PYBIND11_MODULE(permutation, m)
{
		m.doc() = "pybind permutation";
		py::bind_vector<VInt> (m, "VInt");
		py::bind_vector<VVInt> (m, "VVInt");
		m.def("permute", &permute);
}

