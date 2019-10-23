g++  -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` permutation.cpp -o permutation`python3-config --extension-suffix`
