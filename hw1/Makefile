main:
	g++ -O2 --std=c++11 main.cpp -o main
new_main: new_main.cpp
	g++ -O2 --std=c++11 new_main.cpp -o new_main
omp_new_main: new_main.cpp
	g++ -O2 --std=c++11 new_main.cpp -o omp_new_main -fopenmp
