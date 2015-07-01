main: main.cpp dnn.cpp
	g++ -O3 --std=c++11 main.cpp -o main
omp_main: main.cpp dnn.cpp
	g++ -O3 --std=c++11 main.cpp -o main -fopenmp
load_and_test: load_and_test.cpp dnn.cpp
	g++ -O3 --std=c++11 load_and_test.cpp -o load_and_test -fopenmp
load_and_genprob: load_and_genprob.cpp dnn.cpp
	g++ -O3 --std=c++11 load_and_genprob.cpp -o load_and_genprob -fopenmp
