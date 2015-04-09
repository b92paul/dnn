# File system

- data/ (data main dir)
-	data/fbank
- data/label
- data/mfcc
- data/phones
- data/merge (create by your self)
- dnn/ (git main dir)
- dnn/hw1 (all hw1 file in here)
- dnn/hw1/out (test prdiction csv file will generate to this dir)
- dnn/hw1/save_models (for model saving)

# Data preprocessing

1. create merge dir in data/
2. run read.py
3. run fbank.py
4. run mfcc.py
5. run l48to39.py

# Compile main program

1. Download Eigen from [Eigen main website](http://eigen.tuxfamily.org/)
2. link `path/to/Eigen` to your include path
3. go to hw1 dir
4. type "make main" in hw1
	- You could also use `g++ -O2 --std=c++11 main.cpp -o main -Ipath/to/Eigen` if you don't want to include Eigen in your include path.

# Run main program

1. run main by ./main
2. it will output test prediction for every 5000 mini-batchs to out/ in default settings

# For concatenated frame feature training

1. run new_fbank.py
2. run new_l48to39.py
3. set the parameters in new_main.cpp
4. type `make new_main`
5. run new_make by `./new_main`
