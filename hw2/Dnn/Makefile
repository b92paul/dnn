load_and_test: load_and_test.cpp new_fast_dnn.cpp
	g++ -O2 --std=c++11 load_and_test.cpp -o load_and_test
run:
	mkdir ../data
	mkdir ../data/merge
	mkdir ../data/phones
	make load_and_test
	wget -P ../../../data http://googledrive.com/host/0BxhV912_bTsXfkZZZ2ZiaDc4cWZicG0tYmZGeHMzc0VxbkRaZDRfWmRqV2g0NGhTc2RPRVU/test_0.ark
	wget -P ../../../data http://googledrive.com/host/0BxhV912_bTsXfkZZZ2ZiaDc4cWZicG0tYmZGeHMzc0VxbkRaZDRfWmRqV2g0NGhTc2RPRVU/48_idx_chr.map
	./load_and_test
	wget -P ../../../data/merge http://googledrive.com/host/0BxhV912_bTsXfkZZZ2ZiaDc4cWZicG0tYmZGeHMzc0VxbkRaZDRfWmRqV2g0NGhTc2RPRVU/label2.out
	wget -P ../../../data/merge http://googledrive.com/host/0BxhV912_bTsXfkZZZ2ZiaDc4cWZicG0tYmZGeHMzc0VxbkRaZDRfWmRqV2g0NGhTc2RPRVU/lmap2.out
	wget -P ../../../data/phones http://googledrive.com/host/0BxhV912_bTsXfkZZZ2ZiaDc4cWZicG0tYmZGeHMzc0VxbkRaZDRfWmRqV2g0NGhTc2RPRVU/48_39.map
	python viterbi.py
	python convert.py
