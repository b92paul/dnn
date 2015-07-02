make load_and_test
./load_ans_test $1
python hw1/combine_id_label.py hw1/out/test_$1 ../../data/final/test_id_hw1.ark ../../data/final/state_48_39.map > hw1/ans/ans_$1
wc -l hw1/ans/ans_$1
