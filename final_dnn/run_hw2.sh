#python hmm/sentence_len.py ../../data/final/test_id_hw1.ark > hmm/sentence_name_len.out
#make load_and_genprob
#./load_and_genprob $1 ../../data/final/f_test_hw1.ark
#make -C hmm
#./hmm/viterbi $1 $1
python hmm/hmm_to_ans.py hmm/out/$1 hmm/ans/$1.csv
