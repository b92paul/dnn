echo 'run parse_question.py'
python parse_question_sim.py
echo 'run word2vec and calc cos'
./word2vec-read-only/distance word2vec-read-only/novel.bin < testing_parse.txt > cos.txt
echo 'run cos_to_ans.py'
python cos_to_ans.py
