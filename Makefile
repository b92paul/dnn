main:
	make -C hw2/Dnn
run:
	make run -C hw2/Dnn
	cp hw2/Dnn/ans/* .
svm:
	make -C hw2/svm
crf:
	make -C hw2/crf
pla:
	make -C hw2/pla
clean:
	make clean -C hw2/
