main:
	make -C hw2/Dnn
run:
	make run -C hw2/Dnn
	cp hw2/Dnn/ans/* .
