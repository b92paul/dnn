CC = gcc
LD = gcc

CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS =  $(SFLAGS) -O3 -lm -Wall
LIBS = -lm
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

all: struct_perceptron.c 
	make -C ../svm/
	$(LD) $(LDFLAGS) -I../svm/ -w struct_perceptron.c -o struct_perceptron
clean:
		rm struct_perceptron
