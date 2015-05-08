CC = gcc
LD = gcc

CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS =  $(SFLAGS) -O3 -lm -Wall
LIBS = -lm
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

all: crf.c 
		make -C ../svm/
		$(LD) $(LDFLAGS) -I../svm/ -w crf.c -o crf $(LIBS)
clean:
		rm crf
