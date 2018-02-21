I = -I/opt/local/include
L = -L/opt/local/lib

ardop : ardop.cc
	c++ -g -std=c++11 $I -o ardop ardop.cc $L -lsndfile -lfftw3

clean :
	rm -f ardop
