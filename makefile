NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30

#Optimization flags. Don't use this for debugging.
NVCCFLAGS = -c -m64 -O0 --compiler-options -Wall -Xptxas -O0,-v

#No optimizations. Debugging flags. Use this for debugging.
#NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = wrappers.o vecScalarMult.o h_vecScalarMult.o d_vecScalarMult.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

vecScalarMult: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o vecScalarMult

vecScalarMult.o: vecScalarMult.cu h_vecScalarMult.h d_vecScalarMult.h config.h

h_vecScalarMult.o: h_vecScalarMult.cu h_vecScalarMult.h CHECK.h

d_vecScalarMult.o: d_vecScalarMult.cu d_vecScalarMult.h CHECK.h config.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm vecScalarMult *.o
