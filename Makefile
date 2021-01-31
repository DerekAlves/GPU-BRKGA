# Compiler binary:
NVCC= nvcc

# Recommended compiler flags for speed:
#	OpenMP enabled
#	full binary code optimization
#	full error and warning reports
#	no range checking within BRKGA:
FLAGS= -arch=sm_35 -rdc=true -O3 -w

# Compiler flags for debugging; uncomment if needed:
#	range checking enabled in the BRKGA API
#	OpenMP disabled
#	no binary code optimization
#CFLAGS= -DRANGECHECK -Wextra -Wall -Weffc++ -ansi -pedantic -Woverloaded-virtual -Wcast-align -Wpointer-arith

# Objects:
OBJECTS= api-usage.o SampleDecoder.o

LFLAGS= -lcurand

# Targets:
all: api-usage

api-usage: $(OBJECTS)
	$(NVCC) $(FLAGS) $(OBJECTS) -o api-usage $(LFLAGS)

api-usage.o:
	$(NVCC) $(FLAGS) -c api-usage.cu

SampleDecoder.o:
	$(NVCC) $(FLAGS) -c SampleDecoder.cu

clean:
	rm -f api-usage $(OBJECTS)
