
CC = nvcc

CPP_FLAGS += -std=c++11 -O2

CUDAC_FLAGS += -Xcudafe "--diag_suppress=2947"

ALL_FLAGS = $(CPP_FLAGS) $(CUDAC_FLAGS) $(SHELL_FLAGS)



EXEC = exec_gf
OBJS = main.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(ALL_FLAGS)

main.o: main.cu cuder.h gf_int.h gf_matrix.h
	$(CC) -c $< -o $@ $(ALL_FLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)