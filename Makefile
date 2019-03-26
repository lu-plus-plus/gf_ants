
CC = nvcc

CCFLAGS += -std=c++14 \
	-Xcudafe "--diag_suppress=2947" \
	-O2

EXEC = exec_gf
OBJS = main.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(CCFLAGS)

main.o: main.cu cuder.h gf_int.h gf_matrix.h
	$(CC) -c $< -o $@ $(CCFLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)