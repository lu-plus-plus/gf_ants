
CC = nvcc

CPPFLAGS += -std=c++14 \
	-Xcudafe "--diag_suppress=2947"

EXEC = exec_gf
OBJS = main.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS)

main.o: main.cu cuder.h gf_int.h gf_matrix.h
	$(CC) -c $< -o $@ $(CPPFLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)