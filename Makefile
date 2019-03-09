
CC = nvcc

CPPFLAGS += -std=c++14

EXEC = exec_gf
OBJS = main.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS)

main.o: main.cu gf_int.h xint.h
	$(CC) -c $< -o $@ $(CPPFLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)