
CC = nvcc

CPPFLAGS += -std=c++11

EXEC = exec_gf
OBJS = main.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS)

main.o: main.cu gf_int.h
	$(CC) -c $< -o $@ $(CPPFLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)