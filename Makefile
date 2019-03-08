
CC = nvcc

CPPFLAGS += -std=c++11

EXEC = exec_gf
OBJS = gf_int.o



.PHONY: all, run
all: $(EXEC)

run: $(EXEC)
	./$(EXEC)



$(EXEC): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS)

gf_int.o: gf_int.cu gf_int.h
	$(CC) -c $< -o $@ $(CPPFLAGS)



.PHONY: clean
clean:
	-rm $(EXEC) $(OBJS)