CC      := gcc
NVCC    := nvcc
CFLAGS  := -O3 -Iinclude
LDFLAGS := -lcudart -lm
TARGET  := dt_cuda

SRCS    := src/main.c src/decision_tree.c src/decision_tree_cuda.cu
OBJS    := main.o decision_tree.o decision_tree_cuda.o

.PHONY: all clean

all: $(TARGET)

main.o: src/main.c include/decision_tree.h
	$(CC) $(CFLAGS) -c $< -o $@

decision_tree.o: src/decision_tree.c include/decision_tree.h
	$(CC) $(CFLAGS) -c $< -o $@

decision_tree_cuda.o: src/decision_tree_cuda.cu include/decision_tree.h
	$(NVCC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
