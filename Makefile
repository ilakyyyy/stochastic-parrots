CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lm

ifeq ($(OS),Windows_NT)
	EXE = .exe
	PLATFORM_SRC = win.c
else
	EXE =
	PLATFORM_SRC =
endif

CORE_SRC = dynamics.c runq.c sampler.c tokenizer.c transformer.c utils.c $(PLATFORM_SRC)

.PHONY: all tests test clean

all: parrot$(EXE) analyze$(EXE)

parrot$(EXE): parrot.c $(CORE_SRC)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

analyze$(EXE): analyze.c $(CORE_SRC)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

tests_runner$(EXE): tests/public_tests.c $(CORE_SRC)
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@

test: tests_runner$(EXE)
	./tests_runner$(EXE)

clean:
	rm -f *.o parrot$(EXE) analyze$(EXE) tests_runner$(EXE)
