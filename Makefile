CXX=g++
CC=g++
CXXFLAGS=-Wall -Wextra -Werror -std=c++14 -pedantic
LDLIBS=-lm

VPATH=src
SRC=main.cc
OBJ=$(SRC:.cc=.o)
BIN=main

.PHONY: all
all: $(BIN)

$(BIN): $(OBJ)

.PHONY:debug
debug: CXXFLAGS+= -g3
debug: $(BIN)

.PHONY: clean
clean:
	$(RM) $(BIN)
	$(RM) $(OBJ)
