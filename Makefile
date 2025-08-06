CC = gcc
CFLAGS = -g -Wall -std=c99
SRC = src
OBJ = obj
SRCS = $(wildcard $(SRC)/*.c)
OBJS = $(patsubst $(SRC)/*.c, $(OBJ)/*.o, $(SRCS))
BINDIR = bin
BIN = $(BINDIR)/main


# Link with GSL and math libraries
LDFLAGS = -lgsl -lgslcblas -lm

all: $(BIN)


$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm -f $(BINDIR)/* $(OBJ)/*
