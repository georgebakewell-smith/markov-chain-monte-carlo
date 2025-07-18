CC = gcc
CFALGS = -g -Wall
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
	$(CC) $(CFALGS) -c $< -o $@



clean:
	rm -f $(BINDIR)/* $(OBJ)/*
