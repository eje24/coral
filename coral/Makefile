CC := clang

TARGET := main
TEST_TARGET := test

SRC = main.c variable.c tensor.c grad.c shape.c
TEST_SRC = test.c variable.c tensor.c grad.c shape.c

OBJ := $(SRC:.c=.o)
TEST_OBJ := $(TEST_SRC:.c=.o)

COMMONFLAGS := -Wall -Werror -Wextra
CFLAGS := $(COMMONFLAGS) -std=gnu99 -g -flto
LDFLAGS := $(COMMONFLAGS) -lm -ldl -flto

ifeq ($(DEBUG),1)
	CFLAGS += -O0
else
	CFLAGS += -O3 -DNDEBUG
endif

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@

$(TEST_TARGET): $(TEST_OBJ)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c Makefile
	$(CC) $(CFLAGS) -MMD -c $< -o $@

clean:
	rm -f *.o *.d main test