CC := clang
CC_FLAGS :=

ifeq ($(DEBUG),1)
	CFLAGS += -O0
else
	CFLAGS += -O3 -DNDEBUG
endif

all:
	$(CC) $(CC_FLAGS) main.c -o main

clean:
	rm main