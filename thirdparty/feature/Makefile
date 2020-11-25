# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

CC           ?= gcc
AR           ?= ar
CPPFLAGS     += -DNDEBUG -I..
CFLAGS       += -std=c11 -g -Wall -Wextra -Werror -pedantic -O3
OBJECTS      := hash_feature.o xxhash.o
LIBRARIES    := libfeature.a

all: $(LIBRARIES)
.PHONY: all

hash_feature.o: hash_feature.c
	$(CC) -c -o $@ $^ $(CPPFLAGS) $(CFLAGS)

xxhash.o: ../lz4-1.9.2/xxhash.c
	$(CC) -c -o $@ $^ $(CPPFLAGS) $(CFLAGS)

libfeature.a: $(OBJECTS)
	$(AR) rcs $@ $^

clean:
	rm -f $(OBJECTS) $(LIBRARIES)
.PHONY: clean
