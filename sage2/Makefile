# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

SOURCE_ROOT  := $(shell pwd)
BUILD_ROOT   := $(shell pwd)
AS           := gcc
CC           ?= gcc
CXX          ?= g++
AR           ?= ar
CPPFLAGS     += -I$(SOURCE_ROOT)/include -I$(SOURCE_ROOT)/src -I$(SOURCE_ROOT)/thirdparty -DXBYAK_NO_OP_NAMES=1
ASFLAGS      +=
CFLAGS       += -pthread -std=c11 -g -Wall -Wextra -Werror -pedantic
CXXFLAGS     += -pthread -std=c++11 -g -Wall -Wextra -Werror -pedantic
LDFLAGS      += -pthread
MACHINE      := $(shell $(CXX) $(CPPFLAGS) $(CXXFLAGS) -dumpmachine)
BUILD_DIR    := build_$(MACHINE)
BUILD_DIR_ABS = $(BUILD_ROOT)/$(BUILD_DIR)
PREFIX       := /usr/local

OS_DARWIN    := 0
OS_LINUX     := 0

ifeq ($(findstring darwin,$(MACHINE)),darwin)
OS_DARWIN    := 1
endif

ifeq ($(findstring linux,$(MACHINE)),linux)
OS_LINUX     := 1
LDFLAGS      += -ldl
endif

DEBUG        ?= 0
ifeq ($(DEBUG),1)
CFLAGS       += -O0
CXXFLAGS     += -O0 -Og
BUILD_DIR    := $(BUILD_DIR)_d
else
CPPFLAGS     += -DNDEBUG
CFLAGS       += -O3
CXXFLAGS     += -O3
BUILD_DIR    := $(BUILD_DIR)_r
endif

SIMD         ?= 0
ifeq ($(SIMD),1)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59800
CFLAGS       := $(filter-out -Werror,$(CFLAGS))
CXXFLAGS     := $(filter-out -Werror,$(CXXFLAGS))
CFLAGS       += -ftree-vectorize -ffast-math -mavx -mfma -mavx2
CXXFLAGS     += -ftree-vectorize -ffast-math -mavx -mfma -mavx2
endif

ifeq ($(EXPORT_MEMCPY),1)
CPPFLAGS     += -DEXPORT_MEMCPY=1
endif

AS_SOURCES   := $(shell find src -type f -name "*.S" | sort)
C_SOURCES    := $(shell find src -type f -name "*.c" | sort)
CXX_SOURCES  := $(shell find src -type f -name "*.cc" | sort)
SOURCES      := $(AS_SOURCES) $(C_SOURCES) $(CXX_SOURCES)
TEST_SOURCES := $(shell find src -type f -name "*_test.c" -or -name "*_test.cc" | sort)
BIN_SOURCES  := $(shell find src -type f -name "*_main.c" -or -name "*_main.cc" | sort)
LIB_SOURCES  := $(filter-out $(TEST_SOURCES) $(BIN_SOURCES),$(SOURCES))
LINT_C_SOURCES := $(C_SOURCES)
LINT_CXX_SOURCES := $(CXX_SOURCES)
LINT_SOURCES := $(LINT_C_SOURCES:.c=.lint) $(LINT_CXX_SOURCES:.cc=.lint)

ifeq ($(filter $(MAKECMDGOALS),clean lint),)
DEPENDS      := $(addprefix $(BUILD_DIR_ABS)/,$(SOURCES))
DEPENDS      := $(DEPENDS:.S=.d)
DEPENDS      := $(DEPENDS:.c=.d)
DEPENDS      := $(DEPENDS:.cc=.d)
else
DEPENDS      :=
endif

LIB_OBJECTS  := $(addprefix $(BUILD_DIR_ABS)/,$(LIB_SOURCES))
LIB_OBJECTS  := $(LIB_OBJECTS:.S=.o)
LIB_OBJECTS  := $(LIB_OBJECTS:.c=.o)
LIB_OBJECTS  := $(LIB_OBJECTS:.cc=.o)

LIBRARIES    := $(BUILD_DIR_ABS)/libsage2.a
TESTS        := $(TEST_SOURCES:src/%.c=$(BUILD_DIR_ABS)/%)
TESTS        := $(TESTS:src/%.cc=$(BUILD_DIR_ABS)/%)
BINARIES     := $(BIN_SOURCES:src/%_main.c=$(BUILD_DIR_ABS)/%)
BINARIES     := $(BINARIES:src/%_main.cc=$(BUILD_DIR_ABS)/%)

################################################################

all: $(LIBRARIES) $(TESTS) $(BINARIES)
	@echo "******************************************"
	@echo "Build succsessfully at $(BUILD_DIR_ABS)"
	@echo "AS:          " $(AS)
	@echo "CC:          " $(CC)
	@echo "CXX:         " $(CXX)
	@echo "AR:          " $(AR)
	@echo "CPPFLAGS:    " $(CPPFLAGS)
	@echo "ASFLAGS:     " $(ASFLAGS)
	@echo "CFLAGS:      " $(CFLAGS)
	@echo "CXXFLAGS:    " $(CXXFLAGS)
	@echo "LDFLAGS:     " $(LDFLAGS)
	@echo "******************************************"
	@echo "DEBUG:       " $(DEBUG)
	@echo "SIMD:        " $(SIMD)
	@echo "******************************************"
.PHONY: all

lib: $(LIBRARIES)
	@echo "******************************************"
	@echo "Build succsessfully at $(BUILD_DIR_ABS)"
	@echo "AS:          " $(AS)
	@echo "CC:          " $(CC)
	@echo "CXX:         " $(CXX)
	@echo "AR:          " $(AR)
	@echo "CPPFLAGS:    " $(CPPFLAGS)
	@echo "ASFLAGS:     " $(ASFLAGS)
	@echo "CFLAGS:      " $(CFLAGS)
	@echo "CXXFLAGS:    " $(CXXFLAGS)
	@echo "LDFLAGS:     " $(LDFLAGS)
	@echo "******************************************"
	@echo "DEBUG:       " $(DEBUG)
	@echo "SIMD:        " $(SIMD)
	@echo "******************************************"
.PHONY: lib

clean:
	@echo Cleaning $(BUILD_DIR_ABS)
	@rm -rf $(BUILD_DIR_ABS)
.PHONY: clean

test: $(TESTS)
	@for t in $(TESTS); do \
		echo $$t; \
		$$t 1; \
	done
.PHONY: test

perftest: $(TESTS)
	@for t in $(TESTS); do \
		echo $$t; \
		$$t 2; \
	done
.PHONY: perftest

lint: $(LINT_SOURCES)
.PHONY: lint

install: lib
	@bash script/install.sh $(PREFIX) $(BUILD_DIR_ABS)
.PHONY: install

build_dir_abs:
	@echo $(BUILD_DIR_ABS)
.PHONY: build_dir_abs

################################################################

$(BUILD_DIR_ABS)/src/%.o: src/%.S
	@echo Assembling $<
	@mkdir -p $(@D)
	@$(AS) $(CPPFLAGS) $(ASFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS)/src/%.o: src/%.c
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS)/src/%.o: src/%.cc
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS)/src/%.d: src/%.S
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(AS) -MM $(CPPFLAGS) $(ASFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

$(BUILD_DIR_ABS)/src/%.d: src/%.c
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CC) -MM $(CPPFLAGS) $(CFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

$(BUILD_DIR_ABS)/src/%.d: src/%.cc
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

-include $(DEPENDS)

src/%.lint: src/%.c
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet $< -- $(CPPFLAGS) $(CFLAGS)

src/%.lint: src/%.cc
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet $< -- $(CPPFLAGS) $(CXXFLAGS)
.PHONY: src/%.lint

################################################################

$(BUILD_DIR_ABS)/libsage2.a: $(LIB_OBJECTS)
	@echo Archiving $@
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^

$(BUILD_DIR_ABS)/%_test: \
$(BUILD_DIR_ABS)/src/%_test.o \
$(BUILD_DIR_ABS)/libsage2.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $^ $(LDFLAGS)
