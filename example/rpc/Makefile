# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

ifeq ($(BUILD_DIR_ABS),)
$(error run "make" at project root directory)
endif

RELPATH      := example/rpc
BUILD_DIR_ABS_RPC := $(BUILD_DIR_ABS)/$(RELPATH)

SOURCES      := $(shell find . -type f -name "*.cc" | sort)
BIN_SOURCES  := $(shell find . -type f -name "*_main.cc" | sort)
LINT_SOURCES := $(SOURCES:.cc=.lint)

ifeq ($(filter $(MAKECMDGOALS),clean lint),)
DEPENDS      := $(addprefix $(BUILD_DIR_ABS_RPC)/,$(SOURCES))
DEPENDS      := $(DEPENDS:.cc=.d)
else
DEPENDS      :=
endif

BINARIES     := \
$(BIN_SOURCES:%_main.cc=$(BUILD_DIR_ABS_RPC)/%)

################################################################

all: $(BINARIES)
.PHONY: all

clean:
.PHONY: clean

test:
.PHONY: test

lint: $(LINT_SOURCES)
.PHONY: lint

################################################################

$(BUILD_DIR_ABS_RPC)/%.o: %.cc
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS_RPC)/%.d: %.cc
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

-include $(DEPENDS)

%.lint: %.cc
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet $< -- $(CPPFLAGS) $(CXXFLAGS)
.PHONY: %.lint

################################################################

$(BUILD_DIR_ABS_RPC)/%: \
$(BUILD_DIR_ABS_RPC)/%_main.o \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $^ $(LDFLAGS)
