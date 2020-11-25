# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

ifeq ($(BUILD_DIR_ABS),)
$(error run "make" at project root directory)
endif

RELPATH      := example/rank
BUILD_DIR_ABS_RANK := $(BUILD_DIR_ABS)/$(RELPATH)

SOURCES      := $(shell find . -type f -name "*.cc" | sort)
LIB_SOURCES  := $(shell find model_zoo -type f -name "*.cc" | sort) \
model_zoo_impl.cc \
trainer_context.cc
LINT_SOURCES := $(SOURCES:.cc=.lint)

LIB_OBJECTS  := $(addprefix $(BUILD_DIR_ABS_RANK)/,$(LIB_SOURCES))
LIB_OBJECTS  := $(LIB_OBJECTS:.cc=.o)

ifeq ($(filter $(MAKECMDGOALS),clean lint),)
DEPENDS      := $(addprefix $(BUILD_DIR_ABS_RANK)/,$(SOURCES))
DEPENDS      := $(DEPENDS:.cc=.d)
else
DEPENDS      :=
endif

LIBRARIES    := \
$(BUILD_DIR_ABS_RANK)/librank.a

BINARIES     := \
$(BUILD_DIR_ABS_RANK)/dist_trainer \
$(BUILD_DIR_ABS_RANK)/model_server_demo \
$(BUILD_DIR_ABS_RANK)/predictor \
$(BUILD_DIR_ABS_RANK)/trainer

ifeq ($(OS_DARWIN),1)
FORCE_LIBS   += -Wl,-all_load $(BUILD_DIR_ABS_RANK)/librank.a
else
FORCE_LIBS   += -Wl,--whole-archive $(BUILD_DIR_ABS_RANK)/librank.a -Wl,--no-whole-archive
endif

################################################################

all: $(LIBRARIES) $(BINARIES)
.PHONY: all

clean:
.PHONY: clean

test:
.PHONY: test

lint: $(LINT_SOURCES)
.PHONY: lint

################################################################

$(BUILD_DIR_ABS_RANK)/%.o: %.cc
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS_RANK)/%.d: %.cc
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

-include $(DEPENDS)

%.lint: %.cc
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet $< -- $(CPPFLAGS) $(CXXFLAGS)
.PHONY: %.lint

################################################################

$(BUILD_DIR_ABS_RANK)/librank.a: $(LIB_OBJECTS)
	@echo Archiving $@
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^

$(BUILD_DIR_ABS_RANK)/dist_trainer: \
$(BUILD_DIR_ABS_RANK)/dist_trainer_main.o \
$(BUILD_DIR_ABS_RANK)/dist_flags.o \
$(BUILD_DIR_ABS_RANK)/dist_server.o \
$(BUILD_DIR_ABS_RANK)/dist_worker.o \
$(BUILD_DIR_ABS_RANK)/librank.a \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS_RANK)/model_server_demo: \
$(BUILD_DIR_ABS_RANK)/model_server_demo_main.o \
$(BUILD_DIR_ABS_RANK)/model_server.o \
$(BUILD_DIR_ABS_RANK)/librank.a \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS_RANK)/predictor: \
$(BUILD_DIR_ABS_RANK)/predictor_main.o \
$(BUILD_DIR_ABS_RANK)/librank.a \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS_RANK)/trainer: \
$(BUILD_DIR_ABS_RANK)/trainer_main.o \
$(BUILD_DIR_ABS_RANK)/librank.a \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)
