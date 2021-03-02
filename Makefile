# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

include Makefile.rule

SOURCES      := $(shell find src -type f -name "*.cc" | sort)
TEST_SOURCES := $(shell find src -type f -name "*_test.cc" | sort)
BIN_SOURCES  := $(shell find src -type f -name "*_main.cc" | sort)
LIB_SOURCES  := $(filter-out $(TEST_SOURCES) $(BIN_SOURCES),$(SOURCES))
LINT_SOURCES := $(SOURCES:.cc=.lint)

TEST_OBJECTS := $(addprefix $(BUILD_DIR_ABS)/,$(TEST_SOURCES))
TEST_OBJECTS := $(TEST_OBJECTS:.cc=.o)
LIB_OBJECTS  := $(addprefix $(BUILD_DIR_ABS)/,$(LIB_SOURCES))
LIB_OBJECTS  := $(LIB_OBJECTS:.cc=.o)

ifeq ($(filter $(MAKECMDGOALS),clean lint),)
DEPENDS      := $(addprefix $(BUILD_DIR_ABS)/,$(SOURCES))
DEPENDS      := $(DEPENDS:.cc=.d)
else
DEPENDS      :=
endif

LIBRARIES    := \
$(BUILD_DIR_ABS)/libdeepx_core.a

BINARIES     := \
$(BUILD_DIR_ABS)/eval_auc \
$(BUILD_DIR_ABS)/feature_kv_demo \
$(BUILD_DIR_ABS)/fs_tool \
$(BUILD_DIR_ABS)/merge_model_shard \
$(BUILD_DIR_ABS)/unit_test

SUBDIRS      := example

ifeq ($(OS_DARWIN),1)
FORCE_LIBS   := -Wl,-all_load $(BUILD_DIR_ABS)/libdeepx_core.a
else
FORCE_LIBS   := -Wl,--whole-archive $(BUILD_DIR_ABS)/libdeepx_core.a -Wl,--no-whole-archive
endif
export FORCE_LIBS

################################################################

all: _all subdir_all
	@echo "******************************************"
	@echo "Build succsessfully at $(BUILD_DIR_ABS)"
	@echo "CC:          " $(CC)
	@echo "CXX:         " $(CXX)
	@echo "AR:          " $(AR)
	@echo "CPPFLAGS:    " $(CPPFLAGS)
	@echo "CFLAGS:      " $(CFLAGS)
	@echo "CXXFLAGS:    " $(CXXFLAGS)
	@echo "LDFLAGS:     " $(LDFLAGS)
	@echo "******************************************"
	@echo "DEBUG:       " $(DEBUG)
	@echo "SIMD:        " $(SIMD)
	@echo "SAGE2:       " $(SAGE2)
	@echo "SAGE2_SGEMM: " $(SAGE2_SGEMM)
	@echo "SAGE2_SGEMM_JIT:" $(SAGE2_SGEMM_JIT)
	@echo "******************************************"
.PHONY: all

include Makefile.3rd
_all: $(3RD_LIBS) $(LIBRARIES) $(BINARIES)
.PHONY: _all

subdir_all: _all
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir all; \
	done
.PHONY: subdir_all

lib: $(3RD_LIBS) $(LIBRARIES)
	@echo "******************************************"
	@echo "Build succsessfully at $(BUILD_DIR_ABS)"
	@echo "CC:          " $(CC)
	@echo "CXX:         " $(CXX)
	@echo "AR:          " $(AR)
	@echo "CPPFLAGS:    " $(CPPFLAGS)
	@echo "CFLAGS:      " $(CFLAGS)
	@echo "CXXFLAGS:    " $(CXXFLAGS)
	@echo "LDFLAGS:     " $(LDFLAGS)
	@echo "******************************************"
	@echo "DEBUG:       " $(DEBUG)
	@echo "SIMD:        " $(SIMD)
	@echo "SAGE2:       " $(SAGE2)
	@echo "SAGE2_SGEMM: " $(SAGE2_SGEMM)
	@echo "SAGE2_SGEMM_JIT:" $(SAGE2_SGEMM_JIT)
	@echo "******************************************"
.PHONY: lib

clean: _clean $(SUBDIRS)
.PHONY: clean

_clean:
	@echo Cleaning $(BUILD_DIR_ABS)
	@rm -rf $(BUILD_DIR_ABS)
.PHONY: _clean

test: _test $(SUBDIRS)
.PHONY: test

_test: $(BUILD_DIR_ABS)/unit_test
	@rm -rf $(BUILD_DIR_ABS)/testdata
	@cp -R -P src/testdata $(BUILD_DIR_ABS)/testdata
	@cd $(BUILD_DIR_ABS) && ./unit_test
	@if test "x$(NO_VALGRIND)" != x1; then \
		cd $(BUILD_DIR_ABS) && which valgrind > /dev/null 2>&1 && valgrind ./unit_test || true; \
	fi
.PHONY: _test

lint: $(LINT_SOURCES) $(SUBDIRS)
.PHONY: lint

$(SUBDIRS):
	@$(MAKE) -C $@ $(MAKECMDGOALS)
.PHONY: $(SUBDIRS)

install: lib
	@bash -e install.sh $(PREFIX) $(BUILD_DIR_ABS)
.PHONY: install

build_dir_abs:
	@echo $(BUILD_DIR_ABS)
.PHONY: build_dir_abs

################################################################

$(BUILD_DIR_ABS)/src/%.o: src/%.cc
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS)/src/%.d: src/%.cc
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

-include $(DEPENDS)

src/%.lint: src/%.cc
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet $< -- $(CPPFLAGS) $(CXXFLAGS)
.PHONY: src/%.lint

################################################################

$(BUILD_DIR_ABS)/libdeepx_core.a: $(LIB_OBJECTS)
	@echo Archiving $@
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^

$(BUILD_DIR_ABS)/eval_auc: \
$(BUILD_DIR_ABS)/src/tools/eval_auc_main.o \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/feature_kv_demo: \
$(BUILD_DIR_ABS)/src/tools/feature_kv_demo_main.o \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/fs_tool: \
$(BUILD_DIR_ABS)/src/tools/fs_tool_main.o \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/merge_model_shard: \
$(BUILD_DIR_ABS)/src/tools/merge_model_shard_main.o \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/unit_test: \
$(TEST_OBJECTS) \
$(BUILD_DIR_ABS)/libdeepx_core.a \
$(BUILD_DIR_ABS)/libdeepx_gflags.a \
$(BUILD_DIR_ABS)/libdeepx_gtest.a \
$(BUILD_DIR_ABS)/libdeepx_lz4.a \
$(BUILD_DIR_ABS)/libdeepx_z.a
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)
