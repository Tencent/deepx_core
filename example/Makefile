# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

ifeq ($(BUILD_DIR_ABS),)
$(error run "make" at project root directory)
endif

SUBDIRS      := $(shell find . -maxdepth 1 -type d -not -name ".")

all: $(SUBDIRS)
.PHONY: all

clean: $(SUBDIRS)
.PHONY: clean

test: $(SUBDIRS)
.PHONY: test

lint: $(SUBDIRS)
.PHONY: lint

$(SUBDIRS):
	@$(MAKE) -C $@ $(MAKECMDGOALS)
.PHONY: $(SUBDIRS)
