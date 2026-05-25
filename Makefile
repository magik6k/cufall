# SPDX-License-Identifier: MIT

CXX ?= g++
CUDA_HOME ?= /opt/cuda/targets/x86_64-linux
PREFIX ?= /usr/local

CPPFLAGS += -I$(CUDA_HOME)/include
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -pedantic
LDFLAGS += -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib/stubs -Wl,-rpath,$(CUDA_HOME)/lib
LDLIBS += -lcupti -lnvperf_host -lcuda -ldl -lpthread

.PHONY: all clean install uninstall

all: cufall

cufall: src/cufall.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f cufall cupti-braille

install: cufall
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 cufall $(DESTDIR)$(PREFIX)/bin/cufall

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/cufall $(DESTDIR)$(PREFIX)/bin/cupti-braille
