# Makefile for alpacc — mirrors the CI steps in .github/workflows/main.yml.
#
# Targets:
#   make build          build and install the alpacc binary to ~/bin
#   make test-unit      cabal unit/property tests
#   make test-futhark   Futhark random-grammar tests + JSON end-to-end (Futhark multicore)
#   make test-c         C random-grammar tests + JSON end-to-end (C backend)
#   make test-cuda      CUDA random-grammar tests + JSON end-to-end (CUDA backend)
#   make test           test-unit + test-futhark + test-c  (no GPU required)

SHELL := bash
INSTALLDIR := $(HOME)/bin

# Arguments forwarded to the random-grammar test scripts (mirror CI values).
RANDOM_TARGET := 50
RANDOM_JOBS   := 10

# CUDA arch (override with: make test-cuda CUDA_ARCH=sm_75)
export CUDA_ARCH ?= native

# Futhark backend used for JSON test (override with: make test-json-futhark FUTHARK_BACKEND=cuda)
FUTHARK_BACKEND ?= multicore

.PHONY: build \
        test-unit \
        test-futhark test-c test-cuda \
        test

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build:
	cabal build all
	cabal install --installdir=$(INSTALLDIR) --overwrite-policy=always

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

test-unit:
	cabal test all

# ---------------------------------------------------------------------------
# Random-grammar differential tests
# ---------------------------------------------------------------------------

# Run one (q, k) config for testc.sh/testcuda.sh and optional mode flag.
# Usage: $(call run_random_c, script, q, k, [--lexer|--parser|])
define run_random_c
	bash $(1) $(2) $(3) $(RANDOM_TARGET) $(RANDOM_JOBS) $(4)
endef

# Run one (q, k) config for testfuthark.sh, forwarding the Futhark backend.
# Usage: $(call run_random_futhark, q, k, [--lexer|--parser|])
define run_random_futhark
	bash tests/testfuthark.sh $(1) $(2) $(RANDOM_TARGET) $(RANDOM_JOBS) $(3) $(FUTHARK_BACKEND)
endef

test-futhark:
	$(call run_random_futhark, 0, 0, --lexer)
	$(call run_random_futhark, 0, 1, --parser)
	$(call run_random_futhark, 1, 1, --parser)
	$(call run_random_futhark, 2, 2, --parser)
	$(call run_random_futhark, 3, 3, --parser)
	$(call run_random_futhark, 0, 1,)
	$(call run_random_futhark, 1, 1,)
	$(call run_random_futhark, 2, 2,)
	$(call run_random_futhark, 3, 3,)
	bash tests/testjson-futhark.sh $(FUTHARK_BACKEND)

test-c:
	$(call run_random_c, tests/testc.sh, 0, 0, --lexer)
	$(call run_random_c, tests/testc.sh, 0, 1, --parser)
	$(call run_random_c, tests/testc.sh, 1, 1, --parser)
	$(call run_random_c, tests/testc.sh, 2, 2, --parser)
	$(call run_random_c, tests/testc.sh, 3, 3, --parser)
	$(call run_random_c, tests/testc.sh, 0, 1,)
	$(call run_random_c, tests/testc.sh, 1, 1,)
	$(call run_random_c, tests/testc.sh, 2, 2,)
	$(call run_random_c, tests/testc.sh, 3, 3,)
	bash tests/testjson-c.sh

test-cuda:
	$(call run_random_c, tests/testcuda.sh, 0, 0, --lexer)
	$(call run_random_c, tests/testcuda.sh, 0, 1, --parser)
	$(call run_random_c, tests/testcuda.sh, 1, 1, --parser)
	$(call run_random_c, tests/testcuda.sh, 2, 2, --parser)
	$(call run_random_c, tests/testcuda.sh, 3, 3, --parser)
	$(call run_random_c, tests/testcuda.sh, 0, 1,)
	$(call run_random_c, tests/testcuda.sh, 1, 1,)
	$(call run_random_c, tests/testcuda.sh, 2, 2,)
	$(call run_random_c, tests/testcuda.sh, 3, 3,)
	bash tests/testjson-cuda.sh $(CUDA_ARCH)

# ---------------------------------------------------------------------------
# Composite targets
# ---------------------------------------------------------------------------

# Mirrors the full CI test job (no GPU required).
test: test-unit test-futhark test-c
