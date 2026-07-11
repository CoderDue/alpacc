# Makefile for alpacc — mirrors the CI steps in .github/workflows/main.yml.
#
# Targets:
#   make build          build and install the alpacc binary to ~/bin
#   make test-unit      cabal unit/property tests
#   make test-futhark   Futhark random-grammar tests + long-input end-to-end (Futhark multicore)
#   make test-c         C random-grammar tests + long-input end-to-end (C backend)
#   make test-cuda      CUDA random-grammar tests + long-input end-to-end (CUDA backend)
#   make test           test-unit + test-futhark + test-c  (no GPU required)

SHELL := bash

# Arguments forwarded to the random-grammar test scripts (mirror CI values).
RANDOM_TARGET := 50
RANDOM_JOBS   := 10

# CUDA arch (override with: make test-cuda CUDA_ARCH=sm_75)
export CUDA_ARCH ?= native

# Futhark backend used for long-input tests (override with: make test-futhark FUTHARK_BACKEND=cuda)
FUTHARK_BACKEND ?= multicore

# Grammars exercised by the long-input end-to-end tests.
LONG_INPUT_GRAMMARS := grammars/json.alp grammars/arithmetic.alp grammars/sexp.alp

.PHONY: build \
        test-unit \
        test-futhark test-c test-cuda \
        test

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build:
	cabal build all

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

# Run test-long-input.sh for every grammar in LONG_INPUT_GRAMMARS.
# Usage: $(call run_long_input, backend)
define run_long_input
	$(foreach g,$(LONG_INPUT_GRAMMARS),bash tests/test-long-input.sh $(g) $(1) &&) true
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
	$(call run_long_input, $(FUTHARK_BACKEND))

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
	$(call run_long_input, c)

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
	$(call run_long_input, cuda)

# ---------------------------------------------------------------------------
# Composite targets
# ---------------------------------------------------------------------------

# Mirrors the full CI test job (no GPU required).
test: test-unit test-futhark test-c
