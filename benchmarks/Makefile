FUTHARK_BACKEND?=cuda
# Token count shared by lexer and parser benchmarks; payload bytes ≈ 2x.
INPUT_SIZE?=52428800

LANGS=json lisp numfmt
MAKEARGS=--no-print-directory FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

.PHONY: bench build clean reset bench-all-sizes \
        bench-json bench-lisp bench-numfmt \
        bench-lexer bench-lexer-json bench-lexer-lisp bench-lexer-numfmt \
        bench-parser bench-parser-json bench-parser-lisp bench-parser-numfmt \
        bench-cuda bench-futhark \
        build-json build-lisp build-numfmt clean-json clean-lisp clean-numfmt reset-json reset-lisp reset-numfmt

bench: bench-json bench-lisp bench-numfmt

bench-json:
	@$(MAKE) -C json bench $(MAKEARGS)

bench-lisp:
	@$(MAKE) -C lisp bench $(MAKEARGS)

bench-numfmt:
	@$(MAKE) -C numfmt bench $(MAKEARGS)

bench-lexer: bench-lexer-json bench-lexer-lisp bench-lexer-numfmt

bench-lexer-json:
	@$(MAKE) -C json bench-lexer $(MAKEARGS)

bench-lexer-lisp:
	@$(MAKE) -C lisp bench-lexer $(MAKEARGS)

bench-lexer-numfmt:
	@$(MAKE) -C numfmt bench-lexer $(MAKEARGS)

bench-parser: bench-parser-json bench-parser-lisp bench-parser-numfmt

bench-parser-json:
	@$(MAKE) -C json bench-parser $(MAKEARGS)

bench-parser-lisp:
	@$(MAKE) -C lisp bench-parser $(MAKEARGS)

bench-parser-numfmt:
	@$(MAKE) -C numfmt bench-parser $(MAKEARGS)

bench-cuda:
	@printf '\n\033[1m=== JSON ===\033[0m\n'
	@$(MAKE) -C json build-cuda $(MAKEARGS)
	@$(MAKE) -C json bench-lexer-cuda bench-parser-cuda $(MAKEARGS)
	@printf '\n\033[1m=== Lisp ===\033[0m\n'
	@$(MAKE) -C lisp build-cuda $(MAKEARGS)
	@$(MAKE) -C lisp bench-lexer-cuda bench-parser-cuda $(MAKEARGS)
	@printf '\n\033[1m=== numfmt ===\033[0m\n'
	@$(MAKE) -C numfmt build-cuda $(MAKEARGS)
	@$(MAKE) -C numfmt bench-lexer-cuda bench-parser-cuda $(MAKEARGS)
	@printf '\n\033[1m=== Done ===\033[0m\n'

bench-futhark:
	@printf '\n\033[1m=== JSON ===\033[0m\n'
	@$(MAKE) -C json bench-lexer-futhark bench-parser-futhark $(MAKEARGS)
	@printf '\n\033[1m=== Lisp ===\033[0m\n'
	@$(MAKE) -C lisp bench-lexer-futhark bench-parser-futhark $(MAKEARGS)
	@printf '\n\033[1m=== Done ===\033[0m\n'

bench-all-sizes:
	@$(MAKE) bench INPUT_SIZE=10485760 $(MAKEARGS)
	@$(MAKE) bench INPUT_SIZE=104857600 $(MAKEARGS)

build: build-json build-lisp build-numfmt

build-json:
	@$(MAKE) -C json build $(MAKEARGS)

build-lisp:
	@$(MAKE) -C lisp build $(MAKEARGS)

build-numfmt:
	@$(MAKE) -C numfmt build $(MAKEARGS)

reset: reset-json reset-lisp reset-numfmt

reset-json:
	@$(MAKE) -C json reset $(MAKEARGS)

reset-lisp:
	@$(MAKE) -C lisp reset $(MAKEARGS)

reset-numfmt:
	@$(MAKE) -C numfmt reset $(MAKEARGS)

clean: clean-json clean-lisp clean-numfmt

clean-json:
	@$(MAKE) -C json clean $(MAKEARGS)

clean-lisp:
	@$(MAKE) -C lisp clean $(MAKEARGS)

clean-numfmt:
	@$(MAKE) -C numfmt clean $(MAKEARGS)
