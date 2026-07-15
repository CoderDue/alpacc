FUTHARK_BACKEND?=cuda
INPUT_SIZE?=10485760

.PHONY: bench build clean reset \
        bench-lexer bench-lexer-json bench-lexer-lisp \
        bench-parser bench-parser-json bench-parser-lisp \
        bench-cuda bench-cuda-json bench-cuda-lisp \
        bench-cuda-parser bench-cuda-parser-json bench-cuda-parser-lisp \
        bench-futhark bench-futhark-json bench-futhark-lisp \
        bench-futhark-parser bench-futhark-parser-json bench-futhark-parser-lisp \
        bench-all-sizes \
        build-json build-lisp build-parser-json build-parser-lisp \
        clean-json clean-lisp clean-parser-json clean-parser-lisp \
        reset-json reset-lisp reset-parser-json reset-parser-lisp

bench: bench-lexer bench-parser

bench-lexer: bench-lexer-json bench-lexer-lisp

bench-lexer-json:
	$(MAKE) -C lexer/json bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-lexer-lisp:
	$(MAKE) -C lexer/lisp bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-parser: bench-parser-json bench-parser-lisp

bench-parser-json:
	$(MAKE) -C parser/json bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-parser-lisp:
	$(MAKE) -C parser/lisp bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-cuda: bench-cuda-json bench-cuda-lisp bench-cuda-parser

bench-cuda-json:
	$(MAKE) -C lexer/json bench-cuda INPUT_SIZE=$(INPUT_SIZE)

bench-cuda-lisp:
	$(MAKE) -C lexer/lisp bench-cuda INPUT_SIZE=$(INPUT_SIZE)

bench-cuda-parser: bench-cuda-parser-json bench-cuda-parser-lisp

bench-cuda-parser-json:
	$(MAKE) -C parser/json bench-cuda INPUT_SIZE=$(INPUT_SIZE)

bench-cuda-parser-lisp:
	$(MAKE) -C parser/lisp bench-cuda INPUT_SIZE=$(INPUT_SIZE)

bench-futhark: bench-futhark-json bench-futhark-lisp bench-futhark-parser

bench-futhark-json:
	$(MAKE) -C lexer/json bench-futhark FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-futhark-lisp:
	$(MAKE) -C lexer/lisp bench-futhark FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-futhark-parser: bench-futhark-parser-json bench-futhark-parser-lisp

bench-futhark-parser-json:
	$(MAKE) -C parser/json bench-futhark FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-futhark-parser-lisp:
	$(MAKE) -C parser/lisp bench-futhark FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-all-sizes:
	$(MAKE) bench INPUT_SIZE=10485760
	$(MAKE) bench INPUT_SIZE=104857600

build: build-json build-lisp build-parser-json build-parser-lisp

build-json:
	$(MAKE) -C lexer/json build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

build-lisp:
	$(MAKE) -C lexer/lisp build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

build-parser-json:
	$(MAKE) -C parser/json build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

build-parser-lisp:
	$(MAKE) -C parser/lisp build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

reset: reset-json reset-lisp reset-parser-json reset-parser-lisp

reset-json:
	$(MAKE) -C lexer/json reset

reset-lisp:
	$(MAKE) -C lexer/lisp reset

reset-parser-json:
	$(MAKE) -C parser/json reset

reset-parser-lisp:
	$(MAKE) -C parser/lisp reset

clean: clean-json clean-lisp clean-parser-json clean-parser-lisp

clean-json:
	$(MAKE) -C lexer/json clean

clean-lisp:
	$(MAKE) -C lexer/lisp clean

clean-parser-json:
	$(MAKE) -C parser/json clean

clean-parser-lisp:
	$(MAKE) -C parser/lisp clean
