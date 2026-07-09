FUTHARK_BACKEND?=cuda
INPUT_SIZE?=104857600

.PHONY: bench build clean \
        bench-json bench-lisp \
        build-json build-lisp \
        clean-json clean-lisp

bench: bench-json bench-lisp

build: build-json build-lisp

clean: clean-json clean-lisp

bench-json:
	$(MAKE) -C lexer/json bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

bench-lisp:
	$(MAKE) -C lexer/lisp bench FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

build-json:
	$(MAKE) -C lexer/json build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

build-lisp:
	$(MAKE) -C lexer/lisp build FUTHARK_BACKEND=$(FUTHARK_BACKEND) INPUT_SIZE=$(INPUT_SIZE)

clean-json:
	$(MAKE) -C lexer/json clean

clean-lisp:
	$(MAKE) -C lexer/lisp clean
