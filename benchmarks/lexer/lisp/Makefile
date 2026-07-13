CC?=cc
CFLAGS=-O3 -Werror -Wall -Wextra -Wunused
H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=104857600
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).in
.PHONY: clean bench build bench-rust bench-c bench-futhark bench-cuda

lexer-c: lexer.c
	$(CC) -o $@ $^ $(CFLAGS)

lexer.cu: sexp.alp
	alpacc cuda sexp.alp --lexer -o $@

lexer-cuda: lexer.cu
	nvcc -O3 -std=c++17 -arch=native -o $@ $^

$(DATA): sexp.alp
	alpacc test generate sexp.alp --lexer --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

lexer-rust: lexer.rs Cargo.toml Cargo.lock
	cargo build --release
	@cp target/release/lexer-rust .

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=24 skip=1

sexp.fut: sexp.alp $(DATA_RAW)
	alpacc futhark sexp.alp --lexer
	@printf -- "\n-- ==\n-- entry: lex_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> sexp.fut
	futhark pkg sync

build: lexer-c lexer-cuda sexp.fut $(DATA_RAW)
	@:

bench-rust: lexer-rust $(DATA_RAW)
	@printf "$(H)Benchmarking Rust implementation:$(E)\n"
	@./lexer-rust $(DATA_RAW)

bench-c: lexer-c $(DATA_RAW)
	@printf "$(H)Benchmarking C implementation:$(E)\n"
	@printf "Note: The result is not saved in memory.\n"
	@./lexer-c $(DATA_RAW)

bench-cuda: lexer-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-futhark: sexp.fut $(DATA_RAW)
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

bench: lexer-c lexer-cuda sexp.fut $(DATA)
	@printf "$(H)Lisp Lexer Benchmark:$(E)\n"

	@printf "$(H)C implementation:$(E)\n"
	@printf "Note: The result is not saved in memory.\n"
	@./lexer-c $(DATA_RAW)

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

clean:
	rm -rf data-*.inputs data-raw-*.in target lexer-rust lexer-c sexp sexp.c sexp.fut lexer-cuda lexer.cu
