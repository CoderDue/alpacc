CC?=cc
CFLAGS=-O3 -Werror -Wall -Wextra -Wunused
H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
# INPUT_SIZE is the number of tokens in the generated input (combined-mode
# single-long generation); the byte size of the payload is roughly 2x that.
INPUT_SIZE?=10485760
GRAMMAR=../../grammars/sexp.alp
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).in
NVCCFLAGS=-O3 -std=c++17 -arch=native

.PHONY: clean reset build bench bench-lexer bench-parser \
        bench-lexer-cuda bench-lexer-futhark bench-lexer-rust bench-lexer-c \
        bench-parser-cuda bench-parser-futhark

# --- Shared dataset (grammatical sexp, used by lexer and parser benchmarks) ---

$(DATA): $(GRAMMAR)
	alpacc test generate $(GRAMMAR) --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=24 skip=1

# --- Lexer implementations ---

lexer-c: lexer.c
	$(CC) -o $@ $^ $(CFLAGS)

lexer.cu: $(GRAMMAR)
	alpacc cuda $(GRAMMAR) --lexer --index32 -o $@

lexer-cuda: lexer.cu
	nvcc $(NVCCFLAGS) -o $@ $^

lex.fut: $(GRAMMAR) $(DATA_RAW)
	alpacc futhark $(GRAMMAR) --lexer --index32 -o $@
	@printf -- "\n-- ==\n-- entry: lex_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

lexer-rust: lexer.rs Cargo.toml Cargo.lock
	cargo build --release
	@cp target/release/lexer-rust .

# --- Parser implementations ---

parser.cu: $(GRAMMAR)
	alpacc cuda $(GRAMMAR) --index32 -o $@

parser-cuda: parser.cu
	nvcc $(NVCCFLAGS) -o $@ $^

parse.fut: $(GRAMMAR) $(DATA_RAW)
	alpacc futhark $(GRAMMAR) --index32 -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

# --- Benchmarks ---

build: lexer-c lexer-cuda parser-cuda lex.fut parse.fut $(DATA_RAW)
	@:

bench-lexer-c: lexer-c $(DATA_RAW)
	@printf "$(H)Benchmarking C lexer:$(E)\n"
	@printf "Note: The result is not saved in memory.\n"
	@./lexer-c $(DATA_RAW)

bench-lexer-cuda: lexer-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) lexer:$(E)\n"
	./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-lexer-futhark: lex.fut $(DATA_RAW)
	futhark pkg sync
	@printf "$(H)Benchmarking Futhark lexer:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) lex.fut

bench-lexer-rust: lexer-rust $(DATA_RAW)
	@printf "$(H)Benchmarking Rust lexer:$(E)\n"
	@./lexer-rust $(DATA_RAW)

bench-lexer:
	@printf "$(H)Lisp Lexer Benchmark:$(E)\n"
	$(MAKE) bench-lexer-c bench-lexer-cuda bench-lexer-futhark

bench-parser-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) parser:$(E)\n"
	./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-parser-futhark: parse.fut $(DATA_RAW)
	futhark pkg sync
	@printf "$(H)Benchmarking Futhark parser:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) parse.fut

bench-parser:
	@printf "$(H)Lisp Parser Benchmark:$(E)\n"
	$(MAKE) bench-parser-cuda bench-parser-futhark

bench: bench-lexer bench-parser

reset:
	rm -rf target lexer-rust lexer-c lexer-cuda lexer.cu lex.fut lex lex.c \
	       parser-cuda parser.cu parse.fut parse parse.c

clean: reset
	rm -rf data-*.inputs data-raw-*.in lib
