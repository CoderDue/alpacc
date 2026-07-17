H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
D=\x1B[2m
GRAMMAR_NAME=$(notdir $(basename $(GRAMMAR)))
FUTHARK_BACKEND?=cuda
# INPUT_SIZE is a token count shared by lexer and parser benchmarks
# (same dataset, so identical payload bytes ≈ 2x the token count).
INPUT_SIZE?=52428800
GRAMMAR=../../grammars/json.alp
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).json
NVCCFLAGS=-O3 -std=c++17 -arch=native

.PHONY: clean reset build bench bench-lexer bench-parser \
        bench-lexer-cuda bench-lexer-futhark bench-lexer-rust \
        bench-parser-cuda bench-parser-futhark

# --- Datasets ---

$(DATA): $(GRAMMAR)
	@printf "$(D)  Generating parser dataset...$(E)\n"
	@alpacc test generate $(GRAMMAR) --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

$(DATA_RAW): $(DATA)
	@dd if=$(DATA) of=$(DATA_RAW) bs=24 skip=1 2>/dev/null

# --- Lexer implementations ---

lexer.cu: $(GRAMMAR)
	@alpacc cuda $(GRAMMAR) --lexer --index32 -o $@

lexer-cuda: lexer.cu
	@printf "$(D)  Compiling CUDA lexer...$(E)\n"
	@nvcc $(NVCCFLAGS) -o $@ $^

lex.fut: $(GRAMMAR) $(DATA_RAW)
	@alpacc futhark $(GRAMMAR) --lexer --index32 -o $@
	@printf -- "\n-- ==\n-- entry: lex_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	@futhark pkg sync > /dev/null 2>&1

lexer-rust: lexer.rs Cargo.toml Cargo.lock
	@printf "$(D)  Compiling Rust lexer...$(E)\n"
	@cargo build --release 2>/dev/null
	@cp target/release/lexer-rust .

# --- Parser implementations ---

parser.cu: $(GRAMMAR)
	@alpacc cuda $(GRAMMAR) --index32 -o $@

parser-cuda: parser.cu
	@printf "$(D)  Compiling CUDA parser...$(E)\n"
	@nvcc $(NVCCFLAGS) -o $@ $^

parse.fut: $(GRAMMAR) $(DATA_RAW)
	@alpacc futhark $(GRAMMAR) --index32 -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	@futhark pkg sync > /dev/null 2>&1

# --- Benchmarks ---

build: lexer-cuda parser-cuda lex.fut parse.fut $(DATA_RAW)
	@printf "$(H)Build complete [$(GRAMMAR_NAME)].$(E)\n"

bench-lexer-cuda: lexer-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) lexer [$(GRAMMAR_NAME)]:$(E)\n"
	@./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-lexer-futhark: lex.fut $(DATA_RAW)
	@futhark pkg sync > /dev/null 2>&1
	@printf "$(H)Benchmarking Futhark lexer [$(GRAMMAR_NAME)]:$(E)\n"
	@futhark bench --backend $(FUTHARK_BACKEND) lex.fut

bench-lexer-rust: lexer-rust $(DATA_RAW)
	@printf "$(H)Benchmarking Rust lexer [$(GRAMMAR_NAME)]:$(E)\n"
	@./lexer-rust $(DATA_RAW)

bench-lexer:
	@printf "$(H)JSON Lexer Benchmark [$(GRAMMAR_NAME)]:$(E)\n"
	@$(MAKE) --no-print-directory bench-lexer-cuda bench-lexer-futhark

bench-parser-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) parser [$(GRAMMAR_NAME)]:$(E)\n"
	@./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-parser-futhark: parse.fut $(DATA_RAW)
	@futhark pkg sync > /dev/null 2>&1
	@printf "$(H)Benchmarking Futhark parser [$(GRAMMAR_NAME)]:$(E)\n"
	@futhark bench --backend $(FUTHARK_BACKEND) parse.fut

bench-parser:
	@printf "$(H)JSON Parser Benchmark [$(GRAMMAR_NAME)]:$(E)\n"
	@$(MAKE) --no-print-directory bench-parser-cuda bench-parser-futhark

bench: bench-lexer bench-parser

reset:
	@rm -rf target lexer-rust lexer-cuda lexer.cu lex.fut lex lex.c \
	       parser-cuda parser.cu parse.fut parse parse.c

clean: reset
	@rm -rf data-*.inputs data-raw-*.json lib
