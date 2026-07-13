CC?=cc
CFLAGS=-O3
H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=104857600
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).json

.PHONY: clean bench build bench-rust bench-futhark bench-cuda

lexer.cu: json.alp
	alpacc cuda json.alp --lexer -o $@

lexer-cuda: lexer.cu
	nvcc -O3 -std=c++17 -arch=native -o $@ $^

$(DATA): json.alp
	alpacc test generate json.alp --lexer --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=24 skip=1

lexer-rust: lexer.rs Cargo.toml Cargo.lock
	cargo build --release
	@cp target/release/lexer-rust .

json.fut: json.alp $(DATA_RAW)
	alpacc futhark json.alp --lexer
	@printf -- "\n-- ==\n-- entry: lex_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> json.fut
	futhark pkg sync

build: lexer-cuda json.fut $(DATA_RAW)
	@:

bench-rust: lexer-rust $(DATA_RAW)
	@printf "$(H)Benchmarking Rust implementation:$(E)\n"
	@./lexer-rust $(DATA_RAW)

bench-cuda: lexer-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-futhark: json.fut $(DATA_RAW)
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

bench: lexer-cuda json.fut $(DATA)
	@printf "$(H)JSON Lexer Benchmark:$(E)\n"

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	./lexer-cuda --benchmark 10 --warmup 3 < $(DATA)

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

clean:
	rm -rf data-*.inputs data-raw-*.json target lexer-rust json json.c json.fut lexer-cuda lexer.cu
