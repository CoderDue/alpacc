CC?=cc
CFLAGS=-O3
H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=104857600

.PHONY: clean bench build bench-rust bench-futhark bench-cuda

lexer.cu: json.alp
	alpacc cuda json.alp --lexer -o $@

lexer-cuda: lexer.cu
	nvcc -O3 -std=c++17 -arch=native -o $@ $^

data.json: json.alp
	alpacc generate-bytes json.alp --length $(INPUT_SIZE) > $@

lexer-rust: lexer.rs Cargo.toml Cargo.lock
	cargo build --release
	@cp target/release/lexer-rust .

json.fut: json.alp data.json
	alpacc futhark json.alp --lexer
	@printf -- "\n-- ==\n-- entry: lex\n-- script input { \$$loadbytes \"data.json\" }\n-- output @ data.out\n" >> json.fut
	futhark pkg sync

build: lexer-rust lexer-cuda json.fut
	@:

data.out: lexer-rust data.json
	@printf "Creating output dataset...\n"
	@./lexer-rust data.json > /dev/null
	@printf "Done creating output dataset.\n"

bench-rust: lexer-rust data.json
	@printf "$(H)Benchmarking Rust implementation:$(E)\n"
	@./lexer-rust data.json

bench-cuda: lexer-cuda data.json
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	-@./lexer-cuda --timeit < data.json > /dev/null

bench-futhark: json.fut data.json data.out
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

bench: lexer-rust lexer-cuda json.fut data.json
	@printf "$(H)JSON Lexer Benchmark:$(E)\n"
	@printf "$(H)Rust implementation:$(E)\n"
	@./lexer-rust data.json

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	-@./lexer-cuda --timeit < data.json > /dev/null

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

clean:
	rm -rf data.json target lexer-rust data.out json json.c json.fut lexer-cuda lexer.cu
