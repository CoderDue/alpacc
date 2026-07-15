H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=10485760
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).in
.PHONY: clean reset bench build bench-futhark bench-cuda

parser.cu: sexp.alp
	alpacc cuda sexp.alp --index32 -o $@

NVCCFLAGS=-O3 -std=c++17 -arch=native

parser-cuda: parser.cu
	nvcc $(NVCCFLAGS) -o $@ $^

$(DATA): sexp.alp
	alpacc test generate sexp.alp --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=24 skip=1

sexp.fut: sexp.alp $(DATA_RAW)
	alpacc futhark sexp.alp --index32 -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

build: parser-cuda sexp.fut $(DATA_RAW)
	@:

bench-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-futhark: sexp.fut $(DATA_RAW)
	futhark pkg sync
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

bench: parser-cuda sexp.fut $(DATA)
	@printf "$(H)Lisp Parser Benchmark:$(E)\n"

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

reset:
	rm -rf parser-cuda parser.cu sexp sexp.c sexp.fut

clean:
	rm -rf data-*.inputs data-raw-*.in parser-cuda parser.cu sexp sexp.c sexp.fut
