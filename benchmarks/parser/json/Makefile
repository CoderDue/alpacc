H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=10485760
DATA=data-$(INPUT_SIZE).inputs
DATA_RAW=data-raw-$(INPUT_SIZE).json
.PHONY: clean bench build bench-futhark bench-cuda

parser.cu: json.alp
	alpacc cuda json.alp --index32 -o $@

NVCCFLAGS=-O3 -std=c++17 -arch=native

parser-cuda: parser.cu
	nvcc $(NVCCFLAGS) -o $@ $^

$(DATA): json.alp
	alpacc test generate json.alp --single-long --no-outputs --length $(INPUT_SIZE) -o data-$(INPUT_SIZE)

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=16 skip=1

json.fut: json.alp $(DATA_RAW)
	alpacc futhark json.alp --index32 -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

build: parser-cuda json.fut $(DATA_RAW)
	@:

bench-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

bench-futhark: json.fut $(DATA_RAW)
	futhark pkg sync
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

bench: parser-cuda json.fut $(DATA)
	@printf "$(H)JSON Parser Benchmark:$(E)\n"

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	./parser-cuda --benchmark 10 --warmup 3 < $(DATA)

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

clean:
	rm -rf data-*.inputs data-raw-*.json parser-cuda parser.cu json json.c json.fut
