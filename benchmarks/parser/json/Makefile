H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=10485760
DATA=data-$(INPUT_SIZE).json
DATA_RAW=data-raw-$(INPUT_SIZE).json
.PHONY: clean bench build bench-futhark bench-cuda

parser.cu: json.alp
	alpacc cuda json.alp -o $@

parser-cuda: parser.cu
	nvcc -O3 -std=c++17 -arch=native -o $@ $^

$(DATA): json.alp
	alpacc test generate json.alp --single-long --no-outputs --length $(INPUT_SIZE) -o $(basename $(DATA))

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=1 skip=16

json.fut: json.alp $(DATA_RAW)
	alpacc futhark json.alp -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

build: parser-cuda json.fut $(DATA_RAW)
	@:

bench-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	-@./parser-cuda --timeit < $(DATA) > /dev/null

bench-futhark: json.fut $(DATA_RAW)
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

bench: parser-cuda json.fut $(DATA)
	@printf "$(H)JSON Parser Benchmark:$(E)\n"

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	-@./parser-cuda --timeit < $(DATA) > /dev/null

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) json.fut

clean:
	rm -rf data-*.json data-raw-*.json parser-cuda parser.cu json json.c json.fut
