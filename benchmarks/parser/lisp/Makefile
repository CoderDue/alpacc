H=\x1B[32m\033[1m
E=\033[0m\x1B[0m
FUTHARK_BACKEND?=cuda
INPUT_SIZE?=10485760
DATA=data-$(INPUT_SIZE).in
DATA_RAW=data-raw-$(INPUT_SIZE).in
.PHONY: clean bench build bench-futhark bench-cuda

parser.cu: sexp.alp
	alpacc cuda sexp.alp -o $@

parser-cuda: parser.cu
	nvcc -O3 -std=c++17 -arch=native -o $@ $^

$(DATA): sexp.alp
	alpacc test generate sexp.alp --single-long --no-outputs --length $(INPUT_SIZE) -o $(basename $(DATA))

$(DATA_RAW): $(DATA)
	dd if=$(DATA) of=$(DATA_RAW) bs=16 skip=1

sexp.fut: sexp.alp $(DATA_RAW)
	alpacc futhark sexp.alp -o $@
	@printf -- "\n-- ==\n-- entry: parse_int\n-- script input { \$$loadbytes \"$(DATA_RAW)\" }\n" >> $@
	futhark pkg sync

build: parser-cuda sexp.fut $(DATA_RAW)
	@:

bench-cuda: parser-cuda $(DATA)
	@printf "$(H)Benchmarking CUDA (alpacc) implementation:$(E)\n"
	-@./parser-cuda --timeit < $(DATA) > /dev/null

bench-futhark: sexp.fut $(DATA_RAW)
	@printf "$(H)Benchmarking Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

bench: parser-cuda sexp.fut $(DATA)
	@printf "$(H)Lisp Parser Benchmark:$(E)\n"

	@printf "$(H)CUDA (alpacc) implementation:$(E)\n"
	-@./parser-cuda --timeit < $(DATA) > /dev/null

	@printf "$(H)Futhark implementation:$(E)\n"
	futhark bench --backend $(FUTHARK_BACKEND) sexp.fut

clean:
	rm -rf data-*.in data-raw-*.in parser-cuda parser.cu sexp sexp.c sexp.fut
