# griz builds with Go's portable SIMD package (Go 1.27, still an experiment),
# which roughly halves embedding time. Without GOEXPERIMENT=simd, griz builds
# with plain Go loops instead; results are the same to float rounding.
GOEXPERIMENT := simd
export GOEXPERIMENT

.PHONY: build install test test-plain race

build:
	go build -o griz .

install:
	go install .

# Tests run with SIMD, then without, so both versions stay correct.
test:
	go vet ./...
	go test ./...
	GOEXPERIMENT= go test ./...

race:
	go test -race ./...
