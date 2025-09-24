BUILD_DIR = build

all:
	mkdir -p log
	mkdir -p output
	mkdir -p $(BUILD_DIR)
	python -m grpc_tools.protoc -I./protos --python_out=./protos/python --grpc_python_out=./protos/python ./protos/tracer_service.proto
	protoc --python_out=./analyzer/visualizer \
	-I ./perfetto \
	./perfetto/protos/perfetto/trace/perfetto_trace.proto
	cmake -B $(BUILD_DIR) -DCMAKE_VERBOSE_MAKEFILE=ON 
	cmake --build $(BUILD_DIR)

clean:
	rm -rf build

.PHONY: clean all