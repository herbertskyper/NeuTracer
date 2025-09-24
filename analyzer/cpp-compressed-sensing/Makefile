# CXX = g++
# CXXFLAGS = -std=c++17 -O2 \
#     -I/home/light7an/include/eigen-3.4.0 \
#     -I/home/light7an/include/EigenRand \
#     -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/osqp

# LDFLAGS = -L$(CONDA_PREFIX)/lib -losqp -lOsqpEigen -lfftw3 -lpthread

# SRC = $(wildcard src/*.cpp)
# OBJ = $(patsubst src/%.cpp, build/%.o, $(SRC))
# TARGET = build/detector
BUILD_DIR = build

all: 
	cmake -B $(BUILD_DIR) -DCMAKE_VERBOSE_MAKEFILE=ON 
	cmake --build $(BUILD_DIR)

# build:
# 	mkdir -p build

# build/%.o: src/%.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# $(TARGET): $(OBJ)
# 	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf build