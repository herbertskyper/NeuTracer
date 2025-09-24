CXX = g++
CXXFLAGS = -std=c++17 -O2 \
    -I/home/light7an/include/eigen-3.4.0 \
    -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/osqp

LDFLAGS = -L$(CONDA_PREFIX)/lib -losqp -lOsqpEigen -lfftw3 -lpthread

SRC = main.cpp cvxpy.cpp
OBJ = $(addprefix build/,$(SRC:.cpp=.o))
TARGET = build/test_cvxpy

all: build $(TARGET)

build:
	mkdir -p build

build/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf build