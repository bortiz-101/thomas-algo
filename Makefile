DBG ?= -g

# Auto-detect Intel compiler; fall back to g++ if not available
ifeq ($(shell which icpc 2>/dev/null),)
  CXX ?= g++
else
  CXX ?= icpc
endif

COMMON = common
CXXFLAGS  = -O3 -std=c++14 -I. -I$(COMMON) $(DBG)
CXXFLAGS += -Wfatal-errors

OMPFLAGS = -fopenmp
COMMON_OBJS = common/timer.o common/dummy.o
EXEC = thomas omp_thomas

thomas: thomas.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o thomas thomas.cpp $(COMMON_OBJS) $(LDFLAGS)

omp_thomas: omp_thomas.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o omp_thomas omp_thomas.cpp $(COMMON_OBJS) $(LDFLAGS)

common/timer.o: common/timer.cpp
	$(CXX) $(CXXFLAGS) -c common/timer.cpp -o common/timer.o

common/dummy.o: common/dummy.cpp
	$(CXX) $(CXXFLAGS) -c common/dummy.cpp -o common/dummy.o

all: thomas omp_thomas

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt common/*.o common/*.d

.PHONY: all clean

