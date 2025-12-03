DBG ?= -g

# Auto-detect Intel compiler; fall back to g++ if not available
ifeq ($(shell which icpc 2>/dev/null),)
  CXX ?= g++
else
  CXX ?= icpc
endif

MPICXX ?= mpic++
COMMON = common
CXXFLAGS  = -O3 -std=c++14 -I. -I$(COMMON) $(DBG)
CXXFLAGS += -Wfatal-errors
CXXFLAGS += -DWITH_MPI

COMMON_OBJS = common/timer.o common/dummy.o
EXEC = thomas mpi_thomas

thomas: thomas.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o thomas thomas.cpp $(COMMON_OBJS) $(LDFLAGS)

mpi_thomas: mpi_thomas.cpp $(COMMON_OBJS)
	$(MPICXX) $(CXXFLAGS) -o mpi_thomas mpi_thomas.cpp $(COMMON_OBJS) $(LDFLAGS)

common/timer.o: common/timer.cpp
	$(CXX) $(CXXFLAGS) -c common/timer.cpp -o common/timer.o

common/dummy.o: common/dummy.cpp
	$(CXX) $(CXXFLAGS) -c common/dummy.cpp -o common/dummy.o

all: thomas mpi_thomas

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt common/*.o common/*.d

.PHONY: all clean
