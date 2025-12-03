# Thomas Algorithm - Tridiagonal Matrix Solver

A C++ implementation of the Thomas algorithm (also known as TDMA - Tridiagonal Matrix Algorithm) for efficiently solving tridiagonal linear systems.

## Overview

The Thomas algorithm solves systems of linear equations **Ax = d** where **A** is a tridiagonal matrix in O(n) time, making it much more efficient than general Gaussian elimination for this special matrix structure.

## Building

### Sequential Version

```bash
g++ -std=c++11 -o thomas thomas.cpp -lm
```

### MPI Version

```bash
mpic++ -std=c++11 -o mpi_thomas mpi_thomas.cpp -lm
```

## Usage

### Sequential Version

```bash
./thomas -n 100
./thomas -n 5 --diag-a a.txt --diag-b b.txt --diag-c c.txt --rhs d.txt
```

### MPI Version

```bash
mpirun -np 4 ./mpi_thomas -n 1000
mpirun -np 4 ./mpi_thomas -n 1000 --diag-a a.txt --diag-b b.txt --diag-c c.txt --rhs d.txt
```

**HPC Cluster Usage:**

```bash
mpirun -np 16 -hostfile hosts ./mpi_thomas -n 10000 --diag-a a.txt --diag-b b.txt --diag-c c.txt --rhs d.txt
```

The `hosts` file should list compute nodes, one per line (e.g., node4, node5, node6, node7).

### Options

- `-h, --help` - Display help message
- `-n, --npoints #` - Number of equations/size of system (default: 100)
- `--diag-a file` - File with subdiagonal coefficients
- `--diag-b file` - File with main diagonal coefficients  
- `--diag-c file` - File with superdiagonal coefficients
- `--rhs file` - File with right-hand side vector

## Matrix Format

The tridiagonal matrix is represented using three diagonal vectors:

- **a[]** = subdiagonal, indexed [1, ..., n-1] (a[0] unused)
- **b[]** = main diagonal, indexed [0, ..., n-1]
- **c[]** = superdiagonal, indexed [0, ..., n-2]
- **x[]** = right-hand side (input) / solution (output)

## Example

Create a test case with known solution x = [1, 2, 3, 4, 5]:

```bash
# Create coefficient files
echo -e "0\n-1\n-1\n-1\n-1" > a.txt
echo -e "4\n4\n4\n4\n4" > b.txt
echo -e "1\n1\n1\n1\n0" > c.txt
echo -e "5\n6\n7\n8\n9" > d.txt

# Solve
./thomas -n 5 --diag-a a.txt --diag-b b.txt --diag-c c.txt --rhs d.txt
```

Output:

```text

Solution (N=5):
x[0] = 1.000000
x[1] = 2.000000
x[2] = 3.000000
x[3] = 4.000000
x[4] = 5.000000

```

## Algorithm

The Thomas algorithm consists of two phases:

1. **Forward elimination** - Compute scratch coefficients
2. **Back-substitution** - Solve for the solution vector

Both operations run in O(n) time with minimal memory overhead.

### MPI Implementation

The MPI version distributes the tridiagonal system across multiple processes:
- Rank 0 reads coefficient files and broadcasts data
- Each process solves its local portion independently
- Results are gathered back to rank 0 for output
