#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <mpi.h>

#include "common/timer.h"
#include "common/dummy.h"

void help(const char* prgname)
{
   fprintf(stderr,"%s\n", prgname);
   fprintf(stderr,"   --help|-h              : write this help\n");
   fprintf(stderr,"   --npoints | -n #       : number of solution points (default: 100)\n");
   fprintf(stderr,"   --diag-a file          : file with subdiagonal values\n");
   fprintf(stderr,"   --diag-b file          : file with main diagonal values\n");
   fprintf(stderr,"   --diag-c file          : file with superdiagonal values\n");
   fprintf(stderr,"   --rhs file             : file with right-hand side vector\n");
   fprintf(stderr,"\n   Note: Each file should contain one value per line\n");
}

// Sequential Thomas algorithm for local solve
void thomas_local(const int X, double x[], 
                  const double a[], const double b[],
                  const double c[], double scratch[])
{
    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (int ix = 1; ix < X; ix++) {
        if (ix < X-1){
            scratch[ix] = c[ix] / (b[ix] - a[ix] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - a[ix] * x[ix - 1]) / (b[ix] - a[ix] * scratch[ix - 1]);
    }

    for (int ix = X - 2; ix >= 0; ix--)
        x[ix] -= scratch[ix] * x[ix + 1];
}

int main (int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 100;                    // default number of solution points
    std::string file_a = "";        // subdiagonal file
    std::string file_b = "";        // main diagonal file
    std::string file_c = "";        // superdiagonal file
    std::string file_rhs = "";      // right-hand side file

    #define check_index(i,str) \
    if ((i) >= argc) \
       { if(rank==0) fprintf(stderr,"Missing argument for %s\n", str); MPI_Finalize(); return 1; }

    for (int i = 1; i < argc; i++) {
        std::string key( argv[i] );

        if ( key == "-h" || key == "--help")
        {
            if (rank == 0) help( argv[0] );
            MPI_Finalize();
            return 0;
        }
        else if (key == "-n" || key == "--npoints")
        {
            check_index(++i,"-n");
            if (isdigit(*argv[i]))
                N = atoi( argv[i] );
        }
        else if (key == "--diag-a")
        {
            check_index(++i,"--diag-a");
            file_a = argv[i];
        }
        else if (key == "--diag-b")
        {
            check_index(++i,"--diag-b");
            file_b = argv[i];
        }
        else if (key == "--diag-c")
        {
            check_index(++i,"--diag-c");
            file_c = argv[i];
        }
        else if (key == "--rhs")
        {
            check_index(++i,"--rhs");
            file_rhs = argv[i];
        }
        else
        {
            if (rank == 0) {
                fprintf(stderr,"Unknown option %s\n", key.c_str());
                help(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Read global data on rank 0, then distribute
    std::vector<double> global_a(N), global_b(N), global_c(N), global_x(N);
    
    if (rank == 0) {
        // Load from files if provided, otherwise use default values
        if (!file_a.empty()) {
            FILE* f = fopen(file_a.c_str(), "r");
            if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_a.c_str()); MPI_Finalize(); return 1; }
            for (int i = 0; i < N && fscanf(f, "%lf", &global_a[i]) == 1; i++);
            fclose(f);
        } else {
            for (int i = 0; i < N; i++) global_a[i] = (i > 0) ? 1.0 : 0.0;
        }
        
        if (!file_b.empty()) {
            FILE* f = fopen(file_b.c_str(), "r");
            if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_b.c_str()); MPI_Finalize(); return 1; }
            for (int i = 0; i < N && fscanf(f, "%lf", &global_b[i]) == 1; i++);
            fclose(f);
        } else {
            for (int i = 0; i < N; i++) global_b[i] = 4.0;
        }
        
        if (!file_c.empty()) {
            FILE* f = fopen(file_c.c_str(), "r");
            if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_c.c_str()); MPI_Finalize(); return 1; }
            for (int i = 0; i < N && fscanf(f, "%lf", &global_c[i]) == 1; i++);
            fclose(f);
        } else {
            for (int i = 0; i < N; i++) global_c[i] = (i < N-1) ? 1.0 : 0.0;
        }
        
        if (!file_rhs.empty()) {
            FILE* f = fopen(file_rhs.c_str(), "r");
            if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_rhs.c_str()); MPI_Finalize(); return 1; }
            for (int i = 0; i < N && fscanf(f, "%lf", &global_x[i]) == 1; i++);
            fclose(f);
        } else {
            for (int i = 0; i < N; i++) global_x[i] = (double)i;
        }
    }

    // Broadcast global data to all processes
    MPI_Bcast(global_a.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_c.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divide work among processes
    int local_size = (N + size - 1) / size;  // ceiling division
    int start_idx = rank * local_size;
    int end_idx = (rank + 1) * local_size < N ? (rank + 1) * local_size : N;
    int actual_local_size = end_idx - start_idx;

    if (actual_local_size <= 0) actual_local_size = 0;

    std::vector<double> local_a(actual_local_size);
    std::vector<double> local_b(actual_local_size);
    std::vector<double> local_c(actual_local_size);
    std::vector<double> local_x(actual_local_size);
    std::vector<double> local_scratch(actual_local_size);

    // Copy local data
    for (int i = 0; i < actual_local_size; i++) {
        local_a[i] = global_a[start_idx + i];
        local_b[i] = global_b[start_idx + i];
        local_c[i] = global_c[start_idx + i];
        local_x[i] = global_x[start_idx + i];
    }

    if (rank == 0) {
        printf("MPI Thomas Solver\n");
        printf("  Global size: %d\n", N);
        printf("  Number of processes: %d\n", size);
        printf("  Local size per process: %d\n", local_size);
    }

    // Barrier to synchronize before solve
    MPI_Barrier(MPI_COMM_WORLD);

    TimerType t_start = getTimeStamp();

    // Run multiple times for accurate timing
    int iterations = (N < 1000) ? 100 : (N < 10000) ? 10 : 1;
    for (int iter = 0; iter < iterations; iter++) {
        // Reset x for each iteration
        if (!file_rhs.empty()) {
            if (rank == 0) {
                FILE* f = fopen(file_rhs.c_str(), "r");
                if (f) {
                    for (int i = 0; i < N && fscanf(f, "%lf", &global_x[i]) == 1; i++);
                    fclose(f);
                }
            }
            MPI_Bcast(global_x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            for (int i = 0; i < N; i++) global_x[i] = (double)i;
        }

        // Copy to local arrays
        for (int i = 0; i < actual_local_size; i++) {
            local_x[i] = global_x[start_idx + i];
        }

        // Solve local system
        if (actual_local_size > 0) {
            thomas_local(actual_local_size, local_x.data(), 
                         local_a.data(), local_b.data(), 
                         local_c.data(), local_scratch.data());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TimerType t_end = getTimeStamp();
    double elapsed = getElapsedTime(t_start, t_end);

    // Gather all results back to rank 0
    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);
    
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i + 1) * local_size < N ? local_size : (N - i * local_size);
        if (recvcounts[i] < 0) recvcounts[i] = 0;
        if (i > 0) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    MPI_Gatherv(local_x.data(), actual_local_size, MPI_DOUBLE,
                global_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print solution and timing on rank 0
    if (rank == 0) {
        printf("\nSolution (N=%d):\n", N);
        for (int i = 0; i < N; i++) {
            printf("x[%d] = %.6f\n", i, global_x[i]);
        }

        printf("\n=== Performance Metrics ===\n");
        printf("Problem size (N): %d\n", N);
        printf("Number of processes: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Total time: %.6f seconds\n", elapsed);
        printf("Average time per solve: %.6e seconds\n", elapsed / iterations);
        printf("Operations per solve (local): %lld\n", 4LL*local_size);
        printf("Total operations: %lld\n", 4LL*N);
        
        double time_per_iter = elapsed / iterations;
        double gflops = (4.0 * N) / (time_per_iter * 1e9);
        printf("Average GFLOP/s: %.6f\n", gflops);
        printf("Theoretical speedup (vs 1 proc): %.2f x\n", (double)size);
        printf("Theoretical efficiency: %.2f %%\n", (100.0 / size));
    }

    MPI_Finalize();
    return 0;
}
