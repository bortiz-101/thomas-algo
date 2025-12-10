#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>

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


void thomas(const int X, double x[], 
            const double a[], const double b[],
            const double c[], double scratch[])
{
    /*
     solves Ax = d, where A is a tridiagonal matrix consisting of vectors a, b, c
     X = number of equations
     x[] = initially contains the input, d, and returns x. indexed from [0, ..., X - 1]
     a[] = subdiagonal, indexed from [1, ..., X - 1]
     b[] = main diagonal, indexed from [0, ..., X - 1]
     c[] = superdiagonal, indexed from [0, ..., X - 2]
     scratch[] = scratch space of length X, provided by caller, allowing a, b, c to be const
     not performed in this example: manual expensive common subexpression elimination
     */

    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    /* loop from 1 to X - 1 inclusive */
    for (int ix = 1; ix < X; ix++) {
        if (ix < X-1){
            scratch[ix] = c[ix] / (b[ix] - a[ix] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - a[ix] * x[ix - 1]) / (b[ix] - a[ix] * scratch[ix - 1]);
    }

    /* loop from X - 2 to 0 inclusive */
    for (int ix = X - 2; ix >= 0; ix--)
        x[ix] -= scratch[ix] * x[ix + 1];
}

int main (int argc, char* argv[])
{
    int N = 100;                    // default number of solution points
    std::string file_a = "";        // subdiagonal file
    std::string file_b = "";        // main diagonal file
    std::string file_c = "";        // superdiagonal file
    std::string file_rhs = "";      // right-hand side file

    #define check_index(i,str) \
    if ((i) >= argc) \
       { fprintf(stderr,"Missing argument for %s\n", str); return 1; }

    for (int i = 1; i < argc; i++) {
        std::string key( argv[i] );

        if ( key == "-h" || key == "--help")
        {
            help( argv[0] );
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
            fprintf(stderr,"Unknown option %s\n", key.c_str());
            help(argv[0]);
            return 1;
        }
    }

    // Initialize vectors
    std::vector<double> a(N), b(N), c(N), x(N), scratch(N);
    
    // Load from files if provided, otherwise use default values
    if (!file_a.empty()) {
        FILE* f = fopen(file_a.c_str(), "r");
        if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_a.c_str()); return 1; }
        for (int i = 0; i < N && fscanf(f, "%lf", &a[i]) == 1; i++);
        fclose(f);
    } else {
        for (int i = 0; i < N; i++) a[i] = (i > 0) ? 1.0 : 0.0;
    }
    
    if (!file_b.empty()) {
        FILE* f = fopen(file_b.c_str(), "r");
        if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_b.c_str()); return 1; }
        for (int i = 0; i < N && fscanf(f, "%lf", &b[i]) == 1; i++);
        fclose(f);
    } else {
        for (int i = 0; i < N; i++) b[i] = 4.0;
    }
    
    if (!file_c.empty()) {
        FILE* f = fopen(file_c.c_str(), "r");
        if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_c.c_str()); return 1; }
        for (int i = 0; i < N && fscanf(f, "%lf", &c[i]) == 1; i++);
        fclose(f);
    } else {
        for (int i = 0; i < N; i++) c[i] = (i < N-1) ? 1.0 : 0.0;
    }
    
    if (!file_rhs.empty()) {
        FILE* f = fopen(file_rhs.c_str(), "r");
        if (!f) { fprintf(stderr,"Error: Cannot open %s\n", file_rhs.c_str()); return 1; }
        for (int i = 0; i < N && fscanf(f, "%lf", &x[i]) == 1; i++);
        fclose(f);
    } else {
        for (int i = 0; i < N; i++) x[i] = (double)i;
    }
    
    // Cache original RHS for iteration resets (avoid re-reading from disk)
    std::vector<double> rhs_original = x;

    // Start timer for Thomas algorithm
    TimerType t_start = getTimeStamp();
    
    // Run multiple times for accurate timing on fast solves
    int iterations = (N < 1000) ? 100 : (N < 10000) ? 10 : 1;
    for (int iter = 0; iter < iterations; iter++) {
        // Reset x from cached RHS (in-memory, no disk I/O)
        x = rhs_original;
        
        thomas(N, x.data(), a.data(), b.data(), c.data(), scratch.data());
    }

    TimerType t_end = getTimeStamp();
    double elapsed = getElapsedTime(t_start, t_end);

    // Print solution to terminal (abbreviated for large N)
    printf("Solution (N=%d):\n", N);
    if (N <= 10) {
        // Print all elements for small N
        for (int i = 0; i < N; i++) {
            printf("x[%d] = %.6f\n", i, x[i]);
        }
    } else {
        // Print only first and last elements for large N
        printf("x[0] = %.6f\n", x[0]);
        printf("... (skipped %d middle elements) ...\n", N - 2);
        printf("x[%d] = %.6f\n", N - 1, x[N - 1]);
    }

    // Print timing information
    printf("\n=== Performance Metrics ===\n");
    printf("Problem size (N): %d\n", N);
    printf("Iterations: %d\n", iterations);
    printf("Total time: %.6f seconds\n", elapsed);
    printf("Average time per solve: %.6e seconds\n", elapsed / iterations);
    printf("Operations per solve: %lld (forward pass + backward pass)\n", 4LL*N);
    
    double time_per_iter = elapsed / iterations;
    double gflops = (4.0 * N) / (time_per_iter * 1e9);
    printf("Average GFLOP/s: %.6f\n", gflops);

    return 0;
}