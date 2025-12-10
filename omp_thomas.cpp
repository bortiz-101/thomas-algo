#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <omp.h>
#include <algorithm>

#include "common/timer.h"
#include "common/dummy.h"

void help(const char* prgname)
{
   fprintf(stderr,"%s\n", prgname);
   fprintf(stderr,"   -h, --help              : Show this help\n");
   fprintf(stderr,"   -n, --npoints #         : Problem size (default: 100)\n");
   fprintf(stderr,"   -i, --iterations #      : Iteration count (default: auto)\n");
   fprintf(stderr,"   --threads #             : OpenMP thread count (default: auto)\n");
   fprintf(stderr,"   --sequential            : Run sequential Thomas (baseline)\n");
   fprintf(stderr,"   --simd                  : Run SIMD-vectorized Thomas\n");
   fprintf(stderr,"   --pcr                   : Run Thomas-PCR hybrid (Laszlo et al. 2016)\n");
   fprintf(stderr,"   --pcr-simd              : Run PCR with SIMD hints (default)\n");
   fprintf(stderr,"   --diag-a file           : Subdiagonal coefficient file\n");
   fprintf(stderr,"   --diag-b file           : Main diagonal coefficient file\n");
   fprintf(stderr,"   --diag-c file           : Superdiagonal coefficient file\n");
   fprintf(stderr,"   --rhs file              : Right-hand side vector file\n");
}

// Thomas-PCR Hybrid: Fast for shared memory (Laszlo, Gilles, Appleyard, ACM TOMS 42(31), 2016)

// @misc{kang2019ptdma,
//     title  = {Parallel tri-diagonal matrix solver using cyclic reduction (CR), parallel CR (PCR), and Thomas+PCR hybrid algorithm},
//     author = {Kang, Ji-Hoon},
//     url    = https://github.com/jihoonakang/parallel_tdma_cpp},
//     year   = {2019}
// }
void pcr_solve_openmp(int N, std::vector<double>& a, std::vector<double>& b,
                      std::vector<double>& c, std::vector<double>& x)
{
    std::vector<double> a_work = a;
    std::vector<double> b_work = b;
    std::vector<double> c_work = c;
    std::vector<double> x_work = x;

    // Phase 1: Forward Thomas elimination (eliminates all but first and last rows)
    for (int i = 1; i < N; i++) {
        if (std::abs(b_work[i - 1]) > 1e-15) {
            double alpha = -a_work[i] / b_work[i - 1];
            a_work[i] = alpha * a_work[i - 1];
            b_work[i] += alpha * c_work[i - 1];
            x_work[i] += alpha * x_work[i - 1];
        }
    }

    //Backward Thomas elimination (eliminate above diagonal)
    for (int i = N - 2; i >= 0; i--) {
        if (std::abs(b_work[i + 1]) > 1e-15) {
            double beta = -c_work[i] / b_work[i + 1];
            c_work[i] = beta * c_work[i + 1];
            x_work[i] += beta * x_work[i + 1];
            if (i > 0) {
                a_work[i] += beta * a_work[i + 1];
            }
        }
    }


    // Now apply PCR to the remaining structure
    // Normalize boundaries
    double denom = b_work[0];
    if (std::abs(denom) > 1e-15) {
        a_work[0] /= denom;
        b_work[0] = 1.0;
        c_work[0] /= denom;
        x_work[0] /= denom;
    }

    denom = b_work[N - 1];
    if (std::abs(denom) > 1e-15) {
        a_work[N - 1] /= denom;
        b_work[N - 1] = 1.0;
        c_work[N - 1] /= denom;
        x_work[N - 1] /= denom;
    }

    // Phase 2: PCR reduction on already-reduced system
    int max_phases = 0;
    int temp = N;
    while (temp > 1) {
        max_phases++;
        temp /= 2;
    }

    for (int phase = 0; phase < max_phases; phase++) {
        int stride = 1 << phase;

        #pragma omp parallel for schedule(static)
        for (int i = stride; i < N - stride; i++) {
            double a_left = (i - stride >= 0) ? a_work[i - stride] : 0.0;
            double c_left = (i - stride >= 0) ? c_work[i - stride] : 0.0;
            double x_left = (i - stride >= 0) ? x_work[i - stride] : 0.0;

            double a_right = (i + stride < N) ? a_work[i + stride] : 0.0;
            double c_right = (i + stride < N) ? c_work[i + stride] : 0.0;
            double x_right = (i + stride < N) ? x_work[i + stride] : 0.0;

            double new_b = b_work[i] - a_work[i] * c_left - c_work[i] * a_right;

            if (std::abs(new_b) > 1e-15) {
                double new_a = -a_work[i] * a_left;
                double new_c = -c_work[i] * c_right;
                double new_x = x_work[i] - a_work[i] * x_left - c_work[i] * x_right;

                a_work[i] = new_a / new_b;
                b_work[i] = 1.0;
                c_work[i] = new_c / new_b;
                x_work[i] = new_x / new_b;
            }
        }

        #pragma omp barrier
    }

    // Copy solution back
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = x_work[i];
    }
}

// Sequential Thomas algorithm 
void thomas_seq(int N, std::vector<double>& a, std::vector<double>& b,
                std::vector<double>& c, std::vector<double>& x)
{
    std::vector<double> scratch(N);

    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    // Forward elimination
    for (int i = 1; i < N; i++) {
        if (i < N - 1) {
            scratch[i] = c[i] / (b[i] - a[i] * scratch[i - 1]);
        }
        x[i] = (x[i] - a[i] * x[i - 1]) / (b[i] - a[i] * scratch[i - 1]);
    }

    // Back-substitution
    for (int i = N - 2; i >= 0; i--) {
        x[i] -= scratch[i] * x[i + 1];
    }
}

// SIMD-accelerated Thomas algorithm
void thomas_simd(int N, std::vector<double>& a, std::vector<double>& b,
                 std::vector<double>& c, std::vector<double>& x)
{
    std::vector<double> scratch(N);

    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    // Forward elimination with SIMD
    #pragma omp simd
    for (int i = 1; i < N; i++) {
        if (i < N - 1) {
            scratch[i] = c[i] / (b[i] - a[i] * scratch[i - 1]);
        }
        x[i] = (x[i] - a[i] * x[i - 1]) / (b[i] - a[i] * scratch[i - 1]);
    }

    for (int i = N - 2; i >= 0; i--) {
        x[i] -= scratch[i] * x[i + 1];
    }
}

// SIMD PCR - Also uses Thomas-PCR hybrid for consistency
void pcr_solve_simd(int N, std::vector<double>& a, std::vector<double>& b,
                    std::vector<double>& c, std::vector<double>& x)
{
    std::vector<double> a_work = a;
    std::vector<double> b_work = b;
    std::vector<double> c_work = c;
    std::vector<double> x_work = x;

    // Phase 1: Forward Thomas elimination (eliminates all but first and last rows)
    for (int i = 1; i < N; i++) {
        if (std::abs(b_work[i - 1]) > 1e-15) {
            double alpha = -a_work[i] / b_work[i - 1];
            a_work[i] = alpha * a_work[i - 1];
            b_work[i] += alpha * c_work[i - 1];
            x_work[i] += alpha * x_work[i - 1];
        }
    }

    //Backward Thomas elimination (eliminate above diagonal)
    for (int i = N - 2; i >= 0; i--) {
        if (std::abs(b_work[i + 1]) > 1e-15) {
            double beta = -c_work[i] / b_work[i + 1];
            c_work[i] = beta * c_work[i + 1];
            x_work[i] += beta * x_work[i + 1];
            if (i > 0) {
                a_work[i] += beta * a_work[i + 1];
            }
        }
    }

    // After forward+backward Thomas, system is reduced
    // Normalize boundaries
    double denom = b_work[0];
    if (std::abs(denom) > 1e-15) {
        a_work[0] /= denom;
        b_work[0] = 1.0;
        c_work[0] /= denom;
        x_work[0] /= denom;
    }

    denom = b_work[N - 1];
    if (std::abs(denom) > 1e-15) {
        a_work[N - 1] /= denom;
        b_work[N - 1] = 1.0;
        c_work[N - 1] /= denom;
        x_work[N - 1] /= denom;
    }

    // Phase 2: PCR reduction on already-reduced system with SIMD
    int max_phases = 0;
    int temp = N;
    while (temp > 1) {
        max_phases++;
        temp /= 2;
    }

    for (int phase = 0; phase < max_phases; phase++) {
        int stride = 1 << phase;

        #pragma omp parallel for simd schedule(static) collapse(1)
        for (int i = stride; i < N - stride; i++) {
            double a_left = (i - stride >= 0) ? a_work[i - stride] : 0.0;
            double c_left = (i - stride >= 0) ? c_work[i - stride] : 0.0;
            double x_left = (i - stride >= 0) ? x_work[i - stride] : 0.0;

            double a_right = (i + stride < N) ? a_work[i + stride] : 0.0;
            double c_right = (i + stride < N) ? c_work[i + stride] : 0.0;
            double x_right = (i + stride < N) ? x_work[i + stride] : 0.0;

            double new_b = b_work[i] - a_work[i] * c_left - c_work[i] * a_right;

            if (std::abs(new_b) > 1e-15) {
                double new_a = -a_work[i] * a_left;
                double new_c = -c_work[i] * c_right;
                double new_x = x_work[i] - a_work[i] * x_left - c_work[i] * x_right;

                a_work[i] = new_a / new_b;
                b_work[i] = 1.0;
                c_work[i] = new_c / new_b;
                x_work[i] = new_x / new_b;
            }
        }
    }

    // Copy solution back
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        x[i] = x_work[i];
    }
}

int main(int argc, char* argv[])
{
    int N = 100;
    int num_threads = -1;
    int iterations = -1;
    std::string file_a = "", file_b = "", file_c = "", file_rhs = "";
    std::string solver = "pcr-simd";  // Default: PCR with SIMD

    #define check_index(i,str) \
    if ((i) >= argc) \
       { fprintf(stderr,"Missing argument for %s\n", str); return 1; }

    for (int i = 1; i < argc; i++) {
        std::string key(argv[i]);

        if (key == "-h" || key == "--help") {
            help(argv[0]);
            return 0;
        } else if (key == "-n" || key == "--npoints") {
            check_index(++i, "-n");
            N = atoi(argv[i]);
        } else if (key == "-i" || key == "--iterations") {
            check_index(++i, "-i");
            iterations = atoi(argv[i]);
        } else if (key == "--threads") {
            check_index(++i, "--threads");
            num_threads = atoi(argv[i]);
        } else if (key == "--sequential") {
            solver = "sequential";
        } else if (key == "--simd") {
            solver = "simd";
        } else if (key == "--pcr") {
            solver = "pcr";
        } else if (key == "--pcr-simd") {
            solver = "pcr-simd";
        } else if (key == "--diag-a") {
            check_index(++i, "--diag-a");
            file_a = argv[i];
        } else if (key == "--diag-b") {
            check_index(++i, "--diag-b");
            file_b = argv[i];
        } else if (key == "--diag-c") {
            check_index(++i, "--diag-c");
            file_c = argv[i];
        } else if (key == "--rhs") {
            check_index(++i, "--rhs");
            file_rhs = argv[i];
        } else {
            fprintf(stderr, "Unknown option %s\n", key.c_str());
            help(argv[0]);
            return 1;
        }
    }

    if (num_threads > 0)
        omp_set_num_threads(num_threads);

    // Initialize vectors
    std::vector<double> a(N), b(N), c(N), x(N);

    // Load coefficients from files or use defaults
    auto load_vector = [&](const std::string& file, std::vector<double>& vec, double default_val) {
        if (!file.empty()) {
            FILE* f = fopen(file.c_str(), "r");
            if (!f) {
                fprintf(stderr, "Error: Cannot open %s\n", file.c_str());
                return false;
            }
            for (int i = 0; i < N && fscanf(f, "%lf", &vec[i]) == 1; i++);
            fclose(f);
        } else {
            std::fill(vec.begin(), vec.end(), default_val);
        }
        return true;
    };

    // Load with defaults: a[0]=0, b[*]=4, c[N-1]=0, x=[0..N-1]
    if (!load_vector(file_a, a, 1.0)) return 1;
    a[0] = 0.0;
    
    if (!load_vector(file_b, b, 4.0)) return 1;
    
    if (!load_vector(file_c, c, 1.0)) return 1;
    c[N-1] = 0.0;
    
    if (!file_rhs.empty()) {
        FILE* f = fopen(file_rhs.c_str(), "r");
        if (!f) {
            fprintf(stderr, "Error: Cannot open %s\n", file_rhs.c_str());
            return 1;
        }
        for (int i = 0; i < N && fscanf(f, "%lf", &x[i]) == 1; i++);
        fclose(f);
    } else {
        for (int i = 0; i < N; i++) x[i] = (double)i;
    }

    std::vector<double> rhs_original = x;

    // Auto-scale if not specified
    if (iterations < 0) {
        if (N < 1000) iterations = 100;
        else if (N < 10000) iterations = 10;
        else if (N < 100000) iterations = 2;
        else iterations = 1;
    }

    // Run selected algorithm
    std::vector<double> x_test = rhs_original;
    TimerType t_start = getTimeStamp();
    
    if (solver == "sequential") {
        for (int iter = 0; iter < iterations; iter++) {
            x_test = rhs_original;
            thomas_seq(N, a, b, c, x_test);
        }
    } else if (solver == "simd") {
        for (int iter = 0; iter < iterations; iter++) {
            x_test = rhs_original;
            thomas_simd(N, a, b, c, x_test);
        }
    } else if (solver == "pcr") {
        for (int iter = 0; iter < iterations; iter++) {
            x_test = rhs_original;
            pcr_solve_openmp(N, a, b, c, x_test);
        }
    } else {  // pcr-simd (default)
        for (int iter = 0; iter < iterations; iter++) {
            x_test = rhs_original;
            pcr_solve_simd(N, a, b, c, x_test);
        }
    }
    
    TimerType t_end = getTimeStamp();
    
    double time_per_iter = getElapsedTime(t_start, t_end) / iterations;
    double gflops = (4.0 * N) / (time_per_iter * 1e9);
    printf("Avg time/solve: %.6e seconds\n", time_per_iter);
    printf("GFLOP/s: %.6f\n", gflops);

    return 0;
}
