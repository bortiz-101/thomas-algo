#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>

#include "timer.h"
#include "partitioner.hpp"

void help(const char* prgname)
{
   fprintf(stderr,"%s\n", prgname);
   fprintf(stderr,"   --help|-h   : write this help\n");
   fprintf(stderr,"   --npoints | -n #   : number of solution points\n");
   fprintf(stderr,"   --nsteps | -s  #   : maximum number of time-steps\n");
   fprintf(stderr,"   --freq | -f    #   : print status ever <#> steps\n");
}


// void thomas(const int X, double x[restrict X],
//             const double a[restrict X], const double b[restrict X],
//             const double c[restrict X], double scratch[restrict X]) 
int main (int argc, char* argv[])
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

     #define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      std::string key( argv[i++] );

      if ( key == "-h" || key == "--help")
      {
         help( argv[0] );
         return 0;
      }
      else if (key == "-n" || key == "--npoints")
      {
         check_index(i,"-n");
         if (isdigit(*argv[i]))
            N = atoi( argv[i] );
         i++;
      }
      else if (key == "--nsteps" || key == "-s")
      {
         check_index(i,"--nsteps|-s");
         if (isdigit(*argv[i]))
            max_steps = atoi( argv[i] );
         i++;
      }
      else if (key == "--freq" || key == "-f")
      {
         check_index(i,"--freq|-f");
         if (isdigit(*argv[i]))
            print_freq = atoi( argv[i] );
         i++;
      }
      else if (key == "--cfl")
      {
         check_index(i,"--cfl");
         cfl = atof( argv[i++] );
      }
      else if (key == "--write" || key == "-w")
      {
         write_solution = true;
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", key.c_str());
         help(argv[0]);
         return 1;
      }
   }
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