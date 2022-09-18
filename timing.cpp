#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h> 
#include <cmath>
#include <armadillo>
#include "useful.hpp"

int main(){
    //dimension of matrix and declaration
    int k, l, n_runs = 50;
    std::vector<double> time_slow(n_runs), time_fast(n_runs);
    double temp;

    std::string filename_s = "chrono_slow.txt", filename_f = "chrono_fast.txt";
    
    // Create an output file stream and conncect it to our file name
    std::ofstream ofile_s, ofile_f; 
    ofile_s.open (filename_s);
    ofile_f.open (filename_f); 

    ofile_s <<"t dt" <<std::endl;
    ofile_f <<"t dt" <<std::endl;

    for (int N = 2; N <= 100; N++){
        arma::mat A = arma::mat(N,N).eye();

        // fill test matrix
        A(0,N-1) = 0.5;
        A(N-1,0) = A(0,N-1);
        A(1,N-2) = -0.7;
        A(N-2,1) = A(1,N-2);

        for (int i = 0; i < n_runs; i++){
            auto t1_slow = std::chrono::high_resolution_clock::now();
            temp = max_offdiag_symmetric(A, k, l);
            auto t2_slow = std::chrono::high_resolution_clock::now();

            auto t1_fast = std::chrono::high_resolution_clock::now();
            temp = fast_max_offdiag_symmetric(A, k, l);
            auto t2_fast = std::chrono::high_resolution_clock::now();

            time_slow[i] = std::chrono::duration<double>(t2_fast - t1_fast).count();
            time_fast[i] = std::chrono::duration<double>(t2_slow - t1_slow).count();
        }

        double duration_slow = std::accumulate(time_slow.begin(), time_slow.end(), 0.0)/time_slow.size();
        double duration_fast = std::accumulate(time_fast.begin(), time_fast.end(), 0.0)/time_fast.size();

        double var_slow = 0, var_fast = 0;
        for (int i = 0; i < n_runs; i++){
            var_slow += (time_slow[i] - duration_slow)*(time_slow[i] - duration_slow);
            var_fast += (time_fast[i] - duration_fast)*(time_fast[i] - duration_fast);
            // pow(x, 2) is slower than x*x
        }

        var_slow /= n_runs-1;
        var_fast /= n_runs-1;

        double stddev_slow = sqrt(var_slow);
        double stddev_fast = sqrt(var_fast);

        ofile_s << scientific_format(duration_slow, 12, 12) << " " << scientific_format(stddev_slow, 12, 12) << std::endl;
        ofile_f << scientific_format(duration_fast, 12, 12) << " " << scientific_format(stddev_fast, 12, 12) <<std::endl;
        
    }

    std::cout << "Done! Check out the output files" << std::endl;

    ofile_s.close();
    ofile_f.close();

    return 0;
}