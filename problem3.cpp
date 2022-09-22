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
    int N = 4, k, l;
    arma::mat A = arma::mat(N,N).eye();

    // fill test matrix
    A(0,N-1) = 0.5;
    A(N-1,0) = A(0,N-1);
    A(1,N-2) = -0.7;
    A(N-2,1) = A(1,N-2);

    double* a = A.memptr();

    // call function and print A
    double max_off = max_offdiag_symmetric(N, a, k, l);

    std::cout << "The matrix A is given by:" << std::endl;
    A.print();

    // report
    std::cout << "The maximum off diagonal value is " << max_off << 
                " and it's in position A(" << k+1 << "," << l+1 << ")" << std::endl;

    // all good
    return 0;
}
