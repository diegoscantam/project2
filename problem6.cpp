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

    int n = 10; //n steps
    double x_min=0.;
    double x_max=1.;
    double v_0=0.;
    double v_n=0.;
    int width=30;
    int prec=10;
    double h = (x_max - x_min) / n;

    arma::vec x(n+1), v1(n+1), v2(n+1), v3(n+1);

    x(0) = x_min;
    for (int i=0; i < n; i++){
        x(i+1) = x(i) + h;
    }

    arma::mat A = create_symmetric_tridiagonal(n-1, -1./(h*h), 2./(h*h));

    arma::vec eigvals(n-1);
    arma::mat eigvecs(n-1, n-1);
    long maxiter = 1e3, iterations;
    bool converged;
    jacobi_eigensolver(A, 1.e-8, eigvals, eigvecs, maxiter, iterations, converged);

    v1(0) = v_0;
    v2(0) = v_0;
    v3(0) = v_0;
    for (int i = 0; i<n-1; i++){
        v1(i+1) = eigvecs(i, 0);
        v2(i+1) = eigvecs(i, 1);
        v3(i+1) = eigvecs(i, 2);
    }
    v1(n) = v_0;
    v2(n) = v_0;
    v3(n) = v_0;

    std::string filename = "data6.txt";
    std::ofstream ofile;
    ofile.open(filename);

    for (int i = 0; i < n+1; i++){
        ofile << scientific_format(x(i), width, prec) << scientific_format(v1(i), width, prec ) << scientific_format(v2(i), width, prec ) << scientific_format(v3(i), width, prec )<<  std::endl;
    }

    ofile.close();


    return 0;
}