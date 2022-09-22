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
    // set number of cycles for repeated estimations of #iterations (see later)
    int cycles = 5;
    // set maximum number of iterations
    long int maxiter = 1e+8;
    //set parameters for printing
    int width = 30, prec = 10;
    // declare output variables of jacobi algo
    bool converged, converged_tri;
    // set tolerance for jacobi algo
    double eps = 1e-8, mean;

    // Create and open the output file
    std::string filename0 = "iter.txt";
    std::ofstream ofile0;
    ofile0.open(filename0);

    for(int N = 10; N <= 200;  N+=10){
        // for every value of N create the tridiagonal matrix A_tri
        // and solve the problem with jacobi_eigensolver
        arma::vec eigenvalues(N), eigenvalues_tri(N);
        arma::mat eigenvectors(N,N), eigenvectors_tri(N,N);

        // define stepsize and elements of tridiag matrix A_tri
        double h = 1./(N+1), a = -1./(h+h), d = 2./(h*h);

        // fill symmetric tridiag matrix
        arma::mat A_tri = create_symmetric_tridiagonal(N, a, d);

        // declare some parameters for jacobi algo
        long int sum = 0, iterations = 0, iterations_tri = 0;

        // jacobi algo
        jacobi_eigensolver(A_tri, eps, eigenvalues_tri, eigenvectors_tri, maxiter, iterations_tri, converged_tri);
                
        // Create also some dense symmetric matrices, randomly filled, to test
        // the dependence #iterations(N)
        // Some cycles over the same dimension N are performed in order to obtain
        // more consistent results, given that is a random matrix
        for(int i = 0; i < cycles; i++){
            // generic dense matrix
            arma::mat A = arma::mat(N,N).randn(); // fill random values matrix
            A = arma::symmatu(A); // symmetrize

            // jacobi algo
            jacobi_eigensolver(A, eps, eigenvalues, eigenvectors, maxiter, iterations, converged);
            sum += iterations;

        }
        // calculate mean number of iterations over #cycles
        mean = (double) sum / cycles;
        // store data as (N, #iterations for tidiagonal, #iterations for dense)
        ofile0 << scientific_format(N, width, prec ) << scientific_format(iterations_tri, width, prec ) 
                << scientific_format(mean, width, prec )<<  std::endl;
    }


    //close file
    ofile0.close();

    // all good :)
    return 0;
}