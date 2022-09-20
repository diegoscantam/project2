#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h> 
#include <cmath>
#include <armadillo>
#include <cassert>
#include "useful.hpp"

// Return a bool  for asserting if numbers are equal
template <typename T> bool are_equal(const T a, const T b, const T epsilon = 1e-8) {
	return std::abs(a-b) < epsilon;
}

void test_jacobi_rotate(){
    arma::mat A = arma::mat(3,3).eye();
    arma::mat R = arma::mat(3,3).eye();
    A(1, 1) = 100.;

    jacobi_rotate(A, R, 1, 1);

    assert(are_equal(A(1, 1), 0.));
}

void test_jacobi_eigensolver(){
    arma::mat A(2, 2, arma::fill::zeros);
    A(1, 0) = 1.;
    A(0, 1) = 1.;
    
    double eps = 1.0e-8;
    arma::vec eigenvalues=arma::vec(3);
    arma::mat eigenvectors=arma::mat(3,3);
    
    
    long int maxiter = 1e10, iterations;
    bool converged;
    
    jacobi_eigensolver(A, eps, eigenvalues, eigenvectors, maxiter, iterations, converged);

    assert(are_equal(eigenvalues(0), -1., eps));
    assert(are_equal(eigenvalues(1), 1., eps));
}

void test_NxN(int N){
    
    // Generate random N*N matrix
    arma::mat A = arma::mat(N, N, arma::fill::randu);  

    // Symmetrize the matrix by reflecting the upper triangle to lower triangle
    arma::mat exp_A = arma::symmatu(A);
    arma::mat obs_A = exp_A;

    arma::vec exp_eigval, obs_eigval;
    arma::mat exp_eigvec, obs_eigvec;
    arma::eig_sym(exp_eigval, exp_eigvec, exp_A, "std");

    long int maxiter = 200, iterations;
    double eps = 1.e-4;
    bool converged;
    jacobi_eigensolver(obs_A, eps, obs_eigval, obs_eigvec, maxiter, iterations, converged);

    for (int i = 0; i < N; i++){
        assert(are_equal(obs_eigval(i), exp_eigval(i), 0.7));
        for (int j = 0; j < N; j++){
            assert(are_equal(obs_eigvec(j, i), exp_eigvec(j, i), 2.5));
            assert(are_equal(obs_A(j, i), exp_A(j, i), 2.5));
        }
    }
}

int main(){

    test_jacobi_rotate();
    test_jacobi_eigensolver();
    test_NxN(6);
    
    return 0;
}
