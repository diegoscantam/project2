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

// function useful for asserting if numbers are equal
template <typename T> bool are_equal(const T a, const T b, const T epsilon = 1e-10) {
	return std::abs(a-b) < epsilon;
}

void test_create_tridiagonal_vec(){
    int N = 4, k, l;
    arma::vec a(N-1, arma::fill::ones);
    arma::vec b(N, arma::fill::ones);
    arma::vec c(N-1, arma::fill::ones);

    arma::mat A = create_tridiagonal(a, b, c);

    // test diag
    assert(are_equal(A(N-1,N-1), 1.));  
    // test sub diag
    assert(are_equal(A(N-2,N-3), 1.)); 
    // test sup diag
    assert(are_equal(A(N-3,N-2), 1.)); 
    // test off-diag
    assert(are_equal(A(0,N-1), 0.));
}

void test_create_tridiagonal_double(){
    int N = 4, k, l;
    double a = -1.5, b = 2., c = 5.;

    arma::mat A = create_tridiagonal(N, a, b, c);

    // test diag
    assert(are_equal(A(N-1,N-1), 2.));  
    // test sub diag
    assert(are_equal(A(N-2,N-3), -1.5)); 
    // test sup diag
    assert(are_equal(A(N-3,N-2), 5.)); 
    // test off-diag
    assert(are_equal(A(0,N-1), 0.));
}

void test_create_tridiagonal_symm(){
    int N = 4, k, l;
    double a = -1.5, b = 2.;

    arma::mat A = create_symmetric_tridiagonal(N, a, b);

    // test diag
    assert(are_equal(A(N-1,N-1), 2.));  
    // test sub diag
    assert(are_equal(A(N-2,N-3), -1.5)); 
    // test sup diag
    assert(are_equal(A(N-3,N-2), -1.5)); 
    // test off-diag
    assert(are_equal(A(0,N-1), 0.));
}

void test_max_offdiag_symmetric(){
    int N = 4, k, l;
    arma::mat A = arma::mat(N,N).eye();

    // fill test matrix
    A(0,N-1) = 0.5;
    A(N-1,0) = A(0,N-1);
    A(1,N-2) = -0.7;
    A(N-2,1) = A(1,N-2);

    // call function
    double max_off = max_offdiag_symmetric(A, k, l);

    // test
    assert(are_equal(max_off, 0.7));
    assert(k == 1);
    assert(l == N-2);
}

void test_fast_max_offdiag_symmetric(){
    int N = 4, k, l;
    arma::mat A = arma::mat(N,N).eye();
    double *a = A.memptr();

    // fill test matrix
    A(0,N-1) = 0.5;
    A(N-1,0) = A(0,N-1);
    A(1,N-2) = -0.7;
    A(N-2,1) = A(1,N-2);

    // call function
    double max_off = fast_max_offdiag_symmetric(A, k, l);

    // test
    assert(are_equal(max_off, 0.7));
    assert(k == 1);
    assert(l == N-2);

    // reset and test the overloading of the same function
    max_off = 0;
    max_off = fast_max_offdiag_symmetric(N, a, k, l);

    assert(are_equal(max_off, 0.7));
    assert(k == 1);
    assert(l == N-2);

}

int main(){

    test_create_tridiagonal_vec();
    test_create_tridiagonal_double();
    test_create_tridiagonal_symm();
    test_max_offdiag_symmetric();
    test_fast_max_offdiag_symmetric();

    return 0;
}