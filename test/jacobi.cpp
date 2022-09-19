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

int main(){

    test_jacobi_rotate();


    return 0;
}