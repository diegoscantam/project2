
// include guard
#ifndef __useful_hpp__  
#define __useful_hpp__

// include headers
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <armadillo>

// Return a string with a double in scientific notation
std::string scientific_format(double d, const int& width, const int& prec);

// Return a string with a vector<double> in scientific notation
std::string scientific_format(const std::vector<double>& v, const int& width, const int& prec);

// Return solution v to the problem A*v=g, with Thomas' algorithm.
// Input vectors: a subdiagonal, b diagonal, c superdiagonal of matrix A.  
std::vector<double> thomas_algo(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& g);

// Return solution v to the problem A*v=g, with Thomas' algorithm with signature (-1, 2, -1).
std::vector<double> specific_algo(std::vector<double>& g);

// Create a tridiagonal matrix tridiag(a,d,e) of size n*n, 
// from scalar input a, d, and e. That is, create a matrix where
// - all n-1 elements on the subdiagonal have value a
// - all n elements on the diagonal have value d
// - all n-1 elements on the superdiagonal have value e
arma::mat create_tridiagonal(int n, double a, double d, double e);

// Create tridiagonal matrix from vectors.
// - lower diagonal: vector a, lenght n-1
// - main diagonal:  vector d, lenght n
// - upper diagonal: vector e, lenght n-1
arma::mat create_tridiagonal(const arma::vec& a, const arma::vec& d, const arma::vec& e);

// Create a symmetric tridiagonal matrix tridiag(a,d,a) of size n*n
// from scalar input a and d.
arma::mat create_symmetric_tridiagonal(int n, double a, double d);

// Determine the the max off-diagonal element of a symmetric matrix A
// - Saves the matrix element indicies to k and l 
// - Returns absolute value of A(k,l) as the function return value
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l);
double fast_max_offdiag_symmetric(const arma::mat& A, int& k, int& l);
double fast_max_offdiag_symmetric(const int N, double* A, int& k, int& l);

// Performs a single Jacobi rotation, to "rotate away"
// the off-diagonal element at A(k,l).
// - Assumes symmetric matrix, so we only consider k < l
// - Modifies the input matrices A and R
void jacobi_rotate(arma::mat& A, arma::mat& R, int k, int l);

// Jacobi method eigensolver:
// - Runs jacobo_rotate until max off-diagonal element < eps
// - Writes the eigenvalues as entries in the vector "eigenvalues"
// - Writes the eigenvectors as columns in the matrix "eigenvectors"
//   (The returned eigenvalues and eigenvectors are sorted using arma::sort_index)
// - Stops if it the number of iterations reaches "maxiter"
// - Writes the number of iterations to the integer "iterations"
// - Sets the bool reference "converged" to true if convergence was reached before hitting maxiter
void jacobi_eigensolver(arma::mat& A, double eps, arma::vec& eigenvalues, arma::mat& eigenvectors, const long int maxiter, long int& iterations, bool& converged);

#endif  // end of include guard __useful_hpp__
