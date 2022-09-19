#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h> 
#include <cmath>
#include <cassert>
#include <armadillo>
#include "useful.hpp"




// return a string in scientific notation
std::string scientific_format(double d, const int& width, const int& prec){

    std::stringstream s;
    s << std::setw(width) << std::setprecision(prec) << std::scientific << d;
    return s.str();
}


// Return a string with a vector<double> in scientific notation
std::string scientific_format(const std::vector<double>& v, const int& width, const int& prec)
{
  std::stringstream ss;
  for(double elem : v)
  {
    ss << scientific_format(elem, width, prec);
  }
  return ss.str();
}


// Return solution v to the problem A*v=g, with Thomas' algorithm.
// Input vectors: a subdiagonal, b diagonal, c superdiagonal of matrix A.
std::vector<double> thomas_algo(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& g){

    int n= b.size() + 1;

    std::vector<double> b_tilde(n-1);
    std::vector<double> g_tilde(n-1);
    std::vector<double> v(n-1);

    b_tilde[0]= b[0];
    g_tilde[0]= g[0];

    for(int i=1; i<=n-2; i++){
        b_tilde[i]=b[i] - a[i-1]*c[i-1]/b_tilde[i-1];
        g_tilde[i]=g[i] - a[i-1]*g_tilde[i-1]/b_tilde[i-1];

    }
    v[n-2]=g_tilde[n-2]/b_tilde[n-2];
    for(int i=n-3; i>=0; i--){
        v[i]=(g_tilde[i]- c[i]*v[i+1])/b_tilde[i];
    }


    return v;

}

std::vector<double> specific_algo(std::vector<double>& g){

    int n = g.size() + 1;

    std::vector<double> b_tilde(n-1);
    std::vector<double> g_tilde(n-1);
    std::vector<double> v(n-1);

    b_tilde[0]= 2.;
    g_tilde[0]= g[0];

    for(int i=1; i<=n-2; i++){
        b_tilde[i]=2. - 1./b_tilde[i-1];
        g_tilde[i]=g[i] + 1.*g_tilde[i-1]/b_tilde[i-1];

    }
    v[n-2]=g_tilde[n-2]/b_tilde[n-2];
    for(int i=n-3; i>=0; i--){
        v[i]=(g_tilde[i] + 1.*v[i+1])/b_tilde[i];
    }


    return v;

}

// Create tridiagonal matrix from vectors.
// - lower diagonal: vector a, lenght n-1
// - main diagonal:  vector d, lenght n
// - upper diagonal: vector e, lenght n-1
arma::mat create_tridiagonal(const arma::vec& a, const arma::vec& d, const arma::vec& e)
{
  int n = d.n_elem;

  // Start from identity matrix
  arma::mat A = arma::mat(n, n, arma::fill::eye);

  // Fill first row (row index 0)
  A(0,0) = d(0);
  A(0,1) = e(0);

  for(int i=1; i<=n-2; i++){
      A(i, i-1) = a(i-1);
      A(i, i) = d(i);
      A(i, i+1) = e(i);
  }
  // Loop that fills rows 2 to n-1 (row indices 1 to n-2)
  
  // Fill last row (row index n-1)

  A(n-1, n-1) = d(n-1);
  A(n-1, n-2) = a(n-2);
  return A;
}

// Create a tridiagonal matrix tridiag(a,d,e) of size n*n, 
// from scalar input a, d, and e. That is, create a matrix where
// - all n-1 elements on the subdiagonal have value a
// - all n elements on the diagonal have value d
// - all n-1 elements on the superdiagonal have value e
arma::mat create_tridiagonal(int n, double a, double d, double e)
{
  // Start from identity matrix
  arma::mat A = arma::mat(n, n, arma::fill::eye);

  // Fill the first row (row index 0), e.g.
  A(0,0) = d;
  A(0,1) = e;

  // Loop that fills rows 2 to n-1 (row indices 1 to n-2)
  for(int i=1; i<=n-2; i++){
      A(i, i-1) = a;
      A(i, i) = d;
      A(i, i+1) = e;
  }
  // Fill last row (row index n-1)
  A(n-1, n-1) = d;
  A(n-1, n-2) = a;
  return A;
}

// Create a symmetric tridiagonal matrix tridiag(a,d,a) of size n*n
// from scalar input a and d.
arma::mat create_symmetric_tridiagonal(int n, double a, double d)
{
  // Call create_tridiagonal and return the result
  return create_tridiagonal(n, a, d, a);
}


// Determine the max off-diagonal element of a symmetric matrix A with fast algorithm
// - Saves the matrix element indicies to k and l 
// - Returns absolute value of A(k,l) as the function return value
double fast_max_offdiag_symmetric(const arma::mat& A, int& k, int& l){
  double max = 0, x=max;
  int N = A.n_rows;

  //assert(A.is_square());
  //assert(N > 1);

  k=0;
  l=1;
  for(int q = 2; q <= N*(N-1); q++){
    if( (q-1)%N +1 > (q-1)/N  +1  ){
      x = std::abs(A((q-1)/N , (q-1)% N ));
      if( x > max ){
        k = (q-1)/N;
        l = (q-1)%N;
        max = x;
      }   
    }
  }

  return max;
}

// A function that finds the max off-diag element of a symmetric matrix A.
// - The matrix indices of the max element are returned by writing to the  
//   int references k and l (row and column, respectively)
// - The value of the max element A(k,l) is returned as the function
//   return value
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l)
{
  int N = A.n_rows;

  //assert(A.is_square());
  //assert(N > 1);

  k=0;
  l=1;
  auto maxval = A(k, l);

  for (int i=0; i < N; i++){
    for (int j = i + 1; j < N; j++){
      auto A_ij = std::abs(A(i,j));
      if(A_ij > maxval){
        maxval = A_ij;
        k = i;
        l = j;
      }
    }
  }

  return maxval;
}