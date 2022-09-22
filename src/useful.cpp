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

template <typename T> bool are_equal(const T a, const T b, const T epsilon = 1e-8) {
	return std::abs(a) - std::abs(b) < epsilon;
}

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

// A function that finds the max off-diag element of a symmetric matrix A.
// - The matrix indices of the max element are returned by writing to the  
//   int references k and l (row and column, respectively)
// - The value of the max element A(k,l) is returned as the function
//   return value
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l)
{
  int N = A.n_rows;

  k=0;
  l=1;
  auto maxval = A(k, l);

  for (int i=0; i < N; i++){
    for (int j = i + 1; j < N; j++){
      auto A_ij = A(i,j);
      if(A_ij*A_ij > maxval*maxval){
        maxval = A_ij;
        k = i;
        l = j;
      }
    }
  }

  return std::abs(maxval);
}

// A function that finds the max off-diag element of a symmetric matrix A.
// - The matrix indices of the max element are returned by writing to the  
//   int references k and l (row and column, respectively)
// - The value of the max element A(k,l) is returned as the function
//   return value
double max_offdiag_symmetric(const int N, double* a, int& k, int& l)
{
  k=0;
  l=1;
  double max = *(a+N);

  for (int i=0; i < N; i++){
    for (int j = i + 1; j < N; j++){
      auto a_ij = *(a+j*N+i);
      if(a_ij*a_ij > max*max){
        max = a_ij;
        k = i;
        l = j;
      }
    }
  }

  return std::abs(max);
}


// Performs a single Jacobi rotation, to "rotate away"
// the off-diagonal element at A(k,l).
// - Assumes symmetric matrix, so we only consider k < l
// - Modifies the input matrices A and R
void jacobi_rotate(arma::mat& A, arma::mat& R, int k, int l){

  int N = A.n_rows;

  // save in mem the matrix elements
  double a_kk = A(k,k), a_ll = A(l,l), a_kl = A(k, l), c, s;

  // Determine t, c, s of Jacobi rotation
  double tau  = (a_ll - a_kk)/(2.*a_kl), t;
  if  (tau >= 0.)
    t = 1./(tau+std::sqrt(1.+tau*tau));
  else
    t = -1./(-tau+std::sqrt(1.+tau*tau));
  c = 1./std::sqrt(1.+t*t);
  s = c*t;

  // Update A elements and R  elements
  A(k, k) = a_kk*c*c - 2.*a_kl*c*s + a_ll*s*s;
  A(l, l) = a_ll*c*c + 2.*a_kl*c*s + a_kk*s*s;
  A(k, l) = 0.;
  A(l, k) = 0.;
  for (int i = 0; i < N; i++){
    double r_ik = R(i, k);
    R(i, k) =r_ik*c - R(i, l)*s;
    R(i, l) = R(i, l)*c + r_ik*s;

    if (i == k || i == l)
      continue;
    double a_ik = A(i, k);
    A(i, k) = a_ik*c - A(i, l)*s;
    A(k, i) = A(i, k);
    A(i, l) = A(i, l)*c + a_ik*s;
    A(l, i) = A(i, l);
  }

}

// Jacobi method eigensolver:
// - Runs jacobo_rotate until max off-diagonal element < eps
// - Writes the eigenvalues as entries in the vector "eigenvalues"
// - Writes the eigenvectors as columns in the matrix "eigenvectors"
//   (The returned eigenvalues and eigenvectors are sorted using arma::sort_index)
// - Stops if it the number of iterations reaches "maxiter"
// - Writes the number of iterations to the integer "iterations"
// - Sets the bool reference "converged" to true if convergence was reached before hitting maxiter
void jacobi_eigensolver(arma::mat& A, double eps, arma::vec& eigenvalues, arma::mat& eigenvectors, const long int maxiter, long int& iterations, bool& converged){
    iterations = 0;

    //initialize matrix R1
    int N = A.n_rows;
    arma::mat R = arma::eye(N,N);
    int k,l;
    double *a = A.memptr();

    double max = max_offdiag_symmetric(N,a,k,l);
    converged = true;
    while (max > eps){
        jacobi_rotate(A,R,k,l);
        max = max_offdiag_symmetric(A,k,l);
        iterations++;
        
        if (iterations > maxiter){
            converged = false;
            break;
        }
    }
    eigenvalues = A.diag();
    arma::uvec sidx = sort_index(eigenvalues);
    eigenvalues = eigenvalues(sidx);
    eigenvectors = R.cols(sidx);

}


