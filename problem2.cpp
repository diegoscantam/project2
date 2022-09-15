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



#define PI 3.14159265359
int main(){
    // declare variables
    int N = 6, n = 7;
    double x_min = 0, x_max = 1;
    double h= (x_max - x_min)/n; 


    // fill matrix A
    arma::mat A = arma::mat(N,N);
    double a= -1./(h*h), d = 2./(h*h);
    A = create_tridiagonal(N,a,d,a);

    // solve eigenvalue problem A*v=k*v with armadillo library
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, A);

    // normalize to 1 the eigenvectors
    eigvec= arma::normalise(eigvec);

    // check that analytical solution (lambda & v) is equal to the result obtained
    // by armadillo functions (eigval & eigvec)
    arma::vec lambda(N);
    arma::vec tf = arma::vec(N).fill(0.);
    arma::mat v(N,N);
    for(int i=0 ; i<=N-1; i++){
            lambda(i)= d + 2*a*cos( (i+1)*PI/(N + 1));
            for(int k=0; k<=N-1; k++){
                v(k,i)=sin( (k+1)*(i+1)*PI/(N+1));
            }
    }

    // normalize to 1 the eigenvectors
    v= arma::normalise(v);
    
    // 
    for(int i=0 ; i<=N-1; i++){
        std::cout << "******Eigenvalue #"<< i+1 << std::endl << lambda(i) << "    vs  " << eigval(i) << "\n";
        if( std::abs(lambda(i) - eigval(i))<0.000001*std::abs(lambda(i)) )
            std::cout << "The " << i+1 <<"-th eigenvalues are the same" << std::endl;
        std::cout << "\nAnalytical    Numerical\n";
        for(int k=0; k<=N-1; k++){
            std::cout << v(k,i) << "  " << eigvec(k,i) << std::endl;
            if( std::abs(v(k,i) - eigvec(k,i))<0.000001*std::abs(v(k,i)) ){
                tf(k)=1;
            }else if(std::abs(v(k,i) + eigvec(k,i))<0.000001*std::abs(v(k,i))){
                tf(k)=-1;
            }
        }
        if( arma::all(tf == 1) ||  arma::all(tf == -1)){
            std::cout << "The " << i+1 <<"-th analytical eigenvector is equal to the one obtained with eig_sym" << "\n\n";
        }
    }

    // all is good
    return 0;
}