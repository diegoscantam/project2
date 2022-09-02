#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h> 
#include "useful.hpp"

int main(){
    int n=100; //number of steps
    double x_min=0;
    double x_max=1;
    int width=16;
    int prec=4;
    double h = (x_max - x_min) / n;
    std::cout <<"h ="<< h << "\n";

    //Initialize x and v with a length 
    std::vector<double> x(n); 
    std::vector<double> u(n)  ;

    // Set a filename 
    std::string filename = "u.txt";
    // Create and open the output file. Or, technically, create 
    // an "output file stream" (type std::ofstream) and connect it to our filename.
    std::ofstream ofile;
    ofile.open(filename);

    // fill x and u vectors, (print them), and store them in a file "u.txt"
    x[0]=x_min;
    for(int i = 0 ; i <= n; i++){
        x[i] = x_min +  i*h;
        u[i] =(double) 1 + ( exp(-10*x[i]) -1 )*x[i] - exp(-10*x[i]);
        //std::cout << i << scientific_format(x[i], width, prec ) << " , " << scientific_format(u[i], width, prec )<<  "\n";
        ofile << scientific_format(x[i], width, prec ) << scientific_format(u[i], width, prec )<<  std::endl;

    }

    // Close the output file
    ofile.close();

    return 0;
}