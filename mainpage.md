# Documentation

RPForest is a lightweight and easy-to-use library for approximate nearest neighbor search. It is written in C++.


## Installation DemoApp

RPForest uses open source projects to run:

* [Eigen](eigen.tuxfamily.org) -  Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

Link to the repository: https://github.com/Vloods/RPForest  
Build requires: CMake 3.13+(Tested on 3.13), Visual Studio (Tested on VS 2019)

1. Install CMake, VS if necessary
2. Download the repository with the project and unpack it in a convenient place
3. Open the project directory in the CMake GUI
4. Configure the project and run it in Visual Studio
5. Build the RP Forest project
6. Put the `config.txt` file next to the finished application(be SURE to specify the path to the dataset in it, you can download the test dataset from the repository folder: demo_ds)
7. Launch the built application  

Dataset was taken from: http://corpus-texmex.irisa.fr  

### Config 
config.txt file example:
```
#Dataset parameters  

#dspath - Path to ds folder  
#n - Number of samples in dataset  
#ntest - Number of samples for test  
#dim - Sample dimensional  
#k - Nearest neighbour  
#n_trees - Number of trees  
#depth - Depth of each tree  
#density - Expected proportion of non-zero components in the random vectors(default=-1)  
#verbose - Print result for every test  

dspath=C:\dataset\path\  
n=10000  
ntest=1000  
dim=200  
k=10  
n_trees=89  
depth=5  
density=-1  
parallel=1  
verbose=0  
votes=3  
```
## Usage Example
RPForest is a header-only library, so no compilation is required: just include the header `RPForest.h`. The only dependency is the Eigen linear algebra library (Eigen 3.3.5 is bundled in cpp/lib).

Let's first generate a 200-dimensional data set of 10000 points, and a query point (row = dimension, column = data point). Then `RPForest::exact_knn` can be used to find the indices of the true 10 nearest neighbors of the test query.

You need to set the number(`n_trees`) and `depth` of trees, the threshold number of `votes` that trees must give to include a candidate in the selection of responses, and `K` nearest neighbors.

The `grow` function builds a forest for approximate k-nn search. 

The approximate nearest neighbors are then searched by the function `query`;
```
#include <iostream>
#include "Eigen/Dense"
#include "RPForest.h"

int main() {
     int n = 10000, d = 200, k = 10, n_trees = 89, depth = 5, votes = 3;

	Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
	Eigen::MatrixXf q = Eigen::VectorXf::Random(d);

	Eigen::VectorXi indices(k), indices_exact(k);

	RPForest::exact_knn(q, X, k, indices_exact.data());
	std::cout << indices_exact.transpose() << std::endl;

	RPForest rpf(X);
	rpf.grow(n_trees, depth);

	rpf.query(q, k, votes, indices.data());
	std::cout << indices.transpose() << std::endl;
}
```
Output:   
> 8061 8126 3772 1476 5090 7660 9921 3821 7336 2693  
> 8061 8126 3772 1476 5090 7660 9921 3821 7336 2693