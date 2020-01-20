//
// Created by vloods on 1/19/20.
//

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include <typeinfo>
#include <experimental/filesystem>
#include <vector>
#include <cstdio>
#include <stdint.h>
#include <omp.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>

#include <string>
#include <memory>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "RPForest.h"
#include "common.h"


using namespace Eigen;

struct RPFparams {
	int n;
	int ntest;
	int k;
	int n_trees;
	int depth;
	int dim;
	int votes;
	float density;
	bool parallel;
	bool verbose;
	std::string dsfolder;
	int n_points;
};

void read_config(RPFparams &pars) {
	std::ifstream cFile("config.txt");
	if (cFile.is_open())
	{
		std::string line;
		while (getline(cFile, line)) {
			line.erase(std::remove_if(line.begin(), line.end(),
				[](unsigned char x) { return isspace(x); }),
				line.end());
			if (line[0] == '#' || line.empty())
				continue;
			else break;
		}
		pars.dsfolder = line.substr(line.find("=") + 1);
		getline(cFile, line);
		pars.n = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.ntest = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.dim = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.k = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.n_trees = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.depth = std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.density = (float)std::atof(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.parallel = (bool)std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.verbose = (bool)std::atoi(line.substr(line.find("=") + 1).c_str());
		getline(cFile, line);
		pars.votes = std::atoi(line.substr(line.find("=") + 1).c_str());
		if (!pars.dsfolder.empty() && pars.dsfolder.back() != '/' && pars.dsfolder.back() != '\\')
			pars.dsfolder += '/';
		pars.n_points = pars.n - pars.ntest;
	}
	else {
		std::cerr << "Couldn't open config file for reading.\n";
	}
}


int main() {
	RPFparams p;
	std::cout << "---------------------------------------------" << std::endl;
	std::cout << "Random Projection Forest by Vladislav Tishin." << std::endl;
	std::cout << "---------------------------------------------" << std::endl << std::endl;
	read_config(p);

	//get data
	float *train, *test;

	test = read_memory((p.dsfolder + "test.bin").c_str(), p.ntest, p.dim);
	if (!test) {
		std::cerr << "In file " << __FILE__ << ", line " << __LINE__ << ": test data " << p.dsfolder + "test.bin" << " could not be read\n";
		return -1;
	}

	train = read_memory((p.dsfolder + "train.bin").c_str(), p.n_points, p.dim);
	if (!train) {
		std::cerr << "In file " << __FILE__ << ", line " << __LINE__ << ": training data " << p.dsfolder + "train.bin" << " could not be read\n";
		return -1;
	}

	//Building forest
	if (!p.parallel) omp_set_num_threads(1);


	std::cout << "Growing forest..." << std::endl;
	double build_start = omp_get_wtime();
	RPForest index_dense(train, p.dim, p.n_points);
	index_dense.grow(p.n_trees, p.depth);
	double build_time = omp_get_wtime() - build_start;
	std::cout << "Growing Time: " << build_time << std::endl << std::endl;


	//Testing
	std::cout << "Testing...";
	float total_acc = 0.0f;
	double total_rpt = 0.0;
	double total_bft = 0.0;
	for (int i = 0; i < p.ntest; ++i) {
		std::vector<int> result(p.k);
		const Map<const VectorXf> q(&test[i * p.dim], p.dim);
		const Map<const MatrixXf> X(train, p.dim, p.n_points);

		VectorXi indices_exact(p.k);
		double start = omp_get_wtime();
		RPForest::exact_knn(q, X, p.k, indices_exact.data());
		double end = omp_get_wtime();
		double def_time = end - start;
		total_bft += def_time;

		start = omp_get_wtime();
		index_dense.query(q, p.k, p.votes, &result[0]);
		end = omp_get_wtime();
		double time = end - start;
		total_rpt += time;

		float accuracy = 0.0;
		for (int i = 0; i < p.k; ++i) {
			accuracy += (result[i] == indices_exact(i));
		}
		accuracy /= p.k;
		total_acc += accuracy;

		if (p.verbose) {
			std::cout << indices_exact.transpose() << std::endl;
			for (int &y : result) {
				std::cout << y << " ";
			}
			std::cout << std::endl;
			std::cout << "Accuracy: " << setprecision(3) << accuracy << '\t' << "RPTime: " << time << '\t'
				<< "BFTime: " << def_time << std::endl << std::endl;
		}
	}
	std::cout << "Results:" << std::endl;
	std::cout << " - Accuracy: " << total_acc / p.ntest << "\n - Average RPTime: " << total_rpt / p.ntest << "\n - F: " << total_bft / total_rpt << "\n";

	delete[] test;
	delete[] train;
	system("pause");
	return 0;
}
