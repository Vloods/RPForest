#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <utility>
#include <map>
#include <functional>

#include <cstdlib>

double mean(const std::vector<double> &x) {
	int n = x.size();
	double xsum = 0;
	for (int i = 0; i < n; ++i)
		xsum += x[i];
	return xsum / n;
}

double var(const std::vector<double> &x) {
	int n = x.size();
	double xmean = mean(x);
	double ssr = 0;
	for (int i = 0; i < n; ++i)
		ssr += (x[i] - xmean) * (x[i] - xmean);
	return ssr / (n - 1);
}

double median(std::vector<double> x) {
	int n = x.size();
	std::nth_element(x.begin(), x.begin() + n / 2, x.end());

	if (n % 2) {
		return x[n / 2];
	}

	double smaller = *std::max_element(x.begin(), x.begin() + n / 2);
	return (smaller + x[n / 2]) / 2.0;
}

std::vector<std::vector<int>> read_results(std::string truth, int k) {
	std::ifstream fs(truth);
	if (!fs) {
		std::cerr << "File " << truth << " could not be opened for reading!" << std::endl;
		exit(1);
	}

	double time;
	std::vector<std::vector<int>> correct;
	while (fs >> time) {
		std::vector<int> res;
		for (int i = 0; i < k; ++i) {
			int r;
			fs >> r;
			res.push_back(r);
		}
		correct.push_back(res);
	}
	return correct;
}



using namespace std;

int Ks[] = { 1, 10, 100, -1 };

float *get_data(const char *file, size_t dim, size_t *n) {
	struct stat sb;
	stat(file, &sb);
	size_t N = sb.st_size / (sizeof(float) * dim);
	*n = N;

	float *data = new float[N * dim];

	FILE *fd;
	fd = fopen(file, "rb");
	fread(data, sizeof(float), N * dim, fd);
	fclose(fd);

	return data;
}

float *read_memory(const char *file, size_t n, size_t dim) {
	float *data = new float[n * dim];

	struct stat sb;
	stat(file, &sb);

	if (sb.st_size != n * dim * sizeof(float)) {
		std::cerr << "Size of the file is " << sb.st_size << ", while the expected size is: " << n * dim * sizeof(float) << "\n";
		return NULL;
	}

	FILE *fd;
	if ((fd = fopen(file, "rb")) == NULL) {
		std::cerr << "Could not open file " << file << " for reading.\n";
		return NULL;
	}

	size_t read = fread(data, sizeof(float), n * dim, fd);
	if (read != n * dim) {
		std::cerr << "Expected size of the read was " << n * dim << ", but " << read << " was read.\n";
		return NULL;
	}

	fclose(fd);
	return data;
}

int *read_memory_int(const char *file, size_t n, size_t dim) {
	int *data = new int[n * dim];

	struct stat sb;
	stat(file, &sb);

	if (sb.st_size != n * dim * sizeof(int)) {
		std::cerr << "Size of the file is " << sb.st_size << ", while the expected size is: " << n * dim * sizeof(int) << "\n";
		return NULL;
	}

	FILE *fd;
	if ((fd = fopen(file, "rb")) == NULL) {
		std::cerr << "Could not open file " << file << " for reading.\n";
		return NULL;
	}

	size_t read = fread(data, sizeof(int), n * dim, fd);
	if (read != n * dim) {
		std::cerr << "Expected size of the read was " << n * dim << ", but " << read << " was read.\n";
		return NULL;
	}

	fclose(fd);
	return data;
}

template <typename T>
void write_memory(const T *mem, std::string spath, int nrow, int ncol) {
	const char *path = spath.c_str();
	FILE *fd;
	if ((fd = fopen(path, "wb")) == NULL) {
		std::cerr << "common.h: " << "file " << path
			<< " could not be opened for writing" << std::endl;
		return;
	}
	fwrite(mem, sizeof(T), nrow * ncol, fd);
	fclose(fd);
}




std::vector<int> read_parameters(const std::string &par_name, std::ifstream &inf) {
	std::string istr;
	std::vector<int> vs;

	while (inf) {
		getline(inf, istr);
		auto pos = istr.find("=");
		if (pos == std::string::npos) {
			continue;
		}
		std::string first(istr.substr(0, pos));
		if (first != par_name) {
			continue;
		}
		std::string second(istr.substr(pos, istr.size()));
		second.erase(std::remove(second.begin(), second.end(), '='), second.end());
		second.erase(std::remove(second.begin(), second.end(), '\"'), second.end());

		std::istringstream iss(second);
		int i = 0;
		while (iss >> i) {
			vs.push_back(i);
		}
		break;
	}
	return vs;
}

std::pair<std::vector<int>, std::vector<int>> map2vec(const std::map<int, int, std::greater<int>> &M) {
	std::vector<int> v1, v2;
	int acc = 0;
	for (const auto &m : M) {
		v1.push_back(m.first);
		v2.push_back(acc += m.second);
	}
	return std::make_pair(v1, v2);
}

int get_vote_threshold(int target_nn, const std::vector<int> &vote_thresholds,
	const std::vector<int> &nn_found) {
	if (vote_thresholds.size() != nn_found.size()) {
		throw std::logic_error("vote_thresholds.size and nn_found.size are different.");
	}

	int v = 1;
	for (int i = 0; i < nn_found.size(); ++i) {
		if (nn_found[i] >= target_nn) {
			v = vote_thresholds[i];
			break;
		}
	}
	return v;
}

int get_vote_threshold_probability(double target_nn, int k, const std::vector<int> &vote_thresholds,
	const std::vector<int> &inn_found, const std::vector<int> &top_votes) {
	if (vote_thresholds.size() != inn_found.size()) {
		throw std::logic_error("vote_thresholds.size and nn_found.size are different.");
	}

	std::vector<double> nn_found;
	for (int i = 0; i < vote_thresholds.size(); ++i)
		nn_found.push_back(inn_found[i] / static_cast<double>(k));

	std::random_device rd;
	std::mt19937 gen(rd());

	int v = 1;

	if (k == 1) {
		v = top_votes[0];
		std::bernoulli_distribution dist(target_nn);
		if (!vote_thresholds.empty() && dist(gen)) {
			v = vote_thresholds[0];
		}
		return v;
	}

	for (int i = 0; i < nn_found.size(); ++i) {
		if (nn_found[i] >= target_nn) {
			v = vote_thresholds[i];
			if (nn_found[i] > target_nn && i != 0) {
				double interval = nn_found[i] - nn_found[i - 1];
				double top = nn_found[i] - target_nn;
				double prob = top / interval;
				std::bernoulli_distribution dist(prob);
				if (dist(gen)) {
					v = vote_thresholds[i - 1];
				}
			}
			break;
		}
	}
	return v;
}