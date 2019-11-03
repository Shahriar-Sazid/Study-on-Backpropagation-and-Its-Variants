// Copyright (c) 2013, Manuel Blum
// All rights reserved.

// Define this symbol to enable runtime tests for allocations
//#define EIGEN_RUNTIME_NO_MALLOC

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <ctime>
#include "nn.h"

using namespace std;

inline void swap(int &val)
{
	val = (val << 24) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) | (val >> 24);
}

matrix_t read_mnist_images(std::string filename)
{

	matrix_t X;
	std::ifstream fs(filename.c_str(), std::ios::binary);
	if (fs)
	{
		int magic_number, num_images, num_rows, num_columns;
		fs.read((char *)&magic_number, sizeof(magic_number));
		fs.read((char *)&num_images, sizeof(num_images));
		fs.read((char *)&num_rows, sizeof(num_rows));
		fs.read((char *)&num_columns, sizeof(num_columns));
		if (magic_number != 2051)
		{
			swap(magic_number);
			swap(num_images);
			swap(num_rows);
			swap(num_columns);
		}

		X = matrix_t::Zero(num_images, num_rows * num_columns);

		for (size_t i = 0; i < num_images; ++i)
		{
			for (size_t j = 0; j < num_rows * num_columns; ++j)
			{
				unsigned char temp = 0;
				fs.read((char *)&temp, sizeof(temp));
				X(i, j) = (double)temp;
			}
		}
		fs.close();
	}
	else
	{
		std::cout << "error reading file: " << filename << std::endl;
		exit(1);
	}
	return X;
}

matrix_t read_mnist_labels(std::string filename)
{
	matrix_t Y;
	std::ifstream fs(filename.c_str(), std::ios::binary);
	if (fs)
	{
		int magic_number, num_images, num_rows, num_columns;
		fs.read((char *)&magic_number, sizeof(magic_number));
		fs.read((char *)&num_images, sizeof(num_images));
		if (magic_number != 2049)
		{
			swap(magic_number);
			swap(num_images);
		}

		Y = matrix_t::Zero(num_images, 10);

		for (size_t i = 0; i < num_images; ++i)
		{
			unsigned char temp = 0;
			fs.read((char *)&temp, sizeof(temp));
			Y(i, (int)temp) = 1.0;
		}
		fs.close();
	}
	else
	{
		std::cout << "error reading file: " << filename << std::endl;
		exit(1);
	}
	return Y;
}

double measure_accuracy(matrix_t Y, matrix_t Y_test)
{
	int correct = 0;
	for (int i = 0; i < Y.rows(); i++)
	{
		int j = 0;
		for (; j < Y.cols(); j++)
		{
			int out = Y_test(i, j) > 0.7 ? 1 : 0;
			//printf("%3i   %4.4f", (int)Y(i, j), Y_test(i, j));

			if (out != Y(i, j))
				break;
		}
		//cout << endl;
		if (j == Y.cols())
		{
			correct++;
		}
		else
		{
			cout << Y_test.row(i) << endl;
		}
	}
	//cout << "Correct answer " << correct << " among " << Y.rows() << " test" << endl;
	double accuracy = (double)correct / (double)Y.rows() * 100;
	return accuracy;
	//cout << "Accuracy: " <<  accuracy << "%" << endl;
}

int main(int argc, const char *argv[])
{
	
	if (argc != 2)
	{
		std::cout << "please provide path to mnist data ..." << std::endl;
		std::cout << "you can download the dataset at http://yann.lecun.com/exdb/mnist/" << std::endl;
		std::cout << std::endl
				  << "usage: " << argv[0] << " path_to_data" << std::endl
				  << std::endl;
		return 1;
	}

	std::string path = argv[1];

	std::cout << "reading data" << std::endl;

	matrix_t X_train = read_mnist_images(path + "/train-images.idx3-ubyte");
	matrix_t Y_train = read_mnist_labels(path + "/train-labels.idx1-ubyte");
	matrix_t X_test = read_mnist_images(path + "/t10k-images.idx3-ubyte");
	matrix_t Y_test = read_mnist_labels(path + "/t10k-labels.idx1-ubyte");
	// number of optimization steps 
	int max_steps = 5;
	// regularization parameter
	double lambda = 0.0;

	// specify network topology
	Eigen::VectorXi topo(3);
	topo << X_train.cols(), 300, Y_test.cols();
	std::cout << "topology: " << topo.transpose() << std::endl;

	// initialize a neural network with given topology
	std::cout << "initializing network" << std::endl;
	NeuralNet nn(topo);

	std::cout << "scaling the data" << std::endl;
	nn.autoscale(X_train, Y_train);

	//std::cout << Y_train << std::endl;
	// train the network
	std::cout << "starting training" << std::endl;
	std::cout << "iter        error" << std::endl;
	double err;
	clock_t begin = clock();
	for (int i = 0; i < max_steps; ++i)
	{
		err = nn.loss(X_train, Y_train, lambda);
		nn.rprop();
		printf("%4i   %10.7f\n", i, err);
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Training time: " << elapsed_secs << endl;

	// test accuracy
	nn.forward_pass(X_test);
	matrix_t prediction = nn.get_activation();
	int correct = 0;
	int k;
	for (size_t i = 0; i < Y_test.rows(); ++i)
	{
		prediction.row(i).maxCoeff(&k);
		correct += Y_test(i, k);
	}

	std::cout << "test accuracy: " << correct * 1.0 / Y_test.rows()*100 << "% " << std::endl;
	//cout << prediction << endl;
	// cout << "*****************Printing the wrong output***************" << endl; 
	// double accuracy = measure_accuracy(Y_test, prediction);
	// std::cout << "Accuracy: " << accuracy << std::endl;
	nn.write("mnist.nn");
	

	return 0;
}
