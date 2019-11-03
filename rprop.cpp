#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <string>
#include "nn.h"
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

int main(int argc, const char *argv[])
{
	string filename(argv[1]);
	ifstream fin;
	string directory("../Datasets/");
	string filepath = directory + filename;
	fin.open(filepath);

	string str;
	int n_input = 0;  // input dimensionality
	int n_output = 0; // output dimensionality
	int n_sample = 0; // number of training samples

	while (fin >> str)
	{
		n_sample++;
	}

	stringstream sst(str);
	double each_input;
	while (sst >> each_input)
	{
		n_input++;
		if (sst.peek() == ',')
			sst.ignore();
	}
	n_output = atoi(argv[2]);
	n_input = n_input - n_output;
	//cout << n_input << "  " << n_output;
	
	int n_layer = 3;		  // number of layers
	int max_steps = 5;	  // number of optimization steps
	int n_epoch = max_steps;
	double lambda = 0.000001; // regularization parameter

	matrix_t X(n_sample, n_input);
	matrix_t Y(n_sample, n_output);

	fin.clear();
	fin.seekg(0, ios::beg);
	int row = 0;
	while (fin >> str)
	{
		int col = 0;
		double data;
		stringstream ss(str);
		while (ss >> data)
		{
			if (col < n_input)
			{
				X(row, col) = data;
			}
			else
			{
				int pos = col - n_input;
				Y(row, pos) = data;
			}
			if (ss.peek() == ',')
			{
				ss.ignore();
			}
			col++;
		}
		row++;
	}
	fin.close();

	// cout << "training input: " << endl
	// 	 << X << endl;
	// cout << "training output: " << endl
	// 	 << Y << endl;

	// specify network topology
	Eigen::VectorXi topo(n_layer);
	topo << n_input, n_input, n_output;
	// cout << "topology: " << endl
	// 	 << topo << endl;

	// initialize a neural network with given topology
	NeuralNet nn(topo);

	nn.autoscale(X, Y);
	// train the network
	std::cout << "starting training" << endl;
	double err;
	double prevAccuracy = 0;
	double accuracy;
	for (int i = 0; i < max_steps; ++i)
	{
		err = nn.loss(X, Y, lambda);
		nn.rprop();
		//nn.forward_pass(X);
		matrix_t Y_prime = nn.get_activation();
		accuracy = nn.measure_accuracy(Y, Y_prime);
		//cout << "Accuracys: " << accuracy << "  Step No: " << i<<endl;
		cout << err << endl;
		if (accuracy - prevAccuracy < .001 && accuracy>90)
		{
			n_epoch = i;
			break;
		}
		else
		{
			prevAccuracy = accuracy;
		}
		//printf("%3i   %4.4f\n", i, err);
	}

	// write model to disk
	nn.write("example.nn");

	// read model from disk
	NeuralNet nn2("example.nn");

	// testing
	nn2.forward_pass(X);
	matrix_t Y_test = nn2.get_activation();

	//std::cout << "test input:" << endl
	//  << X << endl;
	//std::cout << "test output:" << endl
	//  << Y_test << endl;
	int correct = 0;
	for (int i = 0; i < n_sample; i++)
	{
		int j = 0;
		for (; j < n_output; j++)
		{
			int out = Y_test(i, j) > 0.7 ? 1 : 0;
			//printf("%3i   %4.4f", (int)Y(i, j), Y_test(i, j));
			if (out != Y(i, j))
				break;
		}
		//cout << endl;
		if (j == n_output)
			correct++;
	}
	cout<< "Dataset: " << filename << endl;
	cout << "Correct answer " << correct << " among " << n_sample << " test" << endl;
	cout << "Accuracy: " << (double)correct / (double)n_sample * 100 << "%" << endl;
	cout << "Step Count: " << n_epoch << endl;
	return 0;
}
