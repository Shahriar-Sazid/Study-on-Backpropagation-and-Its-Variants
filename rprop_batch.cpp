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

typedef vector<int> vi;
typedef vector<vi> vvi;
inline void swap(int &val)
{
    val = (val << 24) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) | (val >> 24);
}
class batch_manager
{
    int num_samples;
    int current_count;
    int batch_size;

  public:
    batch_manager(int num_samples, int batch_size)
    {
        this->num_samples = num_samples;
        this->batch_size = batch_size;
        current_count = 0;
    }
    pair<int, int> next_batch()
    {
        pair<int, int> res = make_pair(current_count, current_count + batch_size);
        if (current_count + batch_size > num_samples)
        {
            //cout << "Current count: " << current_count << endl;
            if (current_count == num_samples)
            {
                current_count = 0;
                return make_pair(0, batch_size);
            }
            res.second = num_samples;
            current_count = 0;

            return res;
        }
        current_count += res.second - res.first;
        return res;
    }
};
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
            //cout << Y_test.row(i) << endl;
        }
    }
    //cout << "Correct answer " << correct << " among " << Y.rows() << " test" << endl;
    double accuracy = (double)correct / (double)Y.rows() * 100;
    return accuracy;
    //cout << "Accuracy: " <<  accuracy << "%" << endl;
}

void fill_type_info(matrix_t &Y, vi &type_info)
{
    int num_sample = Y.rows();
    int num_type = Y.cols();
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_type; j++)
        {
            if (Y(i, j) == 1)
            {
                type_info[j]++;
                break;
            }
        }
    }
}

void fill_position(matrix_t &Y, vvi &position)
{
    int num_sample = Y.rows();
    int num_type = Y.cols();
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_type; j++)
        {
            if (Y(i, j) == 1)
            {
                position[j].push_back(i);
                break;
            }
        }
    }
}

int distribution_in_batch(vi &type_distribution, vi &type_info, int proposed_batch_size, int num_sample)
{
    int batch_size = 0;
    int num_type = type_info.size();
    for (int i = 0; i < num_type; i++)
    {
        type_distribution[i] = (int)ceil((double)(type_info[i] * proposed_batch_size) / (double)num_sample);
        batch_size += type_distribution[i];
    }
    return batch_size;
}

void distribute_into_batch(matrix_t &X, matrix_t &Y, matrix_t &Xm, matrix_t &Ym, vi &type_distribution, vvi &position, int batch_size, vi &type_point, int batch_no)
{
    int cnt = 0;
    for (int i = 0; i < type_distribution.size(); i++)
    {
        for (int j = 0; j < type_distribution[i]; j++)
        {
            if (batch_no * batch_size + cnt == Xm.rows())
            {
                Xm.conservativeResize(Xm.rows() + batch_size - cnt, Xm.cols());
                Ym.conservativeResize(Ym.rows() + batch_size - cnt, Ym.cols());
            }
            Xm.row(batch_no * batch_size + cnt) = X.row(position[i][type_point[i]]);
            Ym.row(batch_no * batch_size + cnt) = Y.row(position[i][type_point[i]]);
            cnt++;
            type_point[i]++;
            if (type_point[i] >= position[i].size())
            {
                type_point[i] = 0;
            }
        }
    }
}
void distribute_into_matrix(matrix_t &X, matrix_t &Y, matrix_t &Xm, matrix_t &Ym, vi &type_info, vi &type_distribution, vvi &position, int batch_size)
{
    vi type_point(type_info.size(), 0);
    int num_batch = (int)ceil((double)Y.rows() / (double)batch_size);

    for (int i = 0; i < num_batch; i++)
    {
        distribute_into_batch(X, Y, Xm, Ym, type_distribution, position, batch_size, type_point, i);
    }
}
pair<matrix_t, matrix_t> make_uniform_dataset(matrix_t &X, matrix_t &Y, int &batch_size)
{
    int num_sample = Y.rows();
    int num_type = Y.cols();

    vi type_info(num_type, 0);
    fill_type_info(Y, type_info);

    vi type_distribution(num_type, 0);
    batch_size = distribution_in_batch(type_distribution, type_info, batch_size, num_sample);

    vvi position(num_type);
    fill_position(Y, position);

    matrix_t Xm(num_sample, X.cols());
    matrix_t Ym(num_sample, num_type);

    distribute_into_matrix(X, Y, Xm, Ym, type_info, type_distribution, position, batch_size);

    return make_pair(Xm, Ym);
}
pair<matrix_t, matrix_t> make_worst_batch(matrix_t &X, matrix_t &Y)
{
    matrix_t Xm(X.rows(), X.cols());
    matrix_t Ym(Y.rows(), Y.cols());

    int num_sample = Y.rows();
    int num_type = Y.cols();

    vi type_info(num_type, 0);
    fill_type_info(Y, type_info);

    vvi position(num_type);
    fill_position(Y, position);
    int cnt = 0;
    for (int i = 0; i < num_type; i++)
    {
        for (int j = 0; j < position[i].size(); j++)
        {
            Xm.row(cnt) = X.row(position[i][j]);
            Ym.row(cnt) = Y.row(position[i][j]);
            cnt++;
        }
    }

    return make_pair(Xm, Ym);
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

    // matrix_t X_train = read_mnist_images(path + "/train-images.idx3-ubyte");
    // matrix_t Y_train = read_mnist_labels(path + "/train-labels.idx1-ubyte");
    // matrix_t X_test = read_mnist_images(path + "/t10k-images.idx3-ubyte");
    // matrix_t Y_test = read_mnist_labels(path + "/t10k-labels.idx1-ubyte");

    matrix_t X_train = read_mnist_images(path + "/train-images.idx3-ubyte");
    matrix_t Y_train = read_mnist_labels(path + "/train-labels.idx1-ubyte");
    matrix_t X_test = read_mnist_images(path + "/t10k-images.idx3-ubyte");
    matrix_t Y_test = read_mnist_labels(path + "/t10k-labels.idx1-ubyte");

    //cout << "Number of training sample: " << X_train.rows() << endl;
    int max_steps = 1500;
    double lambda = 0.001;

    // specify network topology
    Eigen::VectorXi topo(3);
    topo << X_train.cols(), 300, Y_test.cols();
    std::cout << "topology: " << topo.transpose() << std::endl;

    // initialize a neural network with given topology
    std::cout << "initializing network" << std::endl;
    NeuralNet nn(topo);

    int batch_size = 100;
    // pair<matrix_t, matrix_t> xy = make_uniform_dataset(X_train, Y_train, batch_size);

    // X_train = xy.first;
    // Y_train = xy.second;

    // pair<matrix_t, matrix_t> xy = make_worst_batch(X_train, Y_train);

    // X_train = xy.first;
    // Y_train = xy.second;

    std::cout<< "scaling the data" << std::endl;
    nn.autoscale(X_train, Y_train);

    int num_attribute = X_train.cols();
    int num_type = Y_train.cols();
    batch_manager batch(X_train.rows(), batch_size);

    std::cout << "starting training" << std::endl;
    std::cout << "iter        error" << std::endl;

    double err;
    clock_t begin = clock();
    for (int i = 0; i < max_steps; ++i)
    {
        pair<int, int> start_end = batch.next_batch();
        const int batch_size = start_end.second - start_end.first;

        matrix_t Xm = X_train.block(start_end.first, 0, batch_size, num_attribute);
        matrix_t Ym = Y_train.block(start_end.first, 0, batch_size, num_type);

        err = nn.loss(Xm, Ym, lambda);
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

    std::cout << "test accuracy: " << correct * 1.0 / Y_test.rows() * 100 << "%" << std::endl;
    // double accuracy = measure_accuracy(Y_test, prediction);
    // std::cout << "Accuracy: " << accuracy << std::endl;
    nn.write("mnist.nn");

    return 0;
}
