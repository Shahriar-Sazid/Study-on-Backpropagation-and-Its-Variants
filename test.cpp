#include <Eigen/Dense>
#include <iostream>
using namespace std;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
int main()
{
  matrix_t m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  cout << m << endl;
  m.conservativeResize(10, 4);
  cout << m << endl;
  m.row(6) = m.row(2);
  cout << endl; 
  cout << m<<endl;
  return 0;
}