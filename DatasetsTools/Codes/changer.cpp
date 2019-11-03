#include <bits/stdc++.h>


using namespace std;


int main(){
    fstream fin, fout;
    fin.open("output.txt");
    fout.open("output2.csv");
    string str;
    while(fin >> str)
    {
        str = str.substr(0, str.size()-1);
        fout << str << endl; 
    }
    fout.close();
    fin.close();
    return 0;
}