#include <bits/stdc++.h>

using namespace std;

int main(int argc, char* argv[])
{
    string file = string(argv[1]);
    vector<string> ss;
    fstream f;
    f.open(file);
    string str;
    int line_cnt = 0;
    while (f >> str)
    {
        ss.push_back(str);
        line_cnt++;
    }
    random_shuffle(ss.begin(), ss.end());
    f.clear();
    f.seekg(0, ios::beg);
    while(line_cnt--){
        f << ss[line_cnt] << endl;
    }
    f.close();
    return 0;
}