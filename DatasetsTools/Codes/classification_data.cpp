#include <bits/stdc++.h>

using namespace std;

int main(int argc, char *argv[])
{
    fstream fin, fout;
    fout.open("../Links/dad.txt");
    string filename(argv[1]);
    fin.open(filename);

    int n_type = atoi(argv[2]);
    vector<int> class_cnt(n_type, 0);
    vector<double> class_prcnt(n_type);
    vector<string> types(n_type);
    string str;
    
    for (int i = 0; i < n_type; i++)
    {
        str += "0,";
    }
    str = str.substr(0, str.size() - 1);
    
    
    for (int i = 0; i < n_type; i++)
    {
        string str1 = str;
        str1[i * 2] = '1';
        types[i] = str1;
    }
    
    
    
    string line;
    int match_len = 2 * n_type - 1;
    int lines = 0;
    
    
    while (fin >> line)
    {
        string match = line.substr(line.size() - match_len, match_len);
        for (int i = 0; i < n_type; i++)
        {
            if (types[i] == match)
            {
                class_cnt[i]++;
                break;
            }
        }
        lines++;
    }
    
    fin.close();
    cout << "######## Dataset: " << filename.substr(12, filename.size()-12) << " ###########" << endl;
    for(int i = 0; i<n_type; i++){
        class_prcnt[i] = (double)class_cnt[i]/(double)lines*100;
        cout << "Class " << i+1 << ": " << class_cnt[i];
        cout << "  Percentage: " << class_prcnt[i] << "%"<< endl;
    }
    cout << endl;
    fout.close();
    return 0;
}