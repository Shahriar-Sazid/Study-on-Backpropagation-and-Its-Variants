#include <bits/stdc++.h>


using namespace std;

int main()
{
    fstream output_file;
    output_file.open("check.txt");
    fstream dataset;
    dataset.open("Datasets/eeg_eye_state.csv");

    string output;
    string expected;
    int correct = 0;
    while(output_file >> output)
    {
        dataset >> expected; 
        double out = stod(output);
        out = (out>0.5)?1:0;
        int expect = expected[expected.size()-1]=='0'?0:1;
        //cout << out << "  " << expect << endl;
        if((int)out==expect) correct++;
    }
    cout << "Correct answer: " << correct << " among 14980 samples" << endl;
    return 0;
}