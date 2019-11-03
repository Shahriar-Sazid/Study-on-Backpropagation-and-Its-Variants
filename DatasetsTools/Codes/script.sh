#!/bin/bash
g++ classification_data.cpp -o class
./class ../Datasets/ahealth_older2.csv 4
./class ../Datasets/cancer12.csv 2
./class ../Datasets/cancer22.csv 2
./class ../Datasets/cancer32.csv 2
./class ../Datasets/eeg_eye_state2.csv 2
./class ../Datasets/ExampleDataSet2.csv 2
./class ../Datasets/iris.csv 3
./class ../Datasets/magic2.csv 2
./class ../Datasets/sonar2.csv 2
./class ../Datasets/wine2.csv 2