#ifndef NNClassifier_H
#define NNClassifier_H

#include <iostream>
using namespace std;

class NNClassifier
{
private:

	int dimFeature;
	int nbSampleTrain;
	int nbSampleTest;
	int* prediction;
	double accuracy;

	double ChiSquareDist(double* x, double* y);
	int ArgMin(double* dist);

	double CalculateAccuracy(int* pred, int* train_labels, int* test_labels);

public:
	NNClassifier(double** train_data, double** test_data, int* train_labels, int* test_labels, int _dim_feature, int _nb_sample_train, int _nb_sample_test);

	double GetAccuracy();

};
#endif