#include "NNClassifier.h"
#include <iostream>
#define pi 3.1415926

//using namespace cv;
using namespace std;

NNClassifier::NNClassifier(double** trainData, double** testData, int* trainLabels, int* testLabels, int _dimFeature, int _nbSampleTrain, int _nbSampleTest)
{
	dimFeature = _dimFeature;
	nbSampleTrain = _nbSampleTrain;
	nbSampleTest = _nbSampleTest;

	prediction = new int[_nbSampleTest];
	for (int i = 0; i < nbSampleTest; i++)
	{
		double* distVec = new double[nbSampleTrain];
		for (int j = 0; j < nbSampleTrain; j++)
			distVec[j] = ChiSquareDist(testData[i], trainData[j]);
		
		prediction[i] = ArgMin(distVec);
		delete[] distVec;
	}
	accuracy = CalculateAccuracy(prediction, trainLabels, testLabels);
	delete[] prediction;
}

double NNClassifier::CalculateAccuracy(int* pred, int* trainLabels, int* testLabels)
{
	double _accuracy = 0;
	int pred_label = 0;
	double nb_correct=0;
	for (int i = 0; i < nbSampleTest; i++)
	{
		pred_label = trainLabels[pred[i]];
		if (pred_label == testLabels[i])
			nb_correct++;
			
	}
	_accuracy = nb_correct / nbSampleTest;
	return _accuracy;
}

double NNClassifier::ChiSquareDist(double* x, double* y) {
	double dist = 0;
	for (int i = 0; i < dimFeature; i++)
	{
		if (x[i] != 0 || y[i] != 0) 
			dist += (x[i] - y[i]) * (x[i] - y[i]) / (x[i] + y[i]);
	}
	return dist;
}

int NNClassifier::ArgMin(double* dist)
{
	int argminIndex = 0;
	double minDist = dist[0];
	for (int i = 1; i < nbSampleTrain; i++)
	{
		if (dist[i] < minDist)
		{
			argminIndex = i;
			minDist = dist[i];
		}
	}
	return argminIndex;
}

double NNClassifier::GetAccuracy()
{
	return accuracy;
}