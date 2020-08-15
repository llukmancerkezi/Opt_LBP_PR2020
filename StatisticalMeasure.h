#ifndef StatisticalMeasure_H
#define StatisticalMeasure_H

#include<iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class StatisticalMeasure
{
private:
	double *marginProbX = NULL, *marginProbY = NULL, *marginProbXY = NULL;
	double *flattenHist = NULL;
	int size;
	Mat join2DHist;
public:
	

	double MeanVal(double *vec,int dim);
	
	double Variance(double *vec, int dim);

	double Entropy(double *vec, int dim);

	double JointEntropy(Mat joint2DHist);

	double MutualInform(Mat joint2DHist, double *arr_1, double *arr_2, int dim);

	double Correlation(double* arr_1, double* arr_2, int dim);

};

#endif