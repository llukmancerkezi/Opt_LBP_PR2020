#include "StatisticalMeasure.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#define pi 3.1415926


using namespace std;
using namespace cv;

double StatisticalMeasure::MeanVal(double *vec, int dim)
{
	double sum = 0;
	for (int i = 0; i < dim; i++) 
		sum += vec[i];
		
	sum = sum / dim;
	return sum;
}

double StatisticalMeasure::Variance(double *vec,int dim)
{
	double sum = 0;
	double mean = MeanVal(vec, dim);
	for (int i = 0; i < dim; i++)
		sum += (vec[i] - mean)*(vec[i] - mean);
	
	sum = sum / (dim - 1);
	return sum;
}

double StatisticalMeasure::Entropy(double *vec, int dim)
{
	double *vec1 = NULL;
	vec1 = new double[dim];
	for (int i = 0; i < dim; i++)
		vec1[i] = vec[i];
	
	double sum = 0;
	for (int i = 0; i < dim; i++)
		sum += vec1[i];

	for (int i = 0; i < dim; i++)
		vec1[i] = vec1[i] / sum;

	double entropy = 0;
	for (int i = 0; i < dim; i++)
	{
		if (vec1[i] != 0)
			entropy += (-vec1[i] * log2(vec1[i]));
	}

	delete[] vec1;
	return entropy;
}


double StatisticalMeasure::JointEntropy(Mat joint2DHist)
{
	double sum = 0; double sumX = 0; double sumY = 0;

	for (int i = 0; i < joint2DHist.rows; i++)
		for (int j = 0; j < joint2DHist.cols; j++) sum += joint2DHist.at<float>(i, j);

	for (int i = 0; i < joint2DHist.rows; i++)
		for (int j = 0; j < joint2DHist.cols; j++)  joint2DHist.at<float>(i, j) = joint2DHist.at<float>(i, j) / sum;

	
	double JointEntr = 0;

	for (int i = 0; i < size; i++){
		for (int j = 0; j <size; j++){
			if (joint2DHist.at<float>(i, j) != 0)
				JointEntr +=  (-joint2DHist.at<float>(i, j) * log2(joint2DHist.at<float>(i, j)));
		}
	}

	return JointEntr;
}

double StatisticalMeasure::MutualInform(Mat joint2DHist, double* arr_1, double* arr_2, int dim)
{
	double sum = 0; double sumX = 0; double sumY = 0;
	
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) sum += joint2DHist.at<float>(i, j);
		
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)  joint2DHist.at<float>(i, j)= joint2DHist.at<float>(i, j)/ sum;
	

	for (int i = 0; i < dim; i++){
		sumX += arr_1[i];
		sumY += arr_2[i];
	}


	for (int i = 0; i < dim; i++){
		arr_1[i] = arr_1[i] / sumX;
		arr_2[i] = arr_2[i] / sumY;
	}


	double mutualInform = 0, temp = 0, temp1=0;
	double joint_val = 0;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			joint_val = joint2DHist.at<float>(i, j);
			if (arr_1[i] != 0 && arr_2[j] != 0 && joint_val != 0)
			{
				temp = joint_val / (arr_1[i] * arr_2[j]);
				temp1 = log2(temp);
				mutualInform += (joint_val *temp1);
			}
		}
	}

	return mutualInform;
}

double StatisticalMeasure::Correlation(double* arr_1, double* arr_2, int dim)
{
	
	double meanX, meanY;
	double varX, varY;
	double sum = 0;
	
	meanX = MeanVal(arr_1, dim);
	meanY = MeanVal(arr_2, dim);

	varX = Variance(arr_1, dim);
	varY = Variance(arr_2, dim);

	for (int i = 0; i < dim; i++)
		sum += ((arr_1[i] - meanX)* (arr_2[i] - meanY));  // covariance 

	double corr;
	corr = sum / (sqrt(varX)*sqrt(varY)*(dim - 1));
	
	return corr;
}

