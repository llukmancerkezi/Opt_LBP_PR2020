#include"Optimization.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

Optimization::Optimization(Mat _joint2DHist, int typeStatistic, int angleStep, int maxMinOptimization)
{

	joint2DHist = _joint2DHist;
	pad_size = (joint2DHist.rows / 2 + 1) / 2;
	paddedJoint2DHist = hist.PadJoint2DHistogram(joint2DHist, pad_size);
	
	numStepAngle = (int)90/ angleStep + 1;
	statisticValArr = new double[numStepAngle];
	rotationAngles = new int[numStepAngle];

	int row_paddedH = paddedJoint2DHist.rows, col_paddedH = paddedJoint2DHist.cols;

	for (int angle = 0, index = 0; angle <= 90; angle += angleStep, index+=1)
	{
		Point2f pt(row_paddedH / 2., col_paddedH / 2.);
		Mat rotation_matrix = getRotationMatrix2D(pt, angle, 1.0);
		warpAffine(paddedJoint2DHist, rotatedJoint2DHist, rotation_matrix, Size(row_paddedH, CV_INTER_NN));
		switch (typeStatistic)
		{
		case 0:
			//Mutual information
			histX = hist.CalculateXHistogram(rotatedJoint2DHist);
			histY = hist.CalculateYHistogram(rotatedJoint2DHist);
			statisticVal = statistics.MutualInform(rotatedJoint2DHist, histX, histY, hist.dimXHist);
			
			statisticValArr[index] = statisticVal;
			rotationAngles[index] = angle;

			delete[] histX;
			delete[] histY;
			histX = NULL;
			histY = NULL;
			
			break;

		case 1:
			//Entropy

			histC = hist.CalculateConcatenateHistogram(rotatedJoint2DHist);
			statisticVal = statistics.Entropy(histC, hist.dimConcatenateHist);

			statisticValArr[index] = statisticVal;
			rotationAngles[index] = angle;

			delete[] histC;
			histC = NULL;
			
			break;
		
		case 2:
			//Correlation

			histX = hist.CalculateXHistogram(rotatedJoint2DHist);
			histY = hist.CalculateYHistogram(rotatedJoint2DHist);
			statisticVal = statistics.Correlation(histX, histY, hist.dimXHist);

			statisticValArr[index] = statisticVal;
			rotationAngles[index] = angle;

			delete[] histX;
			delete[] histY;
			histX = NULL;
			histY = NULL;

			break;

		case 3:
			//JointEntropy

			statisticVal = statistics.JointEntropy(rotatedJoint2DHist);
			
			statisticValArr[index] = statisticVal;
			rotationAngles[index] = angle;

		case 4:
			//Variance

			histC = hist.CalculateConcatenateHistogram(rotatedJoint2DHist);
			statisticVal = statistics.Variance(histC, hist.dimConcatenateHist);

			statisticValArr[index] = statisticVal;
			rotationAngles[index] = angle;

			delete[] histC;
			histC = NULL;
			break;
		}
	}

	switch (maxMinOptimization)
	{
	case 0:
		MaxOptimization();
		break;
	case 1:
		MinOptimization();
		break;

	}

}

void Optimization::MaxOptimization()
{
	optimized_angle = rotationAngles[0];
	optimized_val = statisticValArr[0];
	for (int i = 1; i < numStepAngle; i++) {
		if (statisticValArr[i] > optimized_val) {
			optimized_val = statisticValArr[i];
			optimized_angle = rotationAngles[i];
		}
	}
}

void Optimization::MinOptimization()
{
	optimized_angle = rotationAngles[0];
	optimized_val = statisticValArr[0];
	for (int i = 1; i < numStepAngle; i++) {
		if (statisticValArr[i] < optimized_val) {
			optimized_val = statisticValArr[i];
			optimized_angle = rotationAngles[i];
		}
	}
}

Mat Optimization::GetOptimizedJoint2DHist() {

	Mat optimizedJoint2DH;
	int row_padded = paddedJoint2DHist.rows, col_padded = paddedJoint2DHist.cols;
	Point2f pt(row_padded / 2., col_padded / 2.);
	Mat rotation_matrix = getRotationMatrix2D(pt, optimized_angle, 1.0);
	warpAffine(paddedJoint2DHist, optimizedJoint2DH, rotation_matrix, Size(row_padded, CV_INTER_NN));
	return optimizedJoint2DH;
	
}

int Optimization::GetOptAngle() {
	
	return optimized_angle;
}

int* Optimization::GetSearchAngles() {
	called_search_angles = true;
	return rotationAngles;
}

double* Optimization::GetStatisticsAngles() {
	called_statistics_angles = true;
	return statisticValArr;
}

Optimization::~Optimization() {

	if (flatten)
		delete[] flatten;
	if (histX)
		delete[] histX;
	if (histY)
		delete[] histY;
	if (histC)
		delete[] histC;

	if (!called_search_angles)
		delete[] statisticValArr;
	if (!called_search_angles)
		delete[] rotationAngles;

}