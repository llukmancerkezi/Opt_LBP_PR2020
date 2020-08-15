#ifndef Optimization_H
#define Optimization_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include "StatisticalMeasure.h"
#include "Hist2D.h"

using namespace std;
using namespace cv;


class Optimization
{
private:
	Hist2D hist;
	StatisticalMeasure statistics;
	
	double* statisticValArr;
	int* rotationAngles;
	double* flatten, * histX, * histY, * histC;
	int pad_size;
	bool called_search_angles = false;
	bool called_statistics_angles = false;

public:

	Mat joint2DHist, paddedJoint2DHist,  rotatedJoint2DHist;
	
	double statisticVal;
	int numStepAngle;
	int optimized_angle;
	double optimized_val;
	


	Optimization(Mat _joint2DHist, int typeStatistic, int angleStep, int maxMinOptimization);
	~Optimization();

	void MaxOptimization();
	void MinOptimization();

	Mat GetOptimizedJoint2DHist();
	
	int GetOptAngle();
	double* GetStatisticsAngles();
	int* GetSearchAngles();

};

#endif