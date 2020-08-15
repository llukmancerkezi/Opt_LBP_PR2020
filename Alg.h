#ifndef Alg_H
#define Alg_H

#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


class Alg
{
private:

	//StatisticalMeasure statistics;

	double* optFeatVecConc;
	double* optFeatVecFlatten;
	double* featVecConc;
	double* featVecFlatten;

	bool calledFeatConc = false;
	bool calledFeatFlatten = false;
	bool calledOptFeatConc = false;
	bool calledOptFeatFlatten = false;
	bool calledRotAngles = false;
	bool calledStatisticsArray = false;

	bool createdFeatConc = false;
	bool createdFeatFlatten = false;
	bool createdOptFeatConc = false;
	bool createdOptFeatFlatten = false;


	int dimOptFeatVecConc;
	int dimOptFeatVecFlatten;
	int dimFeatVecConc;
	int dimFeatVecFlatten;

	Mat joint2DHist;
	Mat optJoint2DHist;
	Mat lbpMatrix_1, lbpMatrix_2;

	int optimizationAngle;
	double optVal;
	int* optSearchAngles;
	double* statisticValAngles;

public:


	Alg(Mat img, int typeLBP_1, int modeLBP, int n, int r);
	Alg(Mat img, int typeLBP_1, int typeLBP_2, int modeLBP, int n, int r);
	Alg(Mat img, int typeLBP_1, int typeLBP_2, int n, int r, int optConstraint, int optType, int angle_step);
	~Alg();


	double* GetFeatVecConc();
	double* GetFeatVecFlatten();

	double* GetOptFeatVecConc();
	double* GetOptFeatVecFlatten();

	int* GetRotationAngles();
	double* GetStatitscisValues();

	int GetDimOptFeatVecConc();
	int GetDimOptFeatVecFlatten();
	int GetDimFeatVecConc();
	int GetDimFeatVecFlatten();

	int GetRotAngleOpt();

};
#endif