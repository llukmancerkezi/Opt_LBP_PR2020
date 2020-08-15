#include"Alg.h"
#include"Hist2D.h"
#include "StatisticalMeasure.h"
#include "LBPFeature.h"
#include "Optimization.h"
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


Alg::Alg(Mat img, int typeLBP_1, int modeLBP, int n, int r)
{
	// Naive LBP implementation. 
	Hist2D hist;
	LBPFeature lbp1(img, r, n);
	int maxVal;
	switch (modeLBP)
	{
	case 0:
		if (n > 12)
		{
			cout << "Algorithm will not run for values of n greater than 12." << endl;
			break;
		}

		lbpMatrix_1 = lbp1.LBPDecimalMatrix(typeLBP_1);
		maxVal = lbp1.maxValDecimal;

		featVecConc = hist.CalculateOneDHistogram(lbpMatrix_1, maxVal);
		createdFeatConc = true;
		dimFeatVecConc = hist.dimConcatenateHist;



		break;
	case 1:

		lbpMatrix_1 = lbp1.LBPUniformMatrix(typeLBP_1);
		maxVal = lbp1.maxValUniform;

		featVecConc = hist.CalculateOneDHistogram(lbpMatrix_1, maxVal);
		createdFeatConc = true;
		dimFeatVecConc = hist.dimConcatenateHist;

		break;
	case 2:

		lbpMatrix_1 = lbp1.LBPRiuMatrix(typeLBP_1);
		maxVal = lbp1.maxValRiu;

		featVecConc = hist.CalculateOneDHistogram(lbpMatrix_1, maxVal);
		createdFeatConc = true;
		dimFeatVecConc = hist.dimConcatenateHist;

		break;
	}

}

Alg::Alg(Mat img, int typeLBP_1, int typeLBP_2, int modeLBP, int n, int r)
{
	// Naive 2-D LBP implementation. 

	Hist2D hist;

	LBPFeature lbp1(img, r, n);
	LBPFeature lbp2(img, r, n);
	int maxVal_1, maxVal_2;
	switch (modeLBP)
	{
	case 0:

		if (n > 8)
		{
			cout << "Algorithm will not run for values of n greater than 8." << endl;
			break;
		}

		lbpMatrix_1 = lbp1.LBPDecimalMatrix(typeLBP_1);
		lbpMatrix_2 = lbp2.LBPDecimalMatrix(typeLBP_2);

		maxVal_1 = lbp1.maxValDecimal;
		maxVal_2 = lbp2.maxValDecimal;

		joint2DHist = hist.CalculateJoint2DHistogram(lbpMatrix_1, lbpMatrix_2, maxVal_1, maxVal_2);

		featVecConc = hist.CalculateConcatenateHistogram(joint2DHist);
		createdFeatConc = true;

		featVecFlatten = hist.CalculateFlattenHistogram(joint2DHist);
		createdFeatFlatten = true;

		dimFeatVecConc = hist.dimConcatenateHist;
		dimFeatVecFlatten = hist.dimFlattenHist;

		break;

	case 1:
		if (n > 16) {
			cout << "Algorithm will not run for values of n greater than 16." << endl;
			break;
		}

		lbpMatrix_1 = lbp1.LBPUniformMatrix(typeLBP_1);
		lbpMatrix_2 = lbp2.LBPUniformMatrix(typeLBP_2);
		maxVal_1 = lbp1.maxValUniform;
		maxVal_2 = lbp2.maxValUniform;
		//cout << maxVal_1 << "\t\t" << maxVal_2 << endl;

		joint2DHist = hist.CalculateJoint2DHistogram(lbpMatrix_1, lbpMatrix_2, maxVal_1, maxVal_2);

		featVecConc = hist.CalculateConcatenateHistogram(joint2DHist);
		createdFeatConc = true;

		featVecFlatten = hist.CalculateFlattenHistogram(joint2DHist);
		createdFeatFlatten = true;

		dimFeatVecConc = hist.dimConcatenateHist;
		dimFeatVecFlatten = hist.dimFlattenHist;

		break;
	case 2:

		lbpMatrix_1 = lbp1.LBPRiuMatrix(typeLBP_1);
		lbpMatrix_2 = lbp2.LBPRiuMatrix(typeLBP_2);
		maxVal_1 = lbp1.maxValRiu;
		maxVal_2 = lbp2.maxValRiu;
		
		joint2DHist = hist.CalculateJoint2DHistogram(lbpMatrix_1, lbpMatrix_2, maxVal_1, maxVal_2);

		featVecConc = hist.CalculateConcatenateHistogram(joint2DHist);
		createdFeatConc = true;

		featVecFlatten = hist.CalculateFlattenHistogram(joint2DHist);
		createdFeatFlatten = true;

		dimFeatVecConc = hist.dimConcatenateHist;
		dimFeatVecFlatten = hist.dimFlattenHist;

		break;
	}

}

Alg::Alg(Mat img, int typeLBP_1, int typeLBP_2, int n, int r, int optConstraint, int optType, int angle_step)
{
	Hist2D hist;

	LBPFeature lbp1(img, r, n);
	LBPFeature lbp2(img, r, n);

	lbpMatrix_1 = lbp1.LBPRiuMatrix(typeLBP_1);
	lbpMatrix_2 = lbp2.LBPRiuMatrix(typeLBP_2);
	int maxValRiu_1 = lbp1.maxValRiu;
	int maxValRiu_2 = lbp2.maxValRiu;


	joint2DHist = hist.CalculateJoint2DHistogram(lbpMatrix_1, lbpMatrix_2, maxValRiu_1, maxValRiu_2);

	Optimization optimizer(joint2DHist, optConstraint, angle_step, optType);

	optimizationAngle = optimizer.GetOptAngle();
	optJoint2DHist = optimizer.GetOptimizedJoint2DHist();
	optSearchAngles = optimizer.GetSearchAngles();
	statisticValAngles = optimizer.GetStatisticsAngles();

	optFeatVecConc = hist.CalculateConcatenateHistogram(optJoint2DHist);
	createdOptFeatConc = true;

	optFeatVecFlatten = hist.CalculateFlattenHistogram(optJoint2DHist);
	createdOptFeatFlatten = true;

	dimOptFeatVecConc = hist.dimConcatenateHist;
	dimOptFeatVecFlatten = hist.dimFlattenHist;

}

Alg::~Alg()
{
	if (!calledFeatConc && createdFeatConc) 
		delete[] featVecConc;
	
	if (!calledFeatFlatten && createdFeatFlatten) 
		delete[] featVecFlatten;
	
	if (!calledOptFeatConc && createdOptFeatConc) 
		delete[] optFeatVecConc;
	
	if (!calledOptFeatFlatten && createdOptFeatFlatten) 
		delete[] optFeatVecFlatten;
	
	if (!calledRotAngles)
		delete[] optSearchAngles;
	
	if (!calledStatisticsArray) 
		delete[] statisticValAngles;
	
}

int* Alg::GetRotationAngles()
{
	calledRotAngles = true;
	return optSearchAngles;
}

double* Alg::GetStatitscisValues()
{
	calledStatisticsArray = true;
	return  statisticValAngles;

}

double* Alg::GetFeatVecConc() {
	calledFeatConc = true;
	return  featVecConc;
}

double* Alg::GetFeatVecFlatten() {
	calledFeatFlatten = true;
	return featVecFlatten;
}

double* Alg::GetOptFeatVecConc() {
	calledOptFeatConc = true;
	return optFeatVecConc;

}
double* Alg::GetOptFeatVecFlatten() {
	calledOptFeatFlatten = true;
	return optFeatVecFlatten;
}

int Alg::GetDimOptFeatVecConc() {
	return dimOptFeatVecConc;
}
int Alg::GetDimOptFeatVecFlatten() {
	return dimOptFeatVecFlatten;
}

int Alg::GetDimFeatVecConc() {
	return dimFeatVecConc;
}
int Alg::GetDimFeatVecFlatten() {
	return dimFeatVecFlatten;
}

int Alg::GetRotAngleOpt() {
	return optimizationAngle;
}
