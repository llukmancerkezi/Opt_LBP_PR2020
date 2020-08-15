#ifndef Hist2D_H
#define Hist2D_H

#include<iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


class Hist2D
{
private:
	int maxValLbp_1, maxValLbp_2;
	
	double* oneDHistogram;
	double* oneDXHistogram;
	double* oneDYHistogram;
	double* oneDConcatenateHistogram;
	double* oneDFlattenHistogram;
	

public:



	Mat CalculateJoint2DHistogram(Mat lbpMat_1, Mat lbpMat_2, int _maxValLbp_1, int _maxValLbp_2);
	double* CalculateFlattenHistogram(Mat joint2DH);
	double* CalculateXHistogram(Mat joint2DH);
	double* CalculateYHistogram(Mat joint2DH);
	double* CalculateConcatenateHistogram(Mat joint2DH);

	double* CalculateOneDHistogram(Mat lbpMat, int _maxValLbp_1);

	
	int dimHist, dimXHist, dimYHist;
	int dimConcatenateHist,dimFlattenHist;

	Mat PadJoint2DHistogram(Mat hist, int pad);

};



#endif // !Hist2D_H
