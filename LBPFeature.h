#ifndef LBPFeature_H
#define LBPFeature_H

#include<iostream>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class LBPFeature
{
private:

	int n, radius;
	int width, height;
	float* xPs, * yPs;
	double mp;

	inline int BilInter(Mat image, double a, double b);
	int    CheckUniformity(int* vector);		//A is LBP vector of pixel, neighbors is length of that vector
	int    CalculateRiuVal(int* vector, int uniformVal);     //A is LBP vector of pixel,u is uniformity degree,neighbors is length of that vector
	void   LBPSign(Mat image, int* vector, int x, int y);
	void   LBPMean(Mat image, int* vector, int* vectors1, int x, int y);
	void   ELBP_AD(Mat image, int* vector, int* vector1, int x, int y);
	void   LBPSignGlobalTh(Mat image, int* vector, int x, int y, int threshold);
	int	   ConvertBinToDec(int* binArr);
	int	   CalculateUniformVal(int* vector);
	int	   MeanPixelImg(Mat img);

public:
	LBPFeature(Mat image, int radius, int neighbors);

	Mat riuImage;
	Mat decimalImage;
	Mat uniformImage;
	Mat orgImage;

	int maxValRiu;
	int maxValUniform;
	int maxValDecimal;
	
	Mat LBPRiuMatrix(int type);
	Mat LBPDecimalMatrix(int type);
	Mat LBPUniformMatrix(int type);
};

#endif