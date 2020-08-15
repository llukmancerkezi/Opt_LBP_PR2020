#include"Hist2D.h"
#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <math.h>
//#include <vector>
using namespace std;
using namespace cv;


Mat Hist2D::CalculateJoint2DHistogram(Mat lbpMat_1, Mat lbpMat_2, int _maxValLbp_1, int _maxValLbp_2) {

	maxValLbp_1 = _maxValLbp_1;
	maxValLbp_2 = _maxValLbp_2;
	Mat joint2DHistogram = Mat(_maxValLbp_1, _maxValLbp_2, CV_32FC1, Scalar(0));

	int u, v;
	for (int i = 0; i < lbpMat_1.rows; i++)
	{
		for (int j = 0; j < lbpMat_1.cols; j++)
		{
			u = lbpMat_1.at<float>(i, j);
			v = lbpMat_2.at<float>(i, j);
			joint2DHistogram.at<float>(u, v)++;      // crossHist.at<float>(u, v) + 1
		}
	}
	return joint2DHistogram;
}

double* Hist2D::CalculateOneDHistogram(Mat lbpMat, int _maxValLbp_1)
{
	maxValLbp_1 = _maxValLbp_1;
	oneDHistogram = new double[maxValLbp_1]();
	dimConcatenateHist = _maxValLbp_1;
	int lbpVal;
	for (int i = 0; i < lbpMat.rows; i++) {
		for (int j = 0; j < lbpMat.cols; j++) {
			lbpVal = (int)lbpMat.at<float>(i, j);
			oneDHistogram[lbpVal] += 1;
		}
	}
	return oneDHistogram;
	
}

double* Hist2D::CalculateFlattenHistogram(Mat joint2DH) {
	
	int lbpVal;
	int row, col;
	row = joint2DH.rows;
	col = joint2DH.cols;
	dimFlattenHist = row * col;
	oneDFlattenHistogram = new double[dimFlattenHist]();

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			lbpVal = (int)joint2DH.at<float>(i, j);
			oneDFlattenHistogram[i*col + j] += lbpVal;
		}
	}
	return oneDFlattenHistogram;
}

double* Hist2D::CalculateXHistogram(Mat joint2DH) {

	int lbpVal;
	int row, col;
	row = joint2DH.rows;
	col = joint2DH.cols;
	dimXHist = row;
	oneDXHistogram = new double[dimXHist]();

	for (int i = 0; i < row; i++) {
		int sumX = 0;
		for (int j = 0; j < col; j++) {
			sumX += (int)joint2DH.at<float>(i, j);
		}
		oneDXHistogram[i] = sumX;
	}

	return oneDXHistogram;
}

double* Hist2D::CalculateYHistogram(Mat joint2DH) {

	int lbpVal;
	int row, col;
	row = joint2DH.rows;
	col = joint2DH.cols;
	dimYHist = col;
	oneDYHistogram = new double[dimYHist]();


	for (int i = 0; i < col; i++) {
		int sumY = 0;
		for (int j = 0; j < row; j++) 
			sumY += (int)joint2DH.at<float>(j, i);
		
		oneDYHistogram[i] = sumY;
	}
	return oneDYHistogram;
}

double* Hist2D::CalculateConcatenateHistogram(Mat joint2DH) {
	double*histX, *histY;
	histX = CalculateXHistogram(joint2DH);
	histY = CalculateYHistogram(joint2DH);
	dimConcatenateHist = joint2DH.rows + joint2DH.cols;

	oneDConcatenateHistogram = new double[dimConcatenateHist]();
	for (int i = 0; i < dimXHist; i++) 
		oneDConcatenateHistogram[i] = histX[i];
	
	for (int i = 0; i < dimYHist; i++) 
		oneDConcatenateHistogram[i+ dimXHist] = histY[i];

	delete[] histX;
	delete[] histY;
	return oneDConcatenateHistogram;
}

Mat Hist2D::PadJoint2DHistogram(Mat hist, int pad) {
	Mat joint2DHist;
	copyMakeBorder(hist, joint2DHist, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
	return joint2DHist;
}

