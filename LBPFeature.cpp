#include "LBPFeature.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define pi 3.1415926

using namespace cv;
using namespace std;


inline int LBPFeature::BilInter(Mat image, double x, double y)
{
	//A  A1 B
	//   E  D1 
	//C  C1 D

	//cout << " corresponding coordinates:" << endl << endl;
	//cout << "x: " << x << "     " << "y:  " << y << endl << endl;

	int x0 = floor(x), x1 = ceil(x);
	int y0 = floor(y), y1 = ceil(y);

	double xDrift = x - floor(x);
	double yDrift = y - floor(y);

	int A = image.at<uchar>(x0, y0);
	int B = image.at<uchar>(x1, y0);
	int C = image.at<uchar>(x0, y1);
	int D = image.at<uchar>(x1, y1);
	//cout << A << "    " << B << endl;
	//cout << C << "    " << D << endl;

	//int A = image.data[y0 * width + x0];
	//int B = image.data[y0 * width + x1];
	//int C = image.data[y1 * width + x0];
	//int D = image.data[y1 * width + x1];

	double A1 = A + ((B - A) * xDrift);
	double C1 = C + ((D - C) * xDrift);
	double E1 = A1 + ((C1 - A1) * yDrift);
	//cout << endl << round(E1) << endl;

	return (int)round(E1);
}

int    LBPFeature::CheckUniformity(int* vector) {
	// u - number of uniformity
	int u = 0, v = 0;
	for (int i = 0; i < n; i++)
	{

		(i == n - 1) ? (v = vector[i] - vector[0]) : (v = vector[i + 1] - vector[i]);

		//if (i == n - 1)
		//	v = vector[i] - vector[0];
		//else
		//	v = vector[i + 1] - vector[i];

		if (v != 0) u = u + 1;
	}
	return u;      //return the degree of uniformity
}
int    LBPFeature::CalculateRiuVal(int* vector, int uniformVal) {
	//Calculating riu value of lbp binary code
	int  sum = 0;
	if (uniformVal <= 2) {
		for (int m = 0; m < n; m++)
			sum += vector[m];
	}
	else
		sum = n + 1;
	return sum;
}

int    LBPFeature::ConvertBinToDec(int* binArr) {
	//Converting binary array to decimal value
	float decimalVal = 0.0;
	for (int k = 0; k < n; k++) {
		decimalVal += binArr[k] * pow(2, k);
	}
	return decimalVal;
}
int	   LBPFeature::CalculateUniformVal(int* vector) {

	int pos_transition = 0; // position where 0 to 1 happen
	int sum = 0, uniformVal = 0;
	for (int i = 0; i < n - 1; i++) {
		if (vector[i] == 0 && vector[i + 1] == 1) {
			pos_transition = i;
			break;
		}
	}
	for (int i = 0; i < n; i++) sum += vector[i];

	if (sum == 0) {
		uniformVal = 0;
	}
	else {
		uniformVal = sum * (sum - 1) + pos_transition + 1;
	}
	//for (int i = 0; i < n; i++) cout << vector[i] << "\t";
	//cout << "\t\t" << uniformVal;
	//cout << endl;

	return uniformVal;
}

void   LBPFeature::LBPSign(Mat image, int* vector, int x, int y) {
	//Write Comment
	int pixVal;
	int interValue;
	for (int i = 0; i < n; i++)
	{
		pixVal = image.at<uchar>(x, y);
		interValue = BilInter(image, x + xPs[i], y + yPs[i]);



		vector[i] = (interValue - pixVal >= 0) ? 1 : 0;

		//(interValue - pixVal >= 0) ? (vector[i] = 1) : (vector[i] = 0);

		//if (interValue - pixVal >= 0)
		//	vector[i] = 1;
		//else
		//	vector[i] = 0;
	}
}
void   LBPFeature::LBPSignGlobalTh(Mat image, int* vector, int x, int y, int threshold)
{
	int pixVal;
	int interValue;
	for (int i = 0; i < n; i++)
	{
		interValue = BilInter(image, x + xPs[i], y + yPs[i]);

		(interValue - threshold >= 0) ? (vector[i] = 1) : (vector[i] = 0);

		//if (interValue - threshold >= 0)
		//	vector[i] = 1;
		//else
		//	vector[i] = 0;
	}
}

void   LBPFeature::LBPMean(Mat image, int* vector, int* vector1, int x, int y) {
	//Write Comment,
	int    pix_val;
	double inter_value;
	for (int i = 0; i < n; i++)
	{
		inter_value = BilInter(image, x + xPs[i], y + yPs[i]);
		pix_val = image.at<uchar>(x, y);
		vector1[i] = abs(inter_value - pix_val);
	}


	double sum = 0, count = 0;
	for (int i = 0; i < n; i++)
		sum = sum + vector1[i];

	double mean_poz = 0;
	mean_poz = sum / n;

	for (int i = 0; i < n; i++)
		vector[i] = (vector1[i] - mean_poz > 0) ? 1 : 0;

}
void   LBPFeature::ELBP_AD(Mat image, int* vector, int* vector1, int x, int y)
{
	int    pix_val;
	double inter_value_1, inter_value_2;
	for (int i = 0; i < n; i++)
	{
		if (i < i - 1)
		{
			inter_value_1 = BilInter(image, x + xPs[i], y + yPs[i]);
			inter_value_2 = BilInter(image, x + xPs[i + 1], y + yPs[i + 1]);
			vector1[i] = abs(inter_value_1 - inter_value_2);
		}
		else
		{
			inter_value_1 = BilInter(image, x + xPs[i], y + yPs[i]);
			inter_value_2 = BilInter(image, x + xPs[0], y + yPs[0]);
			vector1[i] = abs(inter_value_1 - inter_value_2);
		}
	}
	double sum = 0, count = 0;
	for (int i = 0; i < n; i++)
		sum = sum + vector1[i];

	double threshold = 0;
	threshold = sum / n;

	for (int i = 0; i < n; i++)
		vector[i] = (vector1[i] - threshold > 0) ? 1 : 0;

}



int	   LBPFeature::MeanPixelImg(Mat img)
{
	int threshold = 0;
	int rowOrgImage = img.rows, colOrgImage = img.cols;
	for (int i = radius; i < rowOrgImage - radius; i++)
	{
		for (int j = radius; j < colOrgImage - radius; j++) threshold += img.at<uchar>(i, j);
	}
	threshold = threshold / (riuImage.cols * riuImage.rows);
	return threshold;
}


LBPFeature::LBPFeature(Mat Image, int _radius, int _neighbors)
{
	orgImage = Image;
	n = _neighbors;
	radius = _radius;

	int height = Image.rows;
	int width = Image.cols;

	xPs = new float[n];
	yPs = new float[n];

	for (int i = 0; i < n; i++)
	{
		xPs[i] = (round((radius * sin(2 * pi * i / n)) * 100)) / 100;
		yPs[i] = (round((radius * cos(2 * pi * i / n)) * 100)) / 100;
	}

	decimalImage = Mat(height - 2 * (radius), width - 2 * (radius), CV_32FC1, Scalar(0));
	riuImage = Mat(height - 2 * (radius), width - 2 * (radius), CV_32FC1, Scalar(0));
	uniformImage = Mat(height - 2 * (radius), width - 2 * (radius), CV_32FC1, Scalar(0));

	//mmaximum value in riuImage image, minimum value is zero
	maxValRiu = n + 2;
	//maximum value in uniformImage image, minimum value is zero
	maxValUniform = n * (n - 1) + 3;

	//maximum value in decimalImage image, minimum value is zero
	maxValDecimal = pow(2, n);
}

Mat LBPFeature::LBPRiuMatrix(int type)
{
	int rowOrgImage = orgImage.rows, colOrgImage = orgImage.cols;

	int lbpVal = 0;
	int numUniform = 0;
	int* zeroVec = new int[n]();
	int* zeroVec1 = new int[n]();
	int threshold;
	Mat medianImage;
	switch (type)
	{
	case 0: //Sign
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(orgImage, zeroVec, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;

	case 1: //Mean
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				LBPMean(orgImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;

	case 2://ELBP_AD
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				ELBP_AD(orgImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;

			}
		}
		break;

	case 3://Sign-GlobalThreshold

		threshold = MeanPixelImg(orgImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSignGlobalTh(orgImage, zeroVec, i, j, threshold);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;


	case 4://MRELBP-SIGN

		//cout << "height and width: " << orgImage.cols << "\t\t" << orgImage.rows << endl;

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(medianImage, zeroVec, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;

	case 5://MRELBP-MEAN

		//cout << "height and width: " << orgImage.cols << "\t\t" << orgImage.rows << endl;

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				LBPMean(medianImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;

	case 6://MRELBP-ELBP-AD

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				ELBP_AD(medianImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;

			}
		}
		break;

	case 7://MRELBP-Sign-GlobalThreshold

		medianBlur(orgImage, medianImage, 3);
		threshold = MeanPixelImg(medianImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSignGlobalTh(medianImage, zeroVec, i, j, threshold);
				numUniform = CheckUniformity(zeroVec);
				lbpVal = CalculateRiuVal(zeroVec, numUniform);
				riuImage.at<float>(i - radius, j - radius) = lbpVal;
			}
		}
		break;

	}

	delete[] zeroVec;
	delete[] zeroVec1;
	return riuImage;
}

Mat LBPFeature::LBPUniformMatrix(int type) {

	int rowOrgImage = orgImage.rows, colOrgImage = orgImage.cols;

	int uniformValue = 0;
	int numUniform = 0;
	int* zeroVec = new int[n]();
	int* zeroVec1 = new int[n]();
	int threshold;
	Mat medianImage;
	switch (type)
	{
	case 0: //Sign
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(orgImage, zeroVec, i, j);
				numUniform = CheckUniformity(zeroVec);


				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				//if (numUniform <=2) {
				//	
				//	uniformValue = CalculateUniformVal(zeroVec);
				//}
				//else {
				//	
				//	uniformValue = maxValUniform-1;
				//}
				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}
		break;

	case 1: //Mean
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				LBPMean(orgImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);

				//(n > 10) ? (m = 100) : (m = 50);
				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);


				//if (numUniform <= 2) {
				//	uniformValue = CalculateUniformVal(zeroVec);
				//}
				//else {
				//	uniformValue = maxValUniform-1;	
				//}
				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}
		break;

	case 2://ELBP_AD
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				ELBP_AD(orgImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);

				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				uniformImage.at<float>(i - radius, j - radius) = uniformValue;

			}
		}
		break;

	case 3://Sign-GlobalThreshold

		threshold = MeanPixelImg(orgImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));

				LBPSignGlobalTh(orgImage, zeroVec, i, j, threshold);
				numUniform = CheckUniformity(zeroVec);

				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}

	case 4: //MRELBP-Sign

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(medianImage, zeroVec, i, j);
				numUniform = CheckUniformity(zeroVec);


				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				//if (numUniform <=2) {
				//	
				//	uniformValue = CalculateUniformVal(zeroVec);
				//}
				//else {
				//	
				//	uniformValue = maxValUniform-1;
				//}
				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}
		break;

	case 5: //MRELBP-Mean

		medianBlur(orgImage, medianImage, 3);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				LBPMean(medianImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);

				//(n > 10) ? (m = 100) : (m = 50);
				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);


				//if (numUniform <= 2) {
				//	uniformValue = CalculateUniformVal(zeroVec);
				//}
				//else {
				//	uniformValue = maxValUniform-1;	
				//}
				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}
		break;

	case 6://MRELBP-ELBP_AD
		medianBlur(orgImage, medianImage, 3);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				memset(zeroVec1, 0, n * sizeof(int));

				ELBP_AD(medianImage, zeroVec, zeroVec1, i, j);
				numUniform = CheckUniformity(zeroVec);

				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				uniformImage.at<float>(i - radius, j - radius) = uniformValue;

			}
		}
		break;

	case 7://MRELBP-Sign-GlobalThreshold

		medianBlur(orgImage, medianImage, 3);
		threshold = MeanPixelImg(medianImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));

				LBPSignGlobalTh(medianImage, zeroVec, i, j, threshold);
				numUniform = CheckUniformity(zeroVec);

				(numUniform <= 2) ? (uniformValue = CalculateUniformVal(zeroVec)) : (uniformValue = maxValUniform - 1);

				uniformImage.at<float>(i - radius, j - radius) = uniformValue;
			}
		}
	}

	delete[] zeroVec;
	delete[] zeroVec1;

	return uniformImage;
}

Mat LBPFeature::LBPDecimalMatrix(int type)
{
	int rowOrgImage = orgImage.rows, colOrgImage = orgImage.cols;
	int decValue;
	int* zeroVec = new int[n]();
	int* zeroVec1 = new int[n]();
	int threshold;
	Mat medianImage;
	switch (type)
	{
	case 0://Sign
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(orgImage, zeroVec, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;
	case 1://Mean
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPMean(orgImage, zeroVec, zeroVec1, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;

	case 2://ELBP_AD
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				ELBP_AD(orgImage, zeroVec, zeroVec1, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;

	case 3: //Sign-GlobalThreshold
		threshold = MeanPixelImg(orgImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSignGlobalTh(orgImage, zeroVec, i, j, threshold);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}

	case 4://MRELBP-Sign

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSign(medianImage, zeroVec, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;
	case 5://MRELBP-Mean

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPMean(medianImage, zeroVec, zeroVec1, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;

	case 6://MRELBP-ELBP_AD

		medianBlur(orgImage, medianImage, 3);

		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				ELBP_AD(medianImage, zeroVec, zeroVec1, i, j);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}
		break;

	case 7: //MRELBP-Sign-GlobalThreshold

		medianBlur(orgImage, medianImage, 3);

		threshold = MeanPixelImg(medianImage);
		for (int i = radius; i < rowOrgImage - radius; i++)
		{
			for (int j = radius; j < colOrgImage - radius; j++)
			{
				memset(zeroVec, 0, n * sizeof(int));
				LBPSignGlobalTh(medianImage, zeroVec, i, j, threshold);
				decValue = ConvertBinToDec(zeroVec);
				decimalImage.at<float>(i - radius, j - radius) = decValue;
			}
		}

	}

	delete[] zeroVec;
	delete[] zeroVec1;
	return decimalImage;
}


