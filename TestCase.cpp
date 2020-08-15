#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Alg.h"


using namespace std;
using namespace cv;

//You can extract feature vectors for different LBP types
enum LBP_TYPE { Sign, Mean, ELBP_A, SignGlobalTh, MRE_Sign, MRE_Mean, MRE_ELBP_A, MRE_SignGlobalTh };
enum LBP_MODE { decimal, uniform, riu };

// Constraints to be optimized - see CASE 3 Test
enum CONSTRAINT { MutualInformation, Entropy, Correlation, JointEntropy, Variance };
enum MIN_MAX { maxMetric, minMetric };

int main(int argc, char** argv)
{
	//reading the image
	Mat image = imread("SampleImage.jpg", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	int dimF, dimC;
	int radius=1;
	int neighbour = 8;

	//***************************** TEST CASE 1 - GET TRADITIONAL LBP FEATURE *****************************

	Alg AlgLBP(image, Sign, riu, neighbour, radius);

	//LBP Features
	double* featVec;
	//Get Feature Vector
	featVec = AlgLBP.GetFeatVecConc();
	// Get dimenstion of the feature vector
	dimC = AlgLBP.GetDimFeatVecConc();

	delete[] featVec;

	//***************************** TEST CASE 2 - GET 2D TRADITIONAL FEATURES *****************************

	radius = 1; // Radius of LBP
	neighbour = 8; // Number of neighbours in LBP

	Alg Alg2D(image, Sign, Mean, uniform, neighbour, radius);

	double* featVecM;
	double* featVecF;

	//Get Marginal Feature Vector
	featVecM = Alg2D.GetFeatVecConc();
	//Get dimension of the feature vector
	dimC = Alg2D.GetDimFeatVecConc();

	//Get Flatten Feature Vector
	featVecF = Alg2D.GetFeatVecFlatten();
	//Get dimension of the feature vector
	dimF = Alg2D.GetDimFeatVecFlatten();

	delete[] featVecF;
	delete[] featVecM;

	//***************************** TEST CASE 3 - GET OPTIMIZED FEATURES *****************************

	radius = 1; // Radius of LBP
	neighbour = 8; // Number of neighbours in LBP

	Alg Alg2DLBPOpt(image, Sign, Mean, neighbour, radius, MutualInformation, maxMetric, 1);

	double* featVecOptM;
	double* featVecOptF;

	//Get Optimized Marginal Feature Vector
	featVecOptM = Alg2D.GetOptFeatVecConc();
	//Get dimension of the feature vector
	dimC = Alg2D.GetDimOptFeatVecConc();

	//Get Optimized Flatten Feature Vector
	featVecOptF = Alg2D.GetOptFeatVecFlatten();
	//Get dimension of the feature vector
	dimF = Alg2D.GetDimOptFeatVecFlatten();

	delete[] featVecOptM;
	delete[] featVecOptF;

	//***********************************************END***************************************************

	cout<<"\n\nEND!\n\n";
	system("PAUSE");
	return 0;
}


