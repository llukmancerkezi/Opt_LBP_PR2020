#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Alg.h"
#include "NNClassifier.h"

using namespace std;
using namespace cv;

//Dataset-Image path
string dirDataset = "./Final resized images UIUC/";
//string dir_dataset = "C:/Users/llukm/Desktop/LBP/UIUC Dataset/Final resized images UIUC/";

string nameTrainImg = "./Final resized images UIUC/train_img.txt";
string nameTestImg = "./Final resized images UIUC/test_img.txt";
string labelTrainImg = "./Final resized images UIUC/train_label.txt";
string labelTestImg = "./Final resized images UIUC/test_label.txt";

int num_sample = 500;

int* ReadLabel(string file_name)
{
	int* labels = new int[num_sample]();
	int i = 0, label;

	ifstream inFile;
	inFile.open(file_name);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	while (inFile >> label)
	{
		labels[i] = label;
		i++;
	}

	return labels;
}
vector<string> ReadImgName(string file_name)
{
	vector<string> img_name_arr;
	string img_name;

	ifstream inFile;
	inFile.open(file_name);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	while (inFile >> img_name)
		img_name_arr.push_back(img_name);

	return img_name_arr;
}
void DeleteArrays(double** vec)
{
	for (int i = 0; i < 500; i++) delete[] vec[i];

	delete[] vec;
}


enum LBP_TYPE { Sign, Mean, ELBP_A, SignGlobalTh, MRE_Sign, MRE_Mean, MRE_ELBP_A, MRE_SignGlobalTh };
enum LBP_MODE { decimal, uniform, riu };



// Constraints to be optimized - see CASE 3 Test
enum CONSTRAINT { MutualInformation, Entropy, Correlation, JointEntropy, Variance };
enum MIN_MAX { maxMetric, minMetric };


int main(int argc, char** argv)
{
	cout << "TestCaseFeatureExtraction\n\n\n";
	Mat image = imread("SampleImage.jpg", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	int dimF, dimC;
	int radius=1;
	int neighbour = 8;


	//***************************** TEST CASE 1 - GET TRADITIONAL LBP FEATURE *****************************	

	//Extracting riu features

	Alg AlgLBP(image, Sign, riu, neighbour, radius);

	//LBP Features
	double* featVec;
	featVec = AlgLBP.GetFeatVecConc();
	dimC = AlgLBP.GetDimFeatVecConc();

	delete[] featVec;

	
	/*
		//***************************** TEST CASE 2 - GET 2D TRADITIONAL FEATURES *****************************


	radius = 1; // R of LBP
	neighbour = 8; // Number of neighbours in LBP

	Alg Alg2D(image, Sign, Mean, uniform, neighbour, radius);

	featTrainC_2[i] = Alg2D.GetFeatVecConc();
	dimC = Alg2D.GetDimFeatVecConc();

	//Extracting features for train set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTrain[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading train images

		Alg Alg2D(image, Sign, Mean, uniform, neighbour, radius);

		//Concatenated Features
		featTrainC_2[i] = Alg2D.GetFeatVecConc();
		dimC = Alg2D.GetDimFeatVecConc();

		// Flatten Features
		featTrainF_2[i] = Alg2D.GetFeatVecFlatten();
		dimF = Alg2D.GetDimFeatVecFlatten();
	}


	//Extracting features for test set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTest[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading test images

		Alg Alg2DLBP(image, Sign, Mean, uniform, neighbour, radius);

		//Concatenated Features
		featTestC_2[i] = Alg2DLBP.GetFeatVecConc();

		//Flatten Features
		featTestF_2[i] = Alg2DLBP.GetFeatVecFlatten();
	}
	cout << "\nClassification with NN classifier\n";
	NNClassifier classifierC_2(featTrainC_2, featTestC_2, labelsTrain, labelsTest, dimC, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Concatenation:\t\t" << 100 * classifierC_2.GetAccuracy() << "\n";

	NNClassifier classifierF_2(featTrainF_2, featTestF_2, labelsTrain, labelsTest, dimF, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Flatten:\t\t" << 100 * classifierF_2.GetAccuracy() << "\n\n\n";

	DeleteArrays(featTrainC_2);
	DeleteArrays(featTestC_2);
	DeleteArrays(featTrainF_2);
	DeleteArrays(featTestF_2);

	//***************************** TEST CASE 3 - GET OPTIMIZED FEATURES *****************************

	radius = 1;
	neighbour = 8;

	double** featTrainC_3 = new double* [500];
	double** featTestC_3 = new double* [500];
	double** featTrainF_3 = new double* [500];
	double** featTestF_3 = new double* [500];


	//Extracting features for train set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTrain[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading train images

		Alg Alg2DLBPOpt(image, Sign, Mean, neighbour, radius, MutualInformation, maxMetric, 1);

		//Concatenation Features
		featTrainC_3[i] = Alg2DLBPOpt.GetOptFeatVecConc();
		dimC = Alg2DLBPOpt.GetDimOptFeatVecConc();

		int rot = Alg2DLBPOpt.GetRotAngleOpt();

		//cout << "Rot: " << rot << "\n";

		// Flatten Features
		featTrainF_3[i] = Alg2DLBPOpt.GetOptFeatVecFlatten();
		dimF = Alg2DLBPOpt.GetDimOptFeatVecFlatten();
	}

	//Extracting features for test set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTest[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading test images

		Alg Alg2DOpt(image, Sign, Mean, neighbour, radius, MutualInformation, maxMetric, 1);

		//Concatenation Features
		featTestC_3[i] = Alg2DOpt.GetOptFeatVecConc();

		//Flatten Features
		featTestF_3[i] = Alg2DOpt.GetOptFeatVecFlatten();

	}
	cout << "\nClassification with NN classifier\n";
	NNClassifier classifierC_3(featTrainC_3, featTestC_3, labelsTrain, labelsTest, dimC, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Concatenation:\t\t" << 100 * classifierC_3.GetAccuracy() << "\n";

	NNClassifier classifierF_3(featTrainF_3, featTestF_3, labelsTrain, labelsTest, dimF, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Flatten:\t\t" << 100 * classifierF_3.GetAccuracy() << "\n\n\n";

	DeleteArrays(featTrainC_3);
	DeleteArrays(featTestC_3);
	DeleteArrays(featTrainF_3);
	DeleteArrays(featTestF_3);
	*/



	//***********************************************END***************************************************

	system("PAUSE");
	return 0;
}