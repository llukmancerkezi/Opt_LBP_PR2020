
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Alg.h"
#include "NNClassifier.h"

using namespace std;
using namespace cv;

//Dataset-Image path
string dirDataset = "./Dataset1/";

string nameTrainImg = "./Dataset1/train_img.txt";
string nameTestImg = "./Dataset1/test_img.txt";
string labelTrainImg = "./Dataset1/train_label.txt";
string labelTestImg = "./Dataset1/test_label.txt";

// number of samples for both train and test set
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
	vector<string> imgArrTrain = ReadImgName(nameTrainImg);
	vector<string> imgArrTest = ReadImgName(nameTestImg);

	int* labelsTrain = ReadLabel(labelTrainImg);
	int* labelsTest = ReadLabel(labelTestImg);

	int radius;
	int neighbour = 8;

	int dimF, dimC;
	Mat image;

	//***************************** TEST CASE 1 - GET TRADITIONAL LBP FEATURE *****************************
	cout << "\n\nTEST CASE 1 - GET TRADITIONAL LBP FEATURE\n\n";
	radius = 1;
	neighbour = 8;
	double** featTrainLBP = new double* [500];
	double** featTestLBP = new double* [500];

	//Extracting features for train set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTrain[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading train images

		Alg AlgLBP(image, Sign, riu, neighbour, radius);

		//LBP Features fot ith train image
		featTrainLBP[i] = AlgLBP.GetFeatVecConc();
		dimC = AlgLBP.GetDimFeatVecConc();
	}

	//Extracting features for test set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTest[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading test images

		Alg AlgLBP(image, Sign, riu, neighbour, radius);

		//LBP Features fot ith test image
		featTestLBP[i] = AlgLBP.GetFeatVecConc();

	}
	cout << "\nClassification with NN classifier\n";
	NNClassifier classifier(featTrainLBP, featTestLBP, labelsTrain, labelsTest, dimC, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy LBP Histogram:\t\t" << 100 * classifier.GetAccuracy() << "\n";

	DeleteArrays(featTrainLBP);
	DeleteArrays(featTestLBP);


	//***************************** TEST CASE 2 - GET 2D TRADITIONAL FEATURES *****************************
	cout << "\n\nTEST CASE 2 - GET 2D TRADITIONAL FEATURES\n\n";
	radius = 1; // R of LBP
	neighbour = 8; // Number of neighbours in LBP

	//Feature vectors for concatenated marginal histograms
	double** featTrainM = new double* [500];
	double** featTestM = new double* [500];

	//Feature vectors for flatten approach
	double** featTrainF = new double* [500];
	double** featTestF = new double* [500];

	//Extracting features for train set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTrain[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading train images

		Alg Alg2D(image, Sign, Mean, uniform, neighbour, radius);

		//Concatenated Features
		featTrainM[i] = Alg2D.GetFeatVecConc();
		// dimension of the vector
		dimC = Alg2D.GetDimFeatVecConc();

		// Flatten Features
		featTrainF[i] = Alg2D.GetFeatVecFlatten();
		// dimension of the vector
		dimF = Alg2D.GetDimFeatVecFlatten();
	}


	//Extracting features for test set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTest[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading test images

		Alg Alg2DLBP(image, Sign, Mean, uniform, neighbour, radius);

		//Concatenated Features
		featTestM[i] = Alg2DLBP.GetFeatVecConc();

		//Flatten Features
		featTestF[i] = Alg2DLBP.GetFeatVecFlatten();
	}
	cout << "\nClassification with NN classifier\n";
	NNClassifier classifierC(featTrainM, featTestM, labelsTrain, labelsTest, dimC, 500, 500);
	cout << "For neighbour= " << neighbour  << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Concatenation:\t\t" << 100 * classifierC.GetAccuracy() << "\n";

	NNClassifier classifierF(featTrainF, featTestF, labelsTrain, labelsTest, dimF, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Flatten:\t\t" << 100 * classifierF.GetAccuracy() << "\n\n\n";

	DeleteArrays(featTrainM);
	DeleteArrays(featTestM);
	DeleteArrays(featTrainF);
	DeleteArrays(featTestF);

	//***************************** TEST CASE 3 - GET OPTIMIZED FEATURES *****************************
	cout << "\n\nTEST CASE 3 - GET OPTIMIZED FEATURES\n\n";
	radius = 1;
	neighbour = 8;

	double** featTrainOptM = new double* [500];
	double** featTestOptM = new double* [500];
	double** featTrainOptF = new double* [500];
	double** featTestOptF = new double* [500];


	//Extracting features for train set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTrain[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading train images

		Alg Alg2DLBPOpt(image, Sign, Mean, neighbour, radius, MutualInformation, maxMetric, 1);

		//Concatenation Features
		featTrainOptM[i] = Alg2DLBPOpt.GetOptFeatVecConc();
		// dimension of the vector
		dimC = Alg2DLBPOpt.GetDimOptFeatVecConc();

		// rotation angle
		int rot = Alg2DLBPOpt.GetRotAngleOpt();

		// Flatten Features
		featTrainOptF[i] = Alg2DLBPOpt.GetOptFeatVecFlatten();
		dimF = Alg2DLBPOpt.GetDimOptFeatVecFlatten();
	}

	//Extracting features for test set
	for (int i = 0; i < 500; i++) {
		image = imread(dirDataset + imgArrTest[i], CV_LOAD_IMAGE_GRAYSCALE);   // Reading test images

		Alg Alg2DOpt(image, Sign, Mean, neighbour, radius, MutualInformation, maxMetric, 1);

		//Concatenation Features
		featTestOptM[i] = Alg2DOpt.GetOptFeatVecConc();

		//Flatten Features
		featTestOptF[i] = Alg2DOpt.GetOptFeatVecFlatten();

	}
	cout << "\nClassification with NN classifier\n";
	NNClassifier classifierC_3(featTrainOptM, featTestOptM, labelsTrain, labelsTest, dimC, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Concatenation:\t\t" << 100 * classifierC_3.GetAccuracy() << "\n";

	NNClassifier classifierF_3(featTrainOptF, featTestOptF, labelsTrain, labelsTest, dimF, 500, 500);
	cout << "For neighbour= " << neighbour << "\tRadius =" << radius << "\t\t";
	cout << "Accuracy Flatten:\t\t" << 100 * classifierF_3.GetAccuracy() << "\n\n\n";

	DeleteArrays(featTrainOptM);
	DeleteArrays(featTestOptM);
	DeleteArrays(featTestOptF);
	DeleteArrays(featTrainOptF);

	//***********************************************END***************************************************

	cout << "\n\nEND!\n\n";
	system("PAUSE");
	return 0;
}


