#include "opencv2\core.hpp"
#include "opencv2\face.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static Mat normTo255(InputArray _src) {
	// Create normalized image
	Mat dst;
	
	switch (_src.getMat().channels())
	{
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
	default:
		_src.getMat().copyTo(dst);
		break;
	}

	return dst;
}

static void readCsv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int main(int argc, const char *argv[]) {
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv_file> <output_folder>" << endl;
		exit(1);
	}

	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}

	// Get path to csv
	string fn_csv = string(argv[1]);

	// Vectors hold images and corresponding labels
	vector<Mat> images;
	vector<int> labels;

	// Read in data
	try {
		readCsv(fn_csv, images, labels);
	}
	catch (cv::Exception &e) {
		cerr << "Error openinv file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	// Check for at least 2 images
	if (images.size() < 2) {
		CV_Error(Error::StsError, "At least 2 images are required in your data set");
	}

	// Get dimensions from first image
	int height = images[0].rows;
	int width = images[0].cols;

	// The last images from dataset are removed from the vector.
	// This is done, so that the training data (which we learn the
	// cv::BasicFaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[images.size() - 1];
	images.pop_back();
	labels.pop_back();

	// Create Eigenface model for the recognition.
	// It is trained with the images from the given CSV file.

	// TODO: Setup model
	// Ptr<FaceRecognizer> model =

	return 0;
}
