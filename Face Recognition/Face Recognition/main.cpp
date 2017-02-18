#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>

// Screen resolution constants
#define R_WIDTH 640
#define R_HEIGHT 480

using namespace std;
using namespace cv;

// Function headers
void detectFace(Mat frame, Scalar &face_colour);
void makeMesh(Mat face, Rect loc);
void terminate(VideoCapture &cap);
void toColour(int num, Scalar &dst);
void showUserId(Mat frame, Scalar colour, int face_count);
int averageColour(vector<int> cols);

// Global Constants
string face_cascade_name = "Data/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";
RNG random_num_gen(12345);
int low_canny_thresh = 45;
int canny_ratio = 3;

struct Colour {
	int r, g, b;
};

int main(int argc, const char **argv) {
	VideoCapture capture(0);
	Mat frame;
	Scalar face_colour;

	if (!capture.isOpened()) {
		printf("Error when opening camera\n");
		return -1;
	}

	// Set resolution
	capture.set(CV_CAP_PROP_FRAME_WIDTH, R_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, R_HEIGHT);

	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) {
		printf("error loading\n");
		return -1;
	}

	while (capture.read(frame)) {
		if (frame.empty()) {
			printf("No captured frame\n");
			break;
		}

		detectFace(frame, face_colour);
		imshow(window_name, frame);

		if (char(waitKey(10) == 27)) {
			break;
		}
		// save screenshot
		if (char(waitKey(10) == char('c'))) {
			printf("Saving\n");
			imwrite("current save.jpg", frame);
		}
	}

	return 0;
}

// Exit program after cleaning up
void terminate(VideoCapture &cap) {
	destroyAllWindows();
	cap.release();
	exit(EXIT_SUCCESS);
}

// Covert an integer colour into BGR format
void toColour(int num, Scalar &dst) {
	dst = Scalar(num &0xFF, (num >> 8) &0xFF, (num >> 16) &0xFF);
	// dst.b = num &0xFF;
	// dst.g = (num >> 8) &0xFF;
	// dst.r = (num >> 16) &0xFF;
}

void showUserId(Mat frame, Scalar colour, int face_count) {
	if (face_count == 0) {
		putText(frame, "No User Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
		return;
	}

	Scalar diff_black = colour - Scalar(0, 0, 0);
	Scalar diff_white = Scalar(255, 255, 255) - colour;
	//cout << diff_black << "\t" << diff_white << endl;

	double v_b = sqrt(pow(diff_black[0], 2) + pow(diff_black[1], 2) + pow(diff_black[2], 2));
	double v_w = sqrt(pow(diff_white[0], 2) + pow(diff_white[1], 2) + pow(diff_white[2], 2));

	if (v_w > v_b) {
		// id 1, me. or any other brown fella
		putText(frame, "User 1 Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
	}
	else {
		// id 2, roman. or another privileged fella
		putText(frame, "User 2 Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
	}
}

void makeMesh(Mat face, Rect loc) {
	Mat edges, disp(R_HEIGHT, R_WIDTH, CV_8UC3);

	disp = Scalar::all(0);

	cvtColor(face, edges, CV_BGR2GRAY);
	cvtColor(disp, disp, CV_BGR2GRAY);
	
	blur(edges, edges, Size(3, 3));
	Canny(edges, edges, low_canny_thresh, low_canny_thresh * canny_ratio);

	for (int i = 0; i < loc.height; i++) {
		for (int j = 0; j < loc.width; j++) {
			disp.at<Vec3b>(loc.y + i, loc.x + j) = edges.at<Vec3b>(i, j);
		}
	}

	imshow("Mesh", disp);
}

void detectFace(Mat frame, Scalar &face_colour) {
	vector<Rect> faces;
	Mat frame_gray;
	Scalar main_face_colour;
 
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	Rect main_face = Rect(0, 0, 0, 0);  // hold the main face to be recognized

	// Detect faces using the cascade
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(80, 80));

	for (size_t i = 0; i < faces.size(); i++) {
		if (faces[i].width * faces[i].height > main_face.width * main_face.height) {
			main_face.x = faces[i].x;
			main_face.y = faces[i].y;
			main_face.width = faces[i].width;
			main_face.height = faces[i].height;
		}

		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(229, 181, 51), 4);
	}

	//int col = frame.at<int>(main_face.x + main_face.width / 2, main_face.y + main_face.height / 2);  // TODO: Average the colour of the pixels in the face region
        //toColour(col, face_colour);

	main_face_colour = mean(frame(main_face));

	circle(frame, Point(main_face.x + main_face.width / 2, main_face.y + main_face.height / 2), 4, face_colour, 4);
	
	if (main_face != Rect(0, 0, 0, 0))  // Send to mask maker
		makeMesh(frame(main_face), main_face);

	rectangle(frame, main_face, main_face_colour, 5);  // Indicate main face

	showUserId(frame, main_face_colour, faces.size());
		
	//cout << face_colour << endl;
}
