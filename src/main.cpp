// TODO: Remember last locations of eyes to filter false positives

#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>

// Screen resolution constants
#define R_WIDTH  640
#define R_HEIGHT 480

using namespace std;
using namespace cv;

// Function headers
void detectFace(Mat frame, Scalar &face_colour);
void detectEyes(Mat frame);
void makeMesh(Mat face, Rect loc);
void terminate(VideoCapture &cap);
void toColour(int num, Scalar &dst);
void showUserId(Mat frame, Scalar colour, int face_count);

// Global Constants
string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eye_cascade_name = "haarcascade_eye.xml";
CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;
string window_name = "Face detection";
RNG random_num_gen(12345);
int low_canny_thresh = 45;
int canny_ratio = 3;

int main(int argc, const char **argv) {
	VideoCapture capture(0);
	Mat frame;
	Scalar face_colour;

	// Check if camera is open
	printf("Here we go\n");
	if (!capture.isOpened()) {
		printf("Error when opening camera\n");
		return -1;
	}

	// Set resolution
	capture.set(CV_CAP_PROP_FRAME_WIDTH, R_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, R_HEIGHT);

	// Load the cascades
	if (!face_cascade.load("Data/" + face_cascade_name)) {
		printf("Error loading face cascade\n");
		return -1;
	}
	if (!eye_cascade.load("Data/" + eye_cascade_name)) {
		printf("Error loading eye cascade\n");
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

void showUserId(Mat frame, Scalar colour, int face_count) {
	if (face_count == 0) {
		putText(frame, "No User Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
		return;
	}

	Scalar diff_black = colour - Scalar(33, 48, 94);
	Scalar diff_white = Scalar(196, 229, 240) - colour;

	//Scalar diff_black = colour - Scalar(0, 0, 0);
        //Scalar diff_white = Scalar(255, 255, 255) - colour;

	//cout << diff_black << "\t" << diff_white << endl;

	double v_b = sqrt(pow(diff_black[0], 2) + pow(diff_black[1], 2) + pow(diff_black[2], 2));
	double v_w = sqrt(pow(diff_white[0], 2) + pow(diff_white[1], 2) + pow(diff_white[2], 2));

	if (v_w > v_b) {
		// id 1, me. or any other brown fella
		putText(frame, "User 1 Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
		// cout << "1" << endl;
	}
	else {
		// id 2, roman. or another privileged fella
		putText(frame, "User 2 Detected", Point(10, 50), 5, 3, Scalar(255, 255, 255), 2);
		// cout << "2" << endl;
	}
}

void makeMesh(Mat face, Rect loc) {
	Mat edges, disp(R_HEIGHT, R_WIDTH, CV_8UC3);

	disp = Scalar::all(0);

	cvtColor(face, edges, CV_BGR2GRAY);
	cvtColor(disp, disp, CV_BGR2GRAY);
	
	blur(edges, edges, Size(3, 3));
	Canny(edges, edges, low_canny_thresh, low_canny_thresh * canny_ratio);

	threshold(edges, edges, 100, 255, cv::THRESH_BINARY);

	for (int i = 0; i < loc.height; i++) {
		for (int j = 0; j < loc.width; j++) {
			if (edges.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {
				disp.at<Vec3b>(loc.y + i, loc.x + j) = Vec3b(255, 255, 255);
			}
		}
	}

	//imshow("Mesh", disp);
}

void detectFace(Mat frame, Scalar &face_colour) {
	vector<Rect> faces, eyes;
	Mat frame_gray, frame_gray_blur;
	Scalar main_face_colour;
 
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	GaussianBlur(frame_gray, frame_gray_blur, Size(3, 3), 0, 0);

	Rect main_face = Rect(0, 0, 0, 0);  // hold the main face to be recognized
	Rect s_eye_area = Rect(0, 0, 0, 0);  // Area to search for eyes in

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

	main_face_colour = mean(frame(main_face));

	circle(frame, Point(main_face.x + main_face.width / 2, main_face.y + main_face.height / 2), 4, face_colour, 4);
	
	if (main_face != Rect(0, 0, 0, 0))  // Send to mask maker
		makeMesh(frame(main_face), main_face);

	rectangle(frame, main_face, main_face_colour, 5);  // Indicate main face

	// Find eyes in main face
	s_eye_area = Rect(main_face.x, main_face.y + (main_face.height * 0.1), main_face.width, main_face.height / 2);
	rectangle(frame, s_eye_area, Scalar(255, 255, 255), 5);

	eye_cascade.detectMultiScale(frame_gray_blur(s_eye_area), eyes, 1.1, 2);

	for (size_t i = 0; i < eyes.size(); i++) {
		circle(frame, Point(main_face.x + eyes[i].x + (eyes[i].width / 2), main_face.y + eyes[i].y + (eyes[i].height / 2)), eyes[i].width / 2, face_colour, 4);
	}

	showUserId(frame, main_face_colour, faces.size());
		
	//cout << face_colour << endl;
}

void detectEye(Mat frame, Rect main_face) {
	vector<Rect> eyes;
	Mat frame_gray, frame_gray_blur;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equilizeHist(frame_gray, frame_gray);

	GaussianBlur(frame_gray, frame_gray_blur, Size(3, 3), 0, 0);

	Rect s_eye_area = Rect(0, 0, 0, 0);  // Area to search for eyes in

	s_eye_area = Rect(main_face.x, main_face.y + (main_face.height * 0.1), main_face.width, main_face.height / 2);
	rectangle(frame, s_eye_area, Scalar(255, 255, 255), 5);

	eye_cascade.detectMultiScale(frame_gray_blur(s_eye_area), eyes, 1.1, 2);

	for (size_t i = 0, i < eyes.size(); i++) {
		circle(frame, Point(main_face.x + eyes[i].x + (eyes[i].width / 2), main_face.y + eyes[i].y + (eyes[i].height / 2)), eyes[i].width / 2, face_colour, 4);
	}
}
