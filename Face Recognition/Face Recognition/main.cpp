#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function headers
void detectFace(Mat frame);
void terminate(VideoCapture &cap);

// Global variables
string face_cascade_name = "Data/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";
RNG random_num_gen(12345);

int main(int argc, const char **argv) {
	VideoCapture capture(0);
	Mat frame;

	if (!capture.isOpened()) {
		printf("Error when opening camera");
		return -1;
	}

	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) {
		printf("error loading");
		return -1;
	}

	while (capture.read(frame)) {
		if (frame.empty()) {
			printf("No captured frame");
			break;
		}

		detectFace(frame);

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

void detectFace(Mat frame) {
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces using the cascade
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(80, 80));

	for (size_t i = 0; i < faces.size(); i++) {
		//Point centre(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].width / 2);
		//ellipse(frame, centre, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4);
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(229, 181, 51), 4);
	}

	imshow(window_name, frame);
}
