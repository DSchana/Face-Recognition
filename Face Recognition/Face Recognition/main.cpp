#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;
using namespace cv;

// Function headers
void detectFace(Mat frame, Scalar &face_colour);
void terminate(VideoCapture &cap);
void toColour(int num, Scalar &dst);
int averageColour(vector<int> cols);

// Global variables
string face_cascade_name = "Data/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";
RNG random_num_gen(12345);

struct Colour {
	int r, g, b;
};

int main(int argc, const char **argv) {
	VideoCapture capture(0);
	Mat frame;
	Scalar face_colour;

	if (!capture.isOpened()) {
		printf("Error when opening camera");
		return -1;
	}

	// Set resolution
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

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
	rectangle(frame, main_face, main_face_colour, 5);  // Indicate main face
	
	//cout << face_colour << endl;
}
