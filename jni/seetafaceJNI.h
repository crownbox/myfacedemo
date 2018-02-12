#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <./FaceIdentification/face_identification.h>
#include <./FaceAlignment/face_alignment.h>
#include <./FaceDetection/face_detection.h>
#include <Tools.h>
#include <net.h>

using namespace cv;
using namespace std;
using namespace seeta;

class SeetafaceJni{
public:
	int init(string RootDir);
	seeta::Rect detect(Mat img);
	vector<int> align(Mat img, seeta::Rect bbox);
	int getFeature(Mat img, seeta::Rect bbox, float* feature);
	int collect(Mat img, string saveDir, int num);
	float faceReco(Mat img, seeta::Rect bbox, double* pose, string readDir);
	float testSimi(Mat img1,Mat img2, seeta::Rect bbox1, seeta::Rect bbox2);

	Mat cropImg(Mat img,double* SRC_5POINTS);
	int getNcnnFeature(Mat img, seeta::Rect bbox, float* feature);
public:
	FaceDetection *detector;
	FaceAlignment *point_detector;
	FaceIdentification *face_recognizer;
	ncnn::Net squeezenet;
	Tools tool;
	vector<facepose> poses;
};
