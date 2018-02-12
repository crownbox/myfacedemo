#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include "net.h"
#include "Tools.h"

#include <com_zh_opencvproto_Detector.h>

#include <./FaceIdentification/face_identification.h>
#include <./FaceAlignment/face_alignment.h>
#include <./FaceDetection/face_detection.h>

#include <android/log.h>

using namespace cv;
using namespace std;
using namespace seeta;

#define IMAGE_WIDTH_STD       96
#define IMAGE_HEIGHT_STD      112//裁剪的图片大小
#define ALIGN_UP              51
#define ALIGN_DOWN            92
const double DST_5POINTS[10] = { 30.2946, 65.5318, 48.0252, 33.5493, 62.7299, 51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

static bool initflag = false;
FaceDetection *detector;
FaceAlignment *point_detector;
FaceIdentification *face_recognizer;
ncnn::Net squeezenet;
Tools tool;
int num = 0;
cv::Rect box(0,0,0,0);
vector<Point2f> srcpoints;
string detectstr = "";
string alignstr = "";
string featurestr = "";
string simstr = "";

static float templatefeature[512];

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;
int getFeature(Mat img,float* feature);

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static float bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
    LOGE("%s time = %f\n",  comment,elasped);
    return elasped;
}

void initmodel(){
	LOGE("initmodel...");
	//detector = new FaceDetection("/storage/emulated/0/facereco/model/seeta_fd_frontal_v1.0.bin");
	//point_detector = new FaceAlignment("/storage/emulated/0/facereco/model/seeta_fa_v1.1.bin");
	//face_recognizer = new FaceIdentification("/storage/emulated/0/facereco/model/seeta_fr_v1.0.bin");
	detector = new FaceDetection("/storage/emulated/0/ncnn/seeta_fd_frontal_v1.0.bin");
	point_detector = new FaceAlignment("/storage/emulated/0/ncnn/seeta_fa_v1.1.bin");
	face_recognizer = new FaceIdentification("/storage/emulated/0/ncnn/seeta_fr_v1.0.bin");
	detector->SetMinFaceSize(80);
	detector->SetScoreThresh(2.f);
	detector->SetImagePyramidScaleFactor(0.8f);
	detector->SetWindowStep(4, 4);
	LOGE("qqqq...");
	squeezenet.load_param("/storage/emulated/0/ncnn/ncnn_resnet_new.proto");
	squeezenet.load_model("/storage/emulated/0/ncnn/ncnn_resnet_new.bin");

	cv::Mat frame = cv::imread("/storage/emulated/0/ncnn/1.jpg");
	getFeature(frame,templatefeature);
	tool.saveF("/storage/emulated/0/ncnn", 0, templatefeature);//保存人脸特征
}

void getAffineMatrix(double* src_5pts, const double* dst_5pts, double* M)
{
	double src[10], dst[10];
	memcpy(src, src_5pts, sizeof(double) * 10);
	memcpy(dst, dst_5pts, sizeof(double) * 10);
	// initialization
	double ptmp[2];
	ptmp[0] = ptmp[1] = 0;
	for (int i = 0; i < 5; ++i) {
		ptmp[0] += src[i];
		ptmp[1] += src[5 + i];
	}
	ptmp[0] /= 5;
	ptmp[1] /= 5;
	for (int i = 0; i < 5; ++i) {
		src[i] -= ptmp[0];
		src[5 + i] -= ptmp[1];
		dst[i] -= ptmp[0];
		dst[5 + i] -= ptmp[1];
	}
	// determine init angle
	double dst_x = (dst[3] + dst[4] - dst[0] - dst[1]) / 2, dst_y = (dst[8] + dst[9] - dst[5] - dst[6]) / 2;
	double src_x = (src[3] + src[4] - src[0] - src[1]) / 2, src_y = (src[8] + src[9] - src[5] - src[6]) / 2;
	double theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);
	// determine init scale
	double scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
	double pts1[10];
	double pts0[2];
	double _a = sin(theta), _b = cos(theta);
	pts0[0] = pts0[1] = 0;
	for (int i = 0; i < 5; ++i) {
		pts1[i] = scale*(src[i] * _b + src[i + 5] * _a);
		pts1[i + 5] = scale*(-src[i] * _a + src[i + 5] * _b);
		pts0[0] += (dst[i] - pts1[i]);
		pts0[1] += (dst[i + 5] - pts1[i + 5]);
	}
	pts0[0] /= 5;
	pts0[1] /= 5;

	double sqloss = 0;
	for (int i = 0; i < 5; ++i) {
		sqloss += ((pts0[0] + pts1[i] - dst[i])*(pts0[0] + pts1[i] - dst[i])
			+ (pts0[1] + pts1[i + 5] - dst[i + 5])*(pts0[1] + pts1[i + 5] - dst[i + 5]));
	}
	// optimization
	double square_sum = 0;
	for (int i = 0; i < 10; ++i) {
		square_sum += src[i] * src[i];
	}
	for (int t = 0; t < 200; ++t) {
		//cout << sqloss << endl;
		// determine angle
		_a = 0;
		_b = 0;
		for (int i = 0; i < 5; ++i) {
			_a += ((pts0[0] - dst[i])*src[i + 5] - (pts0[1] - dst[i + 5])*src[i]);
			_b += ((pts0[0] - dst[i])*src[i] + (pts0[1] - dst[i + 5])*src[i + 5]);
		}
		if (_b < 0) {
			_b = -_b;
			_a = -_a;
		}
		double _s = sqrt(_a*_a + _b*_b);
		_b /= _s;
		_a /= _s;

		for (int i = 0; i < 5; ++i) {
			pts1[i] = scale*(src[i] * _b + src[i + 5] * _a);
			pts1[i + 5] = scale*(-src[i] * _a + src[i + 5] * _b);
		}

		// determine scale
		double _scale = 0;
		for (int i = 0; i < 5; ++i) {
			_scale += ((dst[i] - pts0[0])*pts1[i] + (dst[i + 5] - pts0[1])*pts1[i + 5]);
		}
		_scale /= (square_sum*scale);
		for (int i = 0; i < 10; ++i) {
			pts1[i] *= (_scale / scale);
		}
		scale = _scale;

		// determine pts0
		pts0[0] = pts0[1] = 0;
		for (int i = 0; i < 5; ++i) {
			pts0[0] += (dst[i] - pts1[i]);
			pts0[1] += (dst[i + 5] - pts1[i + 5]);
		}
		pts0[0] /= 5;
		pts0[1] /= 5;

		double _sqloss = 0;
		for (int i = 0; i < 5; ++i) {
			_sqloss += ((pts0[0] + pts1[i] - dst[i])*(pts0[0] + pts1[i] - dst[i])
				+ (pts0[1] + pts1[i + 5] - dst[i + 5])*(pts0[1] + pts1[i + 5] - dst[i + 5]));
		}
		if (abs(_sqloss - sqloss) < 1e-2) {
			break;
		}
		sqloss = _sqloss;
	}
	// generate affine matrix
	for (int i = 0; i < 5; ++i) {
		pts1[i] += (pts0[0] + ptmp[0]);
		pts1[i + 5] += (pts0[1] + ptmp[1]);
	}
	//printMat(pts1, 2, 5);
	M[0] = _b*scale;
	M[1] = _a*scale;
	M[3] = -_a*scale;
	M[4] = _b*scale;
	M[2] = pts0[0] + ptmp[0] - scale*(ptmp[0] * _b + ptmp[1] * _a);
	M[5] = pts0[1] + ptmp[1] - scale*(-ptmp[0] * _a + ptmp[1] * _b);
}

Mat cropImgNew(Mat img,double* SRC_5POINTS){
	Mat M = Mat::zeros(2, 3, CV_64F);
	double *m = M.ptr<double>();

	getAffineMatrix(SRC_5POINTS, DST_5POINTS, m);
	Mat cropImg;
	cropImg = Mat::zeros(IMAGE_HEIGHT_STD, IMAGE_WIDTH_STD, img.type());
	warpAffine(img, cropImg, M, cropImg.size(), CV_INTER_LINEAR, 0, Scalar(127, 127, 127));//裁剪图片
	//cv::imwrite("/storage/emulated/0/ncnn/newcrop.jpg",cropImg);

	return cropImg;
}

string float2string(float a){
	stringstream s;
	s << a;
	string i;
	s >> i;
	return i;
}

int getFeature(Mat img,float* feature){
		bench_start();
		cv::Mat img_gray;
		if (img.channels() != 1){
			cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		}
		else{
			img_gray = img;
		}

		seeta::ImageData img_data;
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		img_data.num_channels = 1;

		std::vector<seeta::FaceInfo> faces = detector->Detect(img_data);

		float totaltime = bench_end("detect");
		detectstr = "Detect time:"+float2string(totaltime)+"ms";
		bench_start();

		int32_t num_face = static_cast<int32_t>(faces.size());

		if (num_face < 1){
			seeta::Rect bbox = { 0, 0, 0, 0 };
			return 0;
		}

		int index = 0;
		int area = 0;
		for (int32_t i = 0; i < num_face; i++) {
			int w = faces[i].bbox.width;
			int h = faces[i].bbox.height;
			int areatmp = w*h;
			if (areatmp > area){
				area = areatmp;
				index = i;
			}
		}
		seeta::Rect bbox = { faces[index].bbox.x, faces[index].bbox.y, faces[index].bbox.width, faces[index].bbox.height };
		//cv::Rect box(bbox.x, bbox.y, bbox.width, bbox.height);
		box.x = bbox.x;
		box.y = bbox.y;
		box.width = bbox.width;
		box.height = bbox.height;
		//cv::rectangle(img, box, Scalar(0, 255, 0), 2, 8, 0);

		cv::cvtColor(img, img_gray, CV_BGR2GRAY);
		ImageData gallery_img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
		gallery_img_data_gray.data = img_gray.data;

		std::vector<seeta::FaceInfo> gallery_faces;
		seeta::FaceInfo info = { bbox, 0, 0, 0, 0 };
		gallery_faces.push_back(info);

		seeta::FacialLandmark gallery_points[5];

		point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

		totaltime = bench_end("align");
		alignstr = "align time:"+float2string(totaltime)+"ms";
		bench_start();

		//vector<Point2f> srcpoints;
		double SRC_5POINTS[10];
		srcpoints.clear();
		for (int i = 0; i < 5; i++)
		{
			Point2f p(gallery_points[i].x, gallery_points[i].y);
			srcpoints.push_back(p);
			SRC_5POINTS[i] = gallery_points[i].x;
			SRC_5POINTS[i+5] = gallery_points[i].y;
		}
		//Mat crop = cropImg(img, srcpoints);
		Mat crop = cropImgNew(img, SRC_5POINTS);

		ncnn::Mat in = ncnn::Mat::from_pixels(crop.data, ncnn::Mat::PIXEL_BGR2RGB, crop.cols, crop.rows);
		const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
		const float var_vals[3] = {1.f/127.5f, 1.f/127.5f, 1.f/127.5f};
		in.substract_mean_normalize(mean_vals, var_vals);
		ncnn::Extractor ex = squeezenet.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(4);
		ex.input("data", in);
		ncnn::Mat out;
		ex.extract("feature", out);
		for (int j=0; j<out.c; j++)
		{
		    const float* prob = out.data + out.cstep * j;
			feature[j] =  prob[0];

		}
		//LOGE("%f ", feature[255]);

		totaltime = bench_end("feature");
		featurestr = "feature time:"+float2string(totaltime)+"ms";
		return 1;
}


float getSimilarity(float* f1,float* f2){
	float sim = face_recognizer->CalcSimilarity(f1, f2,512);
	LOGE("sim = %f\n", sim);
	simstr = "similarity:"+float2string(sim);
	return sim;
}



JNIEXPORT void JNICALL JNICALL Java_com_zh_opencvproto_Detector_detect(JNIEnv *env,
		jobject thiz, jlong matPtr, jlong outPtr) {
	if(!initflag){
		initmodel();
		initflag = true;
	}


	Mat *mat = (Mat*) matPtr;
	Mat *pMatOut =(Mat*) outPtr;
	*pMatOut = *mat;

	Mat input = *(Mat*) matPtr;
	Mat img;
	cvtColor(input, img, CV_RGBA2BGR);
	//cv::imwrite("/storage/emulated/0/facereco/data/crop.jpg",img);
	float feature[512];
	int ret = getFeature(img,feature);
	if(ret == 1){
		cv::rectangle(*pMatOut, box, Scalar(0, 255, 0), 2, 8, 0);
		for (int j=0; j<srcpoints.size(); j++){
			 circle(*pMatOut, srcpoints[j], 5, Scalar(225, 0, 225), 7, 8);

		}
		float similarity = getSimilarity(feature,templatefeature);

		//string siminfo =float2string(similarity);

		putText( *pMatOut, detectstr, Point( 10,30),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
		putText( *pMatOut, alignstr, Point( 10,60),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
		putText( *pMatOut, featurestr, Point( 10,90),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
		putText( *pMatOut, simstr, Point( 10,120),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
		//tool.saveF("/storage/emulated/0/ncnn/feature", num, feature);//保存人脸特征
		num++;
	}

	//LOGE("feature255:%f, templatefeature255:%f", feature[255],templatefeature[255]);

}

