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

#define IMAGE_WIDTH_STD       128
#define IMAGE_HEIGHT_STD      128//裁剪的图片大小
#define ALIGN_UP              40
#define ALIGN_DOWN            88

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

static float templatefeature[256];

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
	detector = new FaceDetection("/storage/emulated/0/facereco/model/seeta_fd_frontal_v1.0.bin");
	point_detector = new FaceAlignment("/storage/emulated/0/facereco/model/seeta_fa_v1.1.bin");
	face_recognizer = new FaceIdentification("/storage/emulated/0/facereco/model/seeta_fr_v1.0.bin");
	detector->SetMinFaceSize(80);
	detector->SetScoreThresh(2.f);
	detector->SetImagePyramidScaleFactor(0.8f);
	detector->SetWindowStep(4, 4);

	squeezenet.load_param("/storage/emulated/0/ncnn/ncnn.proto");
	squeezenet.load_model("/storage/emulated/0/ncnn/ncnn.bin");

	cv::Mat frame = cv::imread("/storage/emulated/0/ncnn/1.jpg");
	getFeature(frame,templatefeature);
	tool.saveF("/storage/emulated/0/ncnn", 0, templatefeature);//保存人脸特征
}

Mat cropImg(Mat image, vector<Point2f> srcpoints){
		cv::Point2f src[2];
		cv::Point2f dst[2];
		src[0] = Point2f((srcpoints[1].x + srcpoints[0].x) / 2, (srcpoints[1].y + srcpoints[0].y) / 2);
		src[1] = Point2f((srcpoints[4].x + srcpoints[3].x) / 2, (srcpoints[4].y + srcpoints[3].y) / 2);
		//line(image, src[0], src[1], Scalar(0,0,255), 2, 8, 0);
		dst[0] = Point2f(IMAGE_WIDTH_STD / 2, ALIGN_UP);
		dst[1] = Point2f(IMAGE_WIDTH_STD / 2, ALIGN_DOWN);

		Mat cropImg;

		double src_d_y = src[0].y - src[1].y;
		double src_d_x = src[0].x - src[1].x;
		double src_dis = sqrt(pow(src_d_y, 2) + pow(src_d_x, 2));
		double dst_d_y = dst[0].y - dst[1].y;
		double dst_d_x = dst[0].x - dst[1].x;
		double dst_dis = sqrt(pow(dst_d_y, 2) + pow(dst_d_x, 2));
		double scale = dst_dis / src_dis;
		double angle = atan2(src_d_y, src_d_x) - atan2(dst_d_y, dst_d_x);// angle between two line segments
		double alpha = cos(angle)*scale;
		double beta = sin(angle)*scale;

		cv::Mat M(2, 3, CV_64F);
		double* m = M.ptr<double>();

		m[0] = alpha;
		m[1] = beta;
		m[2] = (dst[0].x - alpha*src[0].x - beta*src[0].y + dst[1].x - alpha*src[1].x - beta*src[1].y) / 2;
		m[3] = -beta;
		m[4] = alpha;

		// average of two points
		m[5] = (dst[0].y + beta*src[0].x - alpha*src[0].y + dst[1].y + beta*src[1].x - alpha*src[1].y) / 2;
		if (M.empty())
		{
			std::cout << "NULL" << std::endl;
		}

		cropImg = Mat::zeros(IMAGE_HEIGHT_STD, IMAGE_WIDTH_STD, image.type());
		warpAffine(image, cropImg, M, cropImg.size(), CV_INTER_LINEAR, 0, Scalar(127,127,127));//裁剪图片
		cvtColor(cropImg, cropImg, COLOR_BGR2GRAY);
		cv::imwrite("/storage/emulated/0/ncnn/crop.jpg",cropImg);
		//cropImg.convertTo(cropImg, CV_32FC1);
		//cropImg -= 106;
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
		srcpoints.clear();
		for (int i = 0; i < 5; i++)
		{
			Point2f p(gallery_points[i].x, gallery_points[i].y);
			srcpoints.push_back(p);
		}
		Mat crop = cropImg(img, srcpoints);


		ncnn::Mat in = ncnn::Mat::from_pixels_resize(crop.data, ncnn::Mat::PIXEL_GRAY, crop.cols, crop.rows, 128, 128);
		//const float mean_vals[1] = {106.f};
		//in.substract_mean_normalize(mean_vals, 0);
		const float mean_vals[1] = {127.5f};
		const float var_vals[1] = {1.f/127.5f};
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
	float sim = face_recognizer->CalcSimilarity(f1, f2,256);
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
	float feature[256];
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

