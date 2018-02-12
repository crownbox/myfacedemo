#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>

#include "net.h"

#include <com_zh_opencvncnn_Detector.h>


#include <./dlib/dlib/image_processing/frontal_face_detector.h>
#include <./dlib/dlib/image_processing/render_face_detections.h>
#include <./dlib/dlib/image_processing.h>
#include <./dlib/dlib/gui_widgets.h>
#include <./dlib/dlib/image_io.h>
#include <./dlib/dlib/opencv/cv_image.h>
#include <./dlib/dlib/opencv.h>

#include <./track/kcftracker.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <./FaceIdentification/face_identification.h>
#include <./FaceAlignment/face_alignment.h>
#include <./FaceDetection/face_detection.h>
#include <Tools.h>
#include <android/log.h>

using namespace cv;
using namespace std;
using namespace dlib;
using namespace seeta;



void dlibTest(){
	LOGE("dlibTest start...");
	string model ="/storage/emulated/0/facereco/model/shape_predictor_68_face_landmarks.dat";
	shape_predictor sp;//定义个shape_predictor类的实例
	deserialize(model) >> sp;
	frontal_face_detector detector = get_frontal_face_detector();
	cv::Mat result = cv::imread("/storage/emulated/0/ncnn/11.jpg");
	array2d<rgb_pixel> img;//注意变量类型 rgb_pixel 三通道彩色图像
	assign_image(img, cv_image<bgr_pixel>(result));
	std::vector<dlib::rectangle> dets = detector(img);//检测人脸，获得边界框
	if(dets.size() != 0){
		full_object_detection shape = sp(img, dets[0]);//预测姿势，注意输入是两个，一个是图片，另一个是从该图片检测到的边界框
		for (size_t k = 0; k < shape.num_parts(); k++) {
			Point2d p(shape.part(k).x(), shape.part(k).y());
			circle(result, p, 2, cv::Scalar(255, 0, 0), 1);
		}
		cv::Rect box(dets[0].left(), dets[0].top(), dets[0].width(), dets[0].height());
		cv::rectangle(result, box, Scalar(0, 255, 0), 2, 8, 0);
		cv::imwrite("/storage/emulated/0/ncnn/dlib68.jpg", result);

	}
	LOGE("dlibTest end...");
}

void ncnnTest(){
	long start, finish1, finish2, finish3, finish4, finish5, finish6;
	double totaltime;

	LOGE("ncnnTest start...");
	cv::Mat m = cv::imread("/storage/emulated/0/ncnn/11.jpg", 0);
	ncnn::Net squeezenet;
	squeezenet.load_param("/storage/emulated/0/ncnn/ncnn.proto");
	squeezenet.load_model("/storage/emulated/0/ncnn/ncnn.bin");
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_GRAY, m.cols, m.rows, 128, 128);
	start = clock();
	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	finish1 = clock();
	totaltime = (double)(finish1 - start) / CLOCKS_PER_SEC;
	LOGE("create_extractor time = %f\n", totaltime*1000);

	ex.input("data", in);
	finish2 = clock();
	totaltime = (double)(finish2 - finish1) / CLOCKS_PER_SEC;
	LOGE("ex.input time = %f\n", totaltime*1000);

	ncnn::Mat out;
	ex.extract("feature", out);
	finish3 = clock();
	totaltime = (double)(finish3 - finish2) / CLOCKS_PER_SEC;
	LOGE("extract time = %f\n", totaltime*1000);
	for (int j=0; j<out.c; j++)
	{
	    const float* prob = out.data + out.cstep * j;
	    //LOGE("%f ", prob[0]);
	}
	LOGE("ncnnTest end...");
}

void seetafaceTest(){
	LOGE("seetafaceTest start...");
	FaceDetection *detector = new FaceDetection("/storage/emulated/0/facereco/model/seeta_fd_frontal_v1.0.bin");
	FaceAlignment *point_detector = new FaceAlignment("/storage/emulated/0/facereco/model/seeta_fa_v1.1.bin");
	FaceIdentification *face_recognizer = new FaceIdentification("/storage/emulated/0/facereco/model/seeta_fr_v1.0.bin");
	detector->SetMinFaceSize(80);
	detector->SetScoreThresh(2.f);
	detector->SetImagePyramidScaleFactor(0.8f);
	detector->SetWindowStep(4, 4);

	cv::Mat img = imread("/storage/emulated/0/ncnn/11.jpg");
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

	int32_t num_face = static_cast<int32_t>(faces.size());

	if (num_face < 1){
		seeta::Rect bbox = { 0, 0, 0, 0 };
		return ;
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
	cv::Rect box(bbox.x, bbox.y, bbox.width, bbox.height);
	cv::rectangle(img, box, Scalar(0, 255, 0), 2, 8, 0);

	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
	ImageData gallery_img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
	gallery_img_data_gray.data = img_gray.data;

	std::vector<seeta::FaceInfo> gallery_faces;
	seeta::FaceInfo info = { bbox, 0, 0, 0, 0 };
	gallery_faces.push_back(info);

	seeta::FacialLandmark gallery_points[5];

	point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

	for (int i = 0; i < 5; i++)
	{
		Point2d p(gallery_points[i].x, gallery_points[i].y);
		circle(img, p, 2, cv::Scalar(255, 0, 0), 1);
	}

	//ImageData store data of an image without memory alignment.
	ImageData src_img_data(img.cols, img.rows, img.channels());
	src_img_data.data = img.data;
	// Extract feature: ExtractFeatureWithCrop
	float feature[2048];

	face_recognizer->ExtractFeatureWithCrop(src_img_data, gallery_points, feature);
	cv::imwrite("/storage/emulated/0/ncnn/seetaface.jpg", img);
	LOGE("seetafaceTest end...");
	return ;

}

void trackTest(){
	LOGE("trackTest start...");
	KCFTracker tracker(true, false, true, false);
	cv::Rect kcfresult(0, 0, 0, 0);
	cv::Mat frame = cv::imread("/storage/emulated/0/ncnn/11.jpg");
	cv::Rect kcfbox(100, 100, 100, 100);
	tracker.init(kcfbox, frame);
	kcfresult = tracker.update(frame);
	LOGE("trackTest end...");
}

JNIEXPORT void JNICALL JNICALL Java_com_zh_opencvncnn_Detector_detect(JNIEnv *env,jobject thiz) {
	long start, finish1, finish2,finish3,finish4;
	double totaltime1,totaltime2,totaltime3,totaltime4;
	start = clock();
	LOGE("Test start...");

	dlibTest();
	finish1 = clock();
	totaltime1 = (double)(finish1 - start) / CLOCKS_PER_SEC;

	ncnnTest();
	finish2 = clock();
	totaltime2 = (double)(finish2 - finish1) / CLOCKS_PER_SEC;

	seetafaceTest();
	finish3 = clock();
	totaltime3 = (double)(finish3 - finish2) / CLOCKS_PER_SEC;

	trackTest();
	finish4 = clock();
	totaltime4 = (double)(finish4 - finish3) / CLOCKS_PER_SEC;

	LOGE("Test end...");

	LOGE("dlibTest time = %f\n", totaltime1*1000);
	LOGE("ncnnTest time = %f\n", totaltime2*1000);
	LOGE("seetafaceTest time = %f\n", totaltime3*1000);
	LOGE("trackTest time = %f\n", totaltime4*1000);
}




