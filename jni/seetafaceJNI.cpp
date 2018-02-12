#include "seetafaceJNI.h"
#include <android/log.h>


#define IMAGE_WIDTH_STD       96
#define IMAGE_HEIGHT_STD      112//²Ã¼ôµÄÍ¼Æ¬´óÐ¡
#define THERSHOLD             0.48

const double DST_5POINTS[10] = { 30.2946, 65.5318, 48.0252, 33.5493, 62.7299, 51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };


int SeetafaceJni::init(string RootDIR){
	detector = new FaceDetection((RootDIR + "/model/seeta_fd_frontal_v1.0.bin").c_str());
	detector->SetMinFaceSize(80);
	detector->SetScoreThresh(2.f);
	detector->SetImagePyramidScaleFactor(0.8f);
	detector->SetWindowStep(4, 4);
	point_detector = new FaceAlignment((RootDIR + "/model/seeta_fa_v1.1.bin").c_str());
	face_recognizer = new FaceIdentification((RootDIR + "/model/seeta_fr_v1.0.bin").c_str());
	squeezenet.load_param((RootDIR + "/model/ncnn_resnet_new.proto").c_str());
	squeezenet.load_model((RootDIR + "/model/ncnn_resnet_new.bin").c_str());
	poses = tool.readPose(RootDIR+"/data");
	return 0;
}

seeta::Rect SeetafaceJni::detect(Mat img){
	cv::Mat img_gray;
	//LOGE("SeetafaceJni::detect0");
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
	//LOGE("SeetafaceJni::detect1");
	std::vector<seeta::FaceInfo> faces = detector->Detect(img_data);
	//LOGE("SeetafaceJni::detect2");
	int32_t num_face = static_cast<int32_t>(faces.size());

	if (num_face < 1){
		seeta::Rect bbox = { 0, 0, 0, 0 };
		return bbox;
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
	return bbox;
}


vector<int> SeetafaceJni::align(Mat img, seeta::Rect bbox){
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_BGR2GRAY);

	ImageData gallery_img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
	gallery_img_data_gray.data = img_gray.data;

	std::vector<seeta::FaceInfo> gallery_faces;
	seeta::FaceInfo info = { bbox, 0, 0, 0, 0 };
	gallery_faces.push_back(info);

	seeta::FacialLandmark gallery_points[5];

	point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

	vector<int> points;
	for (int i = 0; i < 5; i++)
	{
		std::cout << Point(gallery_points[i].x, gallery_points[i].y) << std::endl;
		points.push_back(gallery_points[i].x);
		points.push_back(gallery_points[i].y);
	}
	return points;
}

int SeetafaceJni::getFeature(Mat img, seeta::Rect bbox, float* feature){
	tool.bench_start();
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_BGR2GRAY);

	ImageData gallery_img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
	gallery_img_data_gray.data = img_gray.data;

	std::vector<seeta::FaceInfo> gallery_faces;
	seeta::FaceInfo info = { bbox, 0, 0, 0, 0 };
	gallery_faces.push_back(info);

	seeta::FacialLandmark gallery_points[5];

	point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
	//for (int i = 0; i < 5; i++)
	//{
		//LOGE("five_points:(%f,%f)\n", gallery_points[i].x, gallery_points[i].y);
	//}
	float totaltime = tool.bench_end();
	LOGE("identfy_faceReco_getFeature_align = %f\n", totaltime);

	tool.bench_start();
	ImageData gallery_img_data_color(img.cols, img.rows, img.channels());
	gallery_img_data_color.data = img.data;
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, feature);
	totaltime = tool.bench_end();
	LOGE("identfy_faceReco_getFeature_ExtractFeature = %f\n", totaltime);

	return 0;
}

int SeetafaceJni::collect(Mat img, string saveDir,int num){

	seeta::Rect box = detect(img);

	if (box.width==0 || box.height==0){
		return 0;
	}
	//float feature[2048];
	//int ret = getFeature(img, box, feature);
	float feature[512];
	int ret = getNcnnFeature(img, box, feature);

	ret = tool.saveF(saveDir, num, feature);

	return ret;
}


float SeetafaceJni::faceReco(Mat img, seeta::Rect bbox, double* pose, string readDir){
	if(poses.size()==0){
		LOGE("poses=null,reload \n");
		poses = tool.readPose(readDir);
	}
	int id = tool.findProperPose(poses, pose);
	float feature1[512];
	float feature2[512];
	getNcnnFeature(img, bbox, feature1);
	tool.readF(readDir, id, feature2);

	char name[128];
	char name2[128];
	sprintf(name, "/sdcard/feature/f1eature.%d", id);
	sprintf(name2, "/sdcard/feature/f2eature.%d", id);
	dumpFile(feature1, 512, name);
	dumpFile(feature2, 512, name2);
	float sim = face_recognizer->CalcSimilarity(feature1, feature2,512);
	return sim;
}

float SeetafaceJni::testSimi(Mat img1,Mat img2, seeta::Rect bbox1, seeta::Rect bbox2){
	float feature1[2048];
	float feature2[2048];
	getFeature(img1, bbox1, feature1);
	getFeature(img2, bbox2, feature2);
	string saveDir="/storage/sdcard1/facereco/data";
	tool.saveF(saveDir, 100, feature1);
	tool.saveF(saveDir, 101, feature2);
	float sim = face_recognizer->CalcSimilarity(feature1, feature2);

	LOGE("SeetafaceJni::testSimisim:%f",sim);
	return sim;
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

Mat SeetafaceJni::cropImg(Mat img,double* SRC_5POINTS){
	Mat M = Mat::zeros(2, 3, CV_64F);
	double *m = M.ptr<double>();

	getAffineMatrix(SRC_5POINTS, DST_5POINTS, m);
	Mat cropImg;
	cropImg = Mat::zeros(IMAGE_HEIGHT_STD, IMAGE_WIDTH_STD, img.type());
	warpAffine(img, cropImg, M, cropImg.size(), CV_INTER_LINEAR, 0, Scalar(127, 127, 127));//²Ã¼ôÍ¼Æ¬
	//cv::imwrite("/storage/emulated/0/facereco/newcrop.jpg",cropImg);

	return cropImg;
}
int SeetafaceJni::getNcnnFeature(Mat img, seeta::Rect bbox, float* feature){
		tool.bench_start();
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, CV_BGR2GRAY);

		ImageData gallery_img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
		gallery_img_data_gray.data = img_gray.data;

		std::vector<seeta::FaceInfo> gallery_faces;
		seeta::FaceInfo info = { bbox, 0, 0, 0, 0 };
		gallery_faces.push_back(info);

		seeta::FacialLandmark gallery_points[5];

		point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

		double SRC_5POINTS[10];
				//srcpoints.clear();
		for (int i = 0; i < 5; i++)
		{
					//Point2f p(gallery_points[i].x, gallery_points[i].y);
					//srcpoints.push_back(p);
			SRC_5POINTS[i] = gallery_points[i].x;
			SRC_5POINTS[i+5] = gallery_points[i].y;
			//LOGE("5point:%d,%d\n",SRC_5POINTS[i],SRC_5POINTS[i+5]);
		}
		Mat crop = cropImg(img, SRC_5POINTS);

		float totaltime = tool.bench_end();
		LOGE("identfy_faceReco_getFeature_align = %f\n", totaltime);
		tool.bench_start();

		ncnn::Mat in = ncnn::Mat::from_pixels(crop.data, ncnn::Mat::PIXEL_BGR2RGB, crop.cols, crop.rows);
		LOGE("in cols:%d,rows:%d,size:%d\n",crop.cols,crop.rows,in.c);
		const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
		const float var_vals[3] = {1.f/127.5f, 1.f/127.5f, 1.f/127.5f};
		in.substract_mean_normalize(mean_vals, var_vals);
		ncnn::Extractor ex = squeezenet.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(4);
		ex.input("data", in);

		ncnn::Mat out;

		int result=ex.extract("feature", out);

		ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
		LOGE("for begin out size:%d\n",out_flatterned.w);
		/*for (int j=0; j<out.c; j++)
		{

			const float* prob =(const float*)( out.data + out.cstep * j);

			feature[j] =  prob[0];
			LOGE("feature[%i]:%f",j,prob[0]);
		}*/


		for (int j=0; j<out_flatterned.w; j++)
		{
			feature[j] = out_flatterned[j];
		}

		LOGE("for end");
		totaltime = tool.bench_end();
		LOGE("identfy_faceReco_getFeature_ExtractFeature = %f\n", totaltime);
		return 0;

}
