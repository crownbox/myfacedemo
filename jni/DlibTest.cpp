#include <jni.h>

#include <com_zh_dlibtest_DlibTest.h>
#include <opencv2/opencv.hpp>

#include <./dlib/dlib/image_processing/frontal_face_detector.h>
#include <./dlib/dlib/image_processing/render_face_detections.h>
#include <./dlib/dlib/image_processing.h>
#include <./dlib/dlib/gui_widgets.h>
#include <./dlib/dlib/image_io.h>
#include <./dlib/dlib/opencv/cv_image.h>
#include <./dlib/dlib/opencv.h>

#include <seetafaceJNI.h>
#include <./track/kcftracker.h>

#include <android/log.h>

#define THERSHOLD             0.48


using namespace dlib;
using namespace std;
using namespace cv;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;//定义个shape_predictor类的实例
std::vector<cv::Point2d> pts2d;		// 用于存储检测的点
array2d<rgb_pixel> img;//注意变量类型 rgb_pixel 三通道彩色图像
std::vector<full_object_detection> shapes;//注意形状变量的类型，full_object_detection
bool initflag = false;


/****************************************************************************************/
cv::Rect kcfresult(0, 0, 0, 0);
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = false;
bool LAB = false;
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
bool trackflag =false;
bool recoflag = false;
bool poseflag = false;
/*****************************************************************************************/

SeetafaceJni seetafaceReco;
Tools tool;

float similarity =0;

bool detectflag = false;

int resizerate = 4;

static void throwJavaException(JNIEnv *env, const std::exception *e,
	const char *method,const string path) {
	std::string what = "unknow exception";
	string errorlogtxt=path+"/errorLog.txt";
	string errormethod=string(method);
	jclass je = 0;

	if (e) {
		std::string exception_type = "std::exception";

		if (dynamic_cast<const cv::Exception*>(e)) {
			exception_type = "cv::Exception";
			je = env->FindClass("org/opencv/core/CvException");
		}
		what = exception_type + ":" + e->what();
	}
	if (!je)
		je = env->FindClass("java/lang/Exception");
	tool.writeTxt(errorlogtxt,"method:"+errormethod+",caught :"+what.c_str());
	env->ThrowNew(je, what.c_str());

	(void) method; //avoid "unused" warning
}

// 画68个点
void draw68Points(cv::Mat &img, std::vector<cv::Point2d> pts2d)
{
	if (pts2d.size() != 68) return;

	for (int i = 0; i < 17; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0), 1);
	for (int i = 17; i < 27; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0), 1);
	for (int i = 27; i < 31; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0), 1);
	for (int i = 31; i < 36; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0));
	for (int i = 36; i < 48; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0), 1);
	for (int i = 48; i < 60; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0));
	for (int i = 60; i < 68; i++)	circle(img, (pts2d)[i], 2, cv::Scalar(255, 0, 0));

	return;
}

void init(string modelpath){
	string model =modelpath+"/model/shape_predictor_68_face_landmarks.dat";
	deserialize(model) >> sp;

}

void detect(cv::Mat &result,string modelpath){
	//LOGE("detect");
	assign_image(img, cv_image<bgr_pixel>(result));

	std::vector<dlib::rectangle> dets = detector(img);//检测人脸，获得边界框

	int Max = 0;
	int area = 0;
	if (dets.size() != 0)
	{
		for (unsigned long t = 0; t < dets.size(); ++t)
		{
			if (area < dets[t].width()*dets[t].height())
			{
				area = dets[t].width()*dets[t].height();
				Max = t;
			}
		}
	}

	full_object_detection shape = sp(img, dets[Max]);//预测姿势，注意输入是两个，一个是图片，另一个是从该图片检测到的边界框

	cv::Rect box(dets[Max].left(), dets[Max].top(), dets[Max].width(), dets[Max].height());
	//cout << box << endl;
	cv::rectangle(result, box, Scalar(0, 255, 0), 2, 8, 0);

	pts2d.clear();
	// 将结果保存至 pts2d 里面
	for (size_t k = 0; k < shape.num_parts(); k++) {
		Point2d p(shape.part(k).x(), shape.part(k).y());
		//cout << p << endl;
		pts2d.push_back(p);
	}

	draw68Points(result, pts2d);
	imwrite(modelpath+"/frame.jpg", result);

}

string int2string(int a){
		stringstream s;
		s << a;
		string i;

		s >> i;
		return i;
	}
/****************************************************************************************************************************/
int poseEstimation(Mat temp, dlib::rectangle box,string modelpath,double *coordinate){

	//double coordinate[3];
	//LOGE("aaa");
	//Mat temp;
	//img.copyTo(temp);
	//cvtColor(img, temp, CV_RGBA2BGR);

	double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
	double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

	Mat cam_matrix = Mat(3, 3, CV_64FC1, K);
	Mat dist_coeffs = Mat(5, 1, CV_64FC1, D);

	//fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
	std::vector<Point3d> object_pts;
	object_pts.push_back(Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
	object_pts.push_back(Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
	object_pts.push_back(Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
	object_pts.push_back(Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
	object_pts.push_back(Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
	object_pts.push_back(Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
	object_pts.push_back(Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
	object_pts.push_back(Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
	object_pts.push_back(Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
	object_pts.push_back(Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
	object_pts.push_back(Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
	object_pts.push_back(Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
	object_pts.push_back(Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
	object_pts.push_back(Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

	//2D ref points(image coordinates), referenced from detected facial feature
	std::vector<Point2d> image_pts;

	//result
	Mat rotation_vec;                           //3 x 1
	Mat rotation_mat;                           //3 x 3 R
	Mat translation_vec;                        //3 x 1 T
	Mat pose_mat = Mat(3, 4, CV_64FC1);     //3 x 4 R | T
	Mat euler_angle = Mat(3, 1, CV_64FC1);

	//reprojected 2D points
	//std::vector<Point2d> reprojectdst;
	//reprojectdst.resize(8);

	//temp buf for decomposeProjectionMatrix()
	Mat out_intrinsics = Mat(3, 3, CV_64FC1);
	Mat out_rotation = Mat(3, 3, CV_64FC1);
	Mat out_translation = Mat(3, 1, CV_64FC1);

	//text on screen
	ostringstream outtext;
	//LOGE("bbb:(%d,%d,%d,%d)",box.left(),box.top(),box.right(),box.bottom());
	//LOGE("channels:%d",temp.channels());
	dlib::cv_image<dlib::bgr_pixel> cimg(temp);
	dlib::full_object_detection shape = sp(cimg, box);
	//LOGE("ccc");
	//draw features
	/*for (unsigned int i = 0; i < 68; ++i)
	{
		circle(temp, Point(shape.part(i).x(), shape.part(i).y()), 2, Scalar(0, 0, 255), -1);
	}*/
	//fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
	image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
	image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
	image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
	image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
	image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
	image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
	image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
	image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
	image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
	image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
	image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
	image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
	image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
	image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

	//calc pose
	solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
	//LOGE("ddd");
	//reproject
	//cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

	//calc euler angle
	Rodrigues(rotation_vec, rotation_mat);
	hconcat(rotation_mat, translation_vec, pose_mat);
	decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, noArray(), noArray(), noArray(), euler_angle);

	//LOGE("eee");
	coordinate[0] = euler_angle.at<double>(0);
	coordinate[1] = euler_angle.at<double>(1);
	coordinate[2] = euler_angle.at<double>(2);

	image_pts.clear();

	return 0;
}

void dlibEstimate(Mat &result,string modelpath){
	assign_image(img, cv_image<bgr_pixel>(result));

	std::vector<dlib::rectangle> dets = detector(img);//检测人脸，获得边界框

	int Max = 0;
	int area = 0;
	if (dets.size() != 0)
	{
		for (unsigned long t = 0; t < dets.size(); ++t)
		{
			if (area < dets[t].width()*dets[t].height())
			{
				area = dets[t].width()*dets[t].height();
				Max = t;
			}
		}
	}
	else
	{
		return;
	}
	double coordinate[3];
	//full_object_detection shape = sp(img, dets[Max]);//预测姿势，注意输入是两个，一个是图片，另一个是从该图片检测到的边界框
	poseEstimation(result, dets[Max],modelpath,coordinate);

}
/****************************************************************************************************************************/

int collect(Mat img , int num,string filepath){
	seeta::Rect bbox = seetafaceReco.detect(img);//检测人脸
	if (bbox.width==0 || bbox.height==0){
		//LOGE("seetafaceReco detect null");
		return 0;
	}
	LOGE("111");
	dlib::rectangle box(bbox.x, bbox.y, (bbox.x + bbox.width), (bbox.y + bbox.height));
	double coordinate[3];
	LOGE("222");
	//cvtColor(img, img, CV_RGBA2BGR);
	poseEstimation(img, box,filepath,coordinate);//预测人脸姿态
	LOGE("333");
	tool.savePose(filepath, num, coordinate);//保存人脸姿态
	LOGE("444");
	//float feature[2048];
	//int ret = seetafaceReco.getFeature(img, bbox, feature);//提取人脸特征

	float feature[512];
	int ret = seetafaceReco.getNcnnFeature(img, bbox, feature);//提取人脸特征

	LOGE("555");
	ret = tool.saveF(filepath, num, feature);//保存人脸特征
	LOGE("666");
	return ret;
}

float identfy(Mat img ,string filepath){

	LOGE("identfy1");
	if(detectflag){
		return -2;
	}else{
		detectflag = true;
	}
	seeta::Rect bbox = seetafaceReco.detect(img);//检测人脸
	detectflag = false;

	LOGE("identfy2");
	if (bbox.width==0 || bbox.height==0){
		return -1;
	}

	LOGE("identfy3");
	dlib::rectangle box(bbox.x, bbox.y, (bbox.x + bbox.width), (bbox.y + bbox.height));
	double coordinate[3];

	poseEstimation(img, box,filepath,coordinate);//预测人脸姿态

	float ret = seetafaceReco.faceReco(img, bbox, coordinate, filepath);//人脸识别

	return ret;
}

float test_identfy(){
	Mat img1,img2;
	img1 = imread("/storage/sdcard1/facereco/data/1.jpg");
	img2 = imread("/storage/sdcard1/facereco/data/2.jpg");
	seeta::Rect bbox1 = seetafaceReco.detect(img1);//检测人脸
	seeta::Rect bbox2 = seetafaceReco.detect(img2);//检测人脸
	if (bbox1.width==0 || bbox2.width==0){
		return -1;
	}
	LOGE("bbox1:(%d,%d,%d,%d)",bbox1.x,bbox1.y,bbox1.width,bbox1.height);
	LOGE("bbox2:(%d,%d,%d,%d)",bbox2.x,bbox2.y,bbox2.width,bbox2.height);


	float ret = seetafaceReco.testSimi(img1,img2,bbox1,bbox2);//人脸识别
	LOGE("SeetafaceJni::testSimisim:%f",ret);
	return ret;
}

void kcfTrack(Mat inframe,int mode){
	tool.bench_start();
	Mat frame;
	cvtColor(inframe,frame,CV_RGBA2BGR);
	//pyrDown(frame,frame,Size(frame.cols/2,frame.rows/2));
	//pyrDown(frame,frame,Size(frame.cols/2,frame.rows/2));
	//resize(frame, frame, Size(frame.cols/resizerate, frame.rows/resizerate));
	//LOGE("mat cols:%d,rows:%d", frame.cols,frame.rows);
	if (!trackflag){

		if(detectflag){
			return ;
		}else{
			detectflag = true;
		}
		seeta::Rect bbox = seetafaceReco.detect(frame);//检测人脸
		detectflag = false;

		if (bbox.width==0 || bbox.height==0){
			LOGE("no face detect");
			return ;
		}
		LOGE("kcfinit");
		//cv::Rect kcfbox(bbox.x+20, bbox.y+30, bbox.width-40, bbox.height/3);
		cv::Rect kcfbox((bbox.x+20)/resizerate, (bbox.y+32)/resizerate, (bbox.width-40)/resizerate, bbox.height/3/resizerate);

		resize(frame, frame, Size(frame.cols/resizerate, frame.rows/resizerate));
		LOGE("mat----------------------- cols:%d,rows:%d", frame.cols,frame.rows);
		tracker.init(kcfbox, frame);
		kcfresult = tracker.update(frame);
		//cv::rectangle(frame, kcfbox, Scalar(0, 255, 255), 1, 8);
		trackflag = true;
		//LOGE("Track init");
	}
	else{
		//LOGE("Track update");
		resize(frame, frame, Size(frame.cols/resizerate, frame.rows/resizerate));
		//LOGE("mat cols:%d,rows:%d", frame.cols,frame.rows);
		kcfresult = tracker.update(frame);

		//cv::rectangle(frame, kcfresult, Scalar(0, 255, 255), 1, 8);
	}

	float totaltime = tool.bench_end();
	LOGE("track time = %f\n", totaltime);
}

//初始化dlib，和seetaface模型
JNIEXPORT jint  JNICALL Java_com_zh_dlibtest_DlibTest_initModel
  (JNIEnv *env, jclass jobject,jstring path) {
	//获取绝对路径
	const char* modelPath;
	modelPath = env->GetStringUTFChars(path, 0);
	if(modelPath == NULL) {
		return 0;
	}
	string MPath = modelPath;
	//MPath="/storage/sdcard1";

	LOGE("jniinitModel");
	try {
		if(!initflag){
			similarity = tool.readConfig(MPath+"/data/Sim.txt");
			LOGE("jniinitModel similarity%f",similarity);
			if(similarity==0){
				LOGE("mkSimtxt");
				similarity = THERSHOLD;
				tool.saveConfig(MPath+"/data/Sim.txt","similar",THERSHOLD);
			}
			resizerate = (int)tool.readConfig(MPath+"/data/Rate.txt");
			LOGE("jniinitModel resizerate%d",resizerate);
			if(resizerate==0){
				LOGE("mkRatetxt");
				resizerate = 4;
				tool.saveConfig(MPath+"/data/Rate.txt","rate",4);
			}

			init(MPath);
			seetafaceReco.init(MPath);
			initflag = true;
		}
		//test_identfy();
		return 1;
	} catch (const std::exception &e) {

	} catch (...) {

	}
	return 0;
}

JNIEXPORT jint  JNICALL Java_com_zh_dlibtest_DlibTest_collect
  (JNIEnv *env, jclass jobject, jlong inPtr, jstring path, jint num) {
	Mat input = *(Mat*) inPtr;
	int cnt = num;
	//获取绝对路径
	const char* modelPath;
	modelPath = env->GetStringUTFChars(path, 0);
	if(modelPath == NULL) {
		return 0;
	}
	string MPath = modelPath;
	MPath=MPath+"/data";

	LOGE("jnicollect");
	try{
		Mat img;
		cvtColor(input, img, CV_RGBA2BGR);
		int ret = collect(img , cnt, MPath);
		return ret;
	} catch (const std::exception &e) {

	} catch (...) {

	}

    return 0;
}

JNIEXPORT jint  JNICALL Java_com_zh_dlibtest_DlibTest_faceReco
  (JNIEnv *env, jclass jobject, jlong inPtr,jstring path) {
	jint ret = 0;
	const char *method_name = "faceReco";
	//获取绝对路径
	const char* modelPath;
	modelPath = env->GetStringUTFChars(path, 0);
	if(modelPath == NULL) {
		env->ReleaseStringUTFChars(path, modelPath);
		return ret;
	}
	//LOGE("fffffffff");
	string MPath = modelPath;
	MPath=MPath+"/data";

	//LOGE("jnifaceReco");
	float sim=0.0;
	try {
		//LOGE("qqqqqqq");
		Mat input = *(Mat*) inPtr;
		//LOGE("ddddddd");
		Mat img;
		//img = imread(MPath+"/11.jpg");
		cvtColor(input, img, CV_RGBA2BGR);
		//LOGE("cvtColor");
		//pyrDown(img,img,Size(img.cols/2,img.rows/2));
		//resize(img, img, Size(img.cols/2, img.rows/2));
		//LOGE("mat cols:%d,rows:%d", img.cols,img.rows);

		sim = identfy(img ,MPath);

		LOGE("similarity:%f,threshold:%f,path:%s\n",sim,similarity,MPath.c_str());
		if(sim>similarity){
			ret= 1;
			LOGE("Reco ok");
		}else if(sim>0){
			ret = 0;
			LOGE("Reco error");
		}else{
			if(trackflag ||(int)sim==-2){
				ret= 1;
				LOGE("nof Reco ok");
			}else if(sim==-1){
				ret= -1;
				LOGE("no detetct face");
			}else{
				LOGE("nof Reco error");
				ret = 0;
			}
		}

		env->ReleaseStringUTFChars(path, modelPath);
	} catch (const std::exception &e) {
		throwJavaException(env, &e, method_name,MPath);
	} catch (...) {
		throwJavaException(env, 0, method_name,MPath);
	}

    return ret;
}

JNIEXPORT jstring  JNICALL Java_com_zh_dlibtest_DlibTest_track
  (JNIEnv *env, jclass jobject, jlong matPtr, jlong outPtr,jint mode) {
	Mat *mat = (Mat*) matPtr;
	Mat *pMatOut =(Mat*) outPtr;
	*pMatOut = *mat;
	int trackmode = mode;

	//LOGE("jnitrack");

	try{
		kcfTrack(*mat,trackmode);
		if(trackflag){
			//LOGE("tracking...");
			if (kcfresult.width == 0 || kcfresult.height == 0){
				trackflag = false;
				string str = "false";
				const char* ret = str.c_str();
				return env->NewStringUTF(ret);
			}else{
				//cv::Rect drawbox(kcfresult.x*resizerate, kcfresult.y*resizerate, kcfresult.width*resizerate, kcfresult.height*resizerate);
				//cv::rectangle(*pMatOut, drawbox, Scalar(0, 255, 255), 1, 8);
				//LOGE("kcfresult:(%d,%d,%d,%d)",kcfresult.x,kcfresult.y,kcfresult.width,kcfresult.height);
				//LOGE("drawbox:(%d,%d,%d,%d)",drawbox.x,drawbox.y,drawbox.width,drawbox.height);
				//cv::rectangle(*pMatOut, kcfresult, Scalar(0, 255, 255), 1, 8);
			}
		}
		//LOGI("mat col: %d", mat->cols);
		//if(kcfresult.x<0 || kcfresult.y<0 || (kcfresult.x+kcfresult.width)>pMatOut->cols || (kcfresult.y+kcfresult.height)>pMatOut->rows){
		if(kcfresult.x*resizerate<0 || kcfresult.y*resizerate<0 || (kcfresult.x+kcfresult.width)*resizerate>pMatOut->cols || (kcfresult.y+kcfresult.height)*resizerate>pMatOut->rows){
			trackflag = false;
			string str = "false";
			const char* ret = str.c_str();
			return env->NewStringUTF(ret);
		}
	} catch (const std::exception &e) {

	} catch (...) {

	}

	string str ="";
	if (kcfresult.width == 0 || kcfresult.height == 0){
		trackflag = false;
		str = "false";
	}else{
		str = int2string(kcfresult.x*resizerate)+","+int2string(kcfresult.y*resizerate)+","+int2string(kcfresult.width*resizerate)+","+int2string(kcfresult.height*resizerate);
	}
    const char* ret = str.c_str();
    return env->NewStringUTF(ret);
}

