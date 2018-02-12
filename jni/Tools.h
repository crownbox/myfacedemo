#include <iostream>
#include <vector>
#include "math.h"
#include "time.h"
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <android/log.h>
#include <fcntl.h>
#include <jni.h>
using namespace std;

#define TAG "Dlib-jni" // 这个是自定义的LOG的标识
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)  // 定义LOGV类型
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__) // 定义LOGF类型

struct facepose
{
	int id;
	string featurepath;
	double x;
	double y;
	double z;
};
void dumpFile(void *addr, off64_t size,const char * filename);
class Tools{

public:
	string int2string(int a);
	string float2string(float a);
	string double2string(double a);
	int string2int(string a);
	int string2float(string a);
	int string2double(string a);
	int saveF(string saveDir, int num, float* feature);
	int readF(string readDir, int num, float* feature);
	int savePose(string poseDir, int num, double* pose);
	vector<facepose> readPose(string poseDir);
	int findProperPose(vector<facepose> poses, double* pose);
	void writeTxt(string path, string content);
	int saveConfig(string configPath,string name, float value);
	float readConfig(string configPath);
	void bench_start();
	float bench_end();
public:
	struct timeval tv_begin;
	struct timeval tv_end;
	double elasped;

};
