#include "Tools.h"

string Tools::int2string(int a){
	stringstream s;
	s << a;
	string i;

	s >> i;
	return i;
}

string Tools::float2string(float a){
	stringstream s;
	s << a;
	string i;

	s >> i;
	return i;
}

string Tools::double2string(double a){
	stringstream s;
	s << a;
	string i;

	s >> i;
	return i;
}

int Tools::string2int(string a){
	stringstream s;
	s << a;
	int i;

	s >> i;
	return i;
}

int Tools::string2float(string a){
	stringstream s;
	s << a;
	float i;

	s >> i;
	return i;
}

int Tools::string2double(string a){
	stringstream s;
	s << a;
	double i;

	s >> i;
	return i;
}

int Tools::saveF(string saveDir, int num, float* feature){
	char fname[100];
	string savepath = saveDir + "/p" + int2string(num) + ".txt";
	//sprintf(fname, savepath.c_str());
	LOGE("saveF:%s\n",savepath.c_str());
	FILE* fp = fopen(savepath.c_str(), "w");
	if (!fp){
		return 0;
	}
	for (int i = 0; i < 512; i++){
		fprintf(fp, "%f,", feature[i]);
	}
	fclose(fp);
	return 1;
}

int Tools::readF(string readDir, int num, float* feature){
	char fname[100];
	string readpath = readDir + "/p" + int2string(num) + ".txt";
	//sprintf(fname, readpath.c_str());

	FILE* fp = fopen(readpath.c_str(), "r");
	if (!fp){
		return 0;
	}
	for (int i = 0; i < 512; i++){
		fscanf(fp, "%f,", &feature[i]);
	}
	fclose(fp);
	
	return 1;
}

int Tools::savePose(string poseDir, int num, double* pose){
	string posefile = poseDir + "/pose.txt";
	ofstream poseout(posefile.c_str(), ios::app);
	if (!poseout.is_open())
	{
		return 0;
	}

	char picpath[100];
	string fpath = poseDir + "/p" + int2string(num) + ".txt";
	sprintf(picpath, "%s", fpath.c_str());

	poseout << num << " " << picpath << " " << pose[0] << " " << pose[1] << " " << pose[2] << "\n";
	poseout.close();
	return 1;
}

vector<facepose> Tools::readPose(string poseDir){
	vector<facepose> poses;
	int lable;
	char fp[100];
	char xx[10];
	char yy[10];
	char zz[10];

	int nNums = 0;

	char fname[100];
	string readpath = poseDir + "/pose.txt";
	//sprintf(fname, readpath.c_str());
	FILE *myfile = fopen(readpath.c_str(), "r");
	if (!myfile)
	{
		return poses;
	}
	while (fgets(fp, sizeof(fp) - 1, myfile)) ++nNums;

	rewind(myfile);

	for (int i = 0; i < nNums; i++){
		facepose pose;
		fscanf(myfile, "%d %s %s %s %s", &lable, fp, xx, yy, zz);
		
		pose.id = lable;
		pose.featurepath = fp;
		pose.x = atof(xx);
		pose.y = atof(yy);
		pose.z = atof(zz);
		poses.push_back(pose);		
	}
	fclose(myfile);
	return poses;
}

int Tools::findProperPose(vector<facepose> poses, double* pose){
	double maxSim = 0.0;
	facepose properface;
	for (int i = 0; i < poses.size(); i++){
		double sim = (pose[0] * poses[i].x + pose[1] * poses[i].y + pose[2] * poses[i].z) /
			(sqrt(pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2])*sqrt(poses[i].x * poses[i].x + poses[i].y * poses[i].y + poses[i].z * poses[i].z));
		if (sim > maxSim){
			maxSim = sim;
			properface = poses[i];
		}
	}
	return properface.id;
}

void Tools::writeTxt(string path, string content){

	FILE *file = fopen(path.c_str(), "a+");
	if (file != NULL)
	{
		fputs(content.c_str(), file);
	}

	fclose(file);
}

int Tools::saveConfig(string configPath,string name, float value){
	//string posefile = simDir + "/Sim.txt";
	ofstream poseout(configPath.c_str());
	if (!poseout.is_open())
	{
		return 0;
	}
	poseout <<name<<" "<< value << "\n";
	poseout.close();
	return 1;
}

float Tools::readConfig(string configPath){
	float value;
	int nNums = 0;
	char fp[100];
	//string readpath = simDir + "/Sim.txt";
	FILE *myfile = fopen(configPath.c_str(), "r");
	if (!myfile)
	{
		return 0;
	}
	while (fgets(fp, sizeof(fp) - 1, myfile)) ++nNums;

	rewind(myfile);

	for (int i = 0; i < nNums; i++){
		fscanf(myfile, "%s %f", fp,&value);
	}
	fclose(myfile);
	return value;
}

void Tools::bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

float Tools::bench_end()
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
    return elasped;
}

void dumpFile(void *addr, off64_t size, const char * filename) {

	int fd;
	fd = open(filename, O_RDWR | O_CREAT, 0666);
	if (fd == -1)
		return;

	int length = write(fd, addr, size);
}
