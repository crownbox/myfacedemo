package com.zh.dlibtest;

public class DlibTest {

	//初始化模型
	native static int initModel(String path);
	
	//人脸收集
	native static int collect(long input,String path,int num);
	
	//人脸识别	
	native static int faceReco(long input,String path);
	
	//人脸跟踪	
	native static String track(long input,long output,int mode);

}
