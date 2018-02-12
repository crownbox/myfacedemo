LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := libncnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

OPENCV_INSTALL_MODULES:=on

include F:\OpenCV-android-sdk\sdk\native\jni\OpenCV.mk


LOCAL_C_INCLUDES += \
    dlib \
	FaceDetection \
    FaceIdentification \
    FaceAlignment \
    track

#LOCAL_MODULE    := OpencvNcnn
LOCAL_MODULE    := DlibTest
#LOCAL_MODULE    := OpencvProto
LOCAL_SRC_FILES := \
	./track/fhog.cpp \
	./track/kcftracker.cpp \
	./FaceDetection/face_detection.cpp \
	./FaceDetection/fust.cpp \
	./FaceDetection/image_pyramid.cpp \
	./FaceDetection/nms.cpp \
	./FaceDetection/lab_feature_map.cpp \
	./FaceDetection/surf_feature_map.cpp \
	./FaceDetection/lab_boost_model_reader.cpp \
	./FaceDetection/surf_mlp_model_reader.cpp \
	./FaceDetection/lab_boosted_classifier.cpp \
	./FaceDetection/mlp.cpp   \
	./FaceDetection/surf_mlp.cpp \
	./FaceAlignment/cfan.cpp  \
	./FaceAlignment/face_alignment.cpp \
	./FaceAlignment/sift.cpp \
	./FaceIdentification/bias_adder_net.cpp \
	./FaceIdentification/blob.cpp \
	./FaceIdentification/bn_net.cpp \
	./FaceIdentification/common_net.cpp \
	./FaceIdentification/conv_net.cpp \
	./FaceIdentification/eltwise_net.cpp \
	./FaceIdentification/inner_product_net.cpp \
	./FaceIdentification/log.cpp \
	./FaceIdentification/math_functions.cpp \
	./FaceIdentification/max_pooling_net.cpp \
	./FaceIdentification/net.cpp \
	./FaceIdentification/pad_net.cpp \
	./FaceIdentification/spatial_transform_net.cpp \
	./FaceIdentification/tform_maker_net.cpp \
	./FaceIdentification/aligner.cpp \
	./FaceIdentification/face_identification.cpp \
	./dlib//dlib/threads/threads_kernel_shared.cpp \
	./dlib/dlib/entropy_decoder/entropy_decoder_kernel_2.cpp \
	./dlib/dlib/base64/base64_kernel_1.cpp \
	./dlib/dlib/threads/threads_kernel_1.cpp \
	./dlib/dlib/threads/threads_kernel_2.cpp \
	Tools.cpp \
	DlibTest.cpp \
	seetafaceJni.cpp
	#OpencvProto.cpp
	#OpencvNcnn.cpp
	#OpencvProtonew.cpp

LOCAL_STATIC_LIBRARIES := ncnn

LOCAL_CFLAGS := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections

LOCAL_CFLAGS += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp
LOCAL_LDLIBS := -lz -llog -ljnigraphics -landroid 

include $(BUILD_SHARED_LIBRARY)