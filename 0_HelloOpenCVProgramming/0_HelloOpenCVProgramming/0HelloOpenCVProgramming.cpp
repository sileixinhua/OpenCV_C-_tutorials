#include "stdafx.h"
#include <opencv2/opencv.hpp>
#pragma comment(lib,"opencv_world330.lib")

//using namespace cv;
//using namespace std;

int ImageShow() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat src = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	cv::imshow("src", src);

	cv::waitKey();

	return 0;
}

int Flip() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat src = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	int flipCode = atoi(filename);

	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::Mat dst;
	cv::flip(src, dst, flipCode);

	cv::imshow("src",src);
	cv::imshow("dst", dst);

	cv::waitKey();

	return 0;
}

int Resize() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat src, dst;

	float scaleW = 0.8;
	float scaleH = scaleW;

	src = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (src.empty()) {
		throw("Faild open file.");
	}

	int width = static_cast<float>(src.cols*scaleW);
	int height = static_cast<float>(src.rows*scaleH);
	resize(src, dst, cv::Size(width, height));

	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::waitKey();

	return 0;
}

int Rotate() {
	cv::Mat src, dst;

	float angle = 55.5;

	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	src = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::Point2f center = cv::Point2f(static_cast<float>(src.cols / 2),
		static_cast<float>(src.rows / 2));
	cv::Mat affineTrans = getRotationMatrix2D(center, angle, 1.0);

	cv::warpAffine(src,dst,affineTrans,src.size(),cv::INTER_CUBIC,cv::BORDER_REPLICATE);

	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\RotateDst.jpg",dst);

	cv::waitKey();
}

int RotateCotinue() {
	cv::Mat src, dst;

	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::Point2f center = cv::Point2f(static_cast<float>(src.cols / 2),
		static_cast<float>(src.rows / 2));

	cv::imshow("src",src);
	cv::namedWindow("dst",cv::WINDOW_AUTOSIZE);
	for (float angle = 0.0; angle < 360.0; angle++) {
		cv::Mat affineTrans = getRotationMatrix2D(center,angle,1.0);
		cv::warpAffine(src,dst,affineTrans,src.size(),cv::INTER_CUBIC);
		cv::imshow("dst",dst);
		if (cv::waitKey(1) >= 0)
			break;
	}

	return 0;
}

int Perspective(){
	cv::Mat src,dst;
	cv::Point2f dstPoint[4];
	int xMergin, yMergin;

	int pattern = 1;//0,1,2

	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	int x0 = src.cols / 4;
	int x1 = (src.cols / 4) / 3;
	int y0 = src.rows / 4;
	int y1 = (src.rows / 4) / 3;
	cv::Point2f srcPoint[4] = {
		cv::Point(x0,y0),
		cv::Point(x0,y1),
		cv::Point(x1,y1),
		cv::Point(x1,y0),
	};

	switch (pattern) {
	case 0:
		xMergin = src.cols / 10;
		yMergin = src.rows / 10;
		dstPoint[0] = cv::Point(x0 + xMergin, y0 + yMergin);
		dstPoint[1] = srcPoint[1];
		dstPoint[2] = srcPoint[2];
		dstPoint[3] = cv::Point(x1 - xMergin, y0 + yMergin);
		break;

	case 1:
		xMergin = src.cols / 8;
		yMergin = src.rows / 8;
		dstPoint[0] = srcPoint[0];
		dstPoint[1] = srcPoint[1];
		dstPoint[2] = cv::Point(x1 - xMergin, y1 - yMergin);
		dstPoint[3] = cv::Point(x1 - xMergin, y0 + yMergin);
		break;

	case 2:
		xMergin = src.cols / 6;
		yMergin = src.rows / 6;
		dstPoint[0] = cv::Point(x0 + xMergin, y0 + yMergin);
		dstPoint[1] = srcPoint[1];
		dstPoint[2] = cv::Point(x1 - xMergin, y1 - yMergin);
		dstPoint[3] = srcPoint[3]; 
		break;
	}

	cv::Mat perspectiveMmat = cv::getPerspectiveTransform(srcPoint,dstPoint);
	cv::warpPerspective(src,dst,perspectiveMmat,src.size(),cv::INTER_CUBIC);

	cv::imshow("src",src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Perspective.jpg", dst);
	
	cv::waitKey();
}

int Circles() {
	cv::Mat img0(400,400,CV_8UC3, cv::Scalar(150,150,150));
	circle(img0, cv::Point(200, 200), 50, cv::Scalar(255, 0, 0));
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\CirclesImg0.jpg",img0);

	cv::Mat img1(400, 400, CV_8UC3, cv::Scalar(150, 150, 150));
	circle(img1, cv::Point(200, 200), 100, cv::Scalar(0, 255, 0),3);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\CirclesImg1.jpg", img1);

	cv::Mat img2(400, 400, CV_8UC3, cv::Scalar(150, 150, 150));
	circle(img2, cv::Point(200, 200), 150, cv::Scalar(0, 0, 255),-1);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\CirclesImg2.jpg", img2);

	cv::imshow("img0", img0);
	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::waitKey();

	return 0;
}

int Lines() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat mat = cv::imread(filename);
	if (mat.empty()) {
		throw("Faild open file.");
	}

	int x0 = mat.cols / 4;
	int x1 = mat.cols * 3 / 4;
	int y0 = mat.rows / 4;
	int y1 = mat.rows * 3 / 4;

	cv::Point p0 = cv::Point(x0,y0);
	cv::Point p1 = cv::Point(x1, y1);
	cv::line(mat, p0, p1, cv::Scalar(0, 0, 255), 3, 4);
	p0.y = y1;
	p1.y = y0;
	cv::line(mat,p0,p1,cv::Scalar(255,0,0),3,4);
	
	cv::imshow("mat",mat);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Lines.jpg", mat);

	cv::waitKey();

	return 0;
}

int DrawRect() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat mat = cv::imread(filename);
	if (mat.empty()) {
		throw("Faild open file.");
	}

	cv::Point p0 = cv::Point(mat.cols / 8,mat.rows / 8);
	cv::Point p1 = cv::Point(mat.cols * 7 / 8,mat.rows * 7 / 8);

	rectangle(mat,p0,p1,cv::Scalar(0,255,0),5,8);

	cv::Point p2 = cv::Point(mat.cols * 2 / 8, mat.rows * 2 / 8);
	cv::Point p3 = cv::Point(mat.cols * 6 / 8, mat.rows * 6 / 8);

	rectangle(mat, p2, p3, cv::Scalar(0, 255, 255), 2, 4);
	
	cv::imshow("mat", mat);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\DrawRect.jpg", mat);

	cv::waitKey();

	return 0;
}

int DrawText() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat mat = cv::imread(filename);
	if (mat.empty()) {
		throw("Faild open file.");
	}

	cv::Point p = cv::Point(50,mat.rows / 2 - 50);
	cv::putText(mat, "Hello OpenCV", p, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);

	cv::imshow("mat", mat);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\DrawText.jpg", mat);

	cv::waitKey();

	return 0;
}

int GrayScale() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat src, dst;

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::cvtColor(src,dst,cv::COLOR_RGB2GRAY);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\GrayScale.jpg", dst);

	cv::waitKey();

	return 0;
}

int Equalize() {
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::Mat src, dst;

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}
	
	cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE).copyTo(src);

	equalizeHist(src,dst);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\GrayScale.jpg", dst);

	cv::waitKey();

	return 0;
}

int Threshold() {
	cv::Mat src, dst;
	double thresh = 60.0, maxval = 180.0;
	int type = cv::THRESH_BINARY;

	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::equalizeHist(src,dst);
	thresh = 80.0;
	maxval = 210.0;
	int number = 0;//0,1,2,3,4

	switch (number) {
	case 0:type = cv::THRESH_BINARY; break;
	case 1:type = cv::THRESH_BINARY_INV; break;
	case 2:type = cv::THRESH_TRUNC; break;
	case 3:type = cv::THRESH_TOZERO; break;
	case 4:type = cv::THRESH_TOZERO_INV; break;
	}

	cv::threshold(src,dst,thresh,maxval,type);

	cv::imshow("src",src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Threshold.jpg",dst);

	cv::waitKey();

	return 0;
}

int BitwiseNot() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	bitwise_not(src,dst);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\BitwiseNot.jpg", dst);

	cv::waitKey();

	return 0;
}

int Blur() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	int ksize = 22;

	blur(src, dst,cv::Size(ksize,ksize));

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Blur.jpg", dst);

	cv::waitKey();

	return 0;
}

int GaussianBlur(){
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	int ksize1 = 11;
	int ksize2 = 11;
	double sigma1 = 10.0;
	double sigma2 = 20.0;
	cv::GaussianBlur(src, dst,cv::Size(ksize1,ksize2), sigma1, sigma2);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\GaussianBlur.jpg", dst);

	cv::waitKey();

	return 0;
}

int Laplacian() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename,CV_LOAD_IMAGE_GRAYSCALE).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	Laplacian(src,dst,0);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Laplacian.jpg", dst);

	cv::waitKey();

	return 0;
}

int Sobel() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	Sobel(src, dst, -1,0,1);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Sobel.jpg", dst);

	cv::waitKey();

	return 0;
}

int Canny() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	double threshold1 = 40.0;
	double threshold2 = 200.0;

	Canny(src, dst, threshold1, threshold2);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Canny.jpg", dst);

	cv::waitKey();

	return 0;
}

int Dilate() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	dilate(src, dst, cv::Mat());

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Dilate.jpg", dst);

	cv::waitKey();

	return 0;
}

int Erode() {
	cv::Mat src, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	erode(src, dst, cv::Mat());

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Erode.jpg", dst);

	cv::waitKey();

	return 0;
}

int Add() {
	cv::Mat src1, src2, dst;

	const char* filename1 = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";
	const char* filename2 = "C:\\Code\\FirstOpenCVProgramming\\test.png";

	cv::imread(filename1).copyTo(src1);
	if (src1.empty()) {
		throw("Faild open file.");
	}
	cv::imread(filename2).copyTo(src2);
	if (src2.empty()) {
		throw("Faild open file.");
	}

	add(src1, src2, dst);
	cv::imshow("src1", src1);
	cv::imshow("src2", src2);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Add.jpg", dst);

	cv::waitKey();

	return 0;
}

int DispBasic() {
	cv::VideoCapture capture(0);
	
	int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	std::cout << "frame size = " << width << " * " << height << std::endl;

	const char* wName = "camera";
	cv::Mat src;
	cv::namedWindow(wName,CV_WINDOW_AUTOSIZE);
	while (true) {
		capture >> src;
		cv::imshow(wName,src);
		if (cv::waitKey(1) >= 0) {
			break;
		}
	}

	return 0;
}

int DetectConers() {
	cv::Mat src, gray, dst;
	const int maxCorners = 50, blockSize = 3;
	const double qualityLevel = 0.01, minDistance = 20.0, k = 0.04;
	const bool useHarrisDetector = false;
	std::vector< cv::Point2f > corners;

	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	dst = src.clone();

	cvtColor(src,gray,cv::COLOR_RGB2GRAY);
	goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel,
		minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

	for (size_t i = 0; i < corners.size(); i++) {
		circle(dst,corners[i],8,cv::Scalar(255,255,0),2);
	}

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\DetectConers.jpg", dst);

	cv::waitKey();

	return 0;
}

int EliminateObjects() {
	cv::Mat src, mask, dst;

	const char* filename1 = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";
	const char* filename2 = "C:\\Code\\FirstOpenCVProgramming\\lenaMask.jpg";

	cv::imread(filename1).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cv::imread(filename2,CV_LOAD_IMAGE_GRAYSCALE).copyTo(mask);
	if (mask.empty()) {
		throw("Faild open file.");
	}

	inpaint(src,mask,dst,1,cv::INPAINT_TELEA);

	cv::imshow("src", src);
	cv::imshow("mask", mask);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\EliminateObjects1.jpg", dst);

	cv::waitKey();

	return 0;
}

int DetectFace() {
	cv::Mat src, gray, equalize, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cvtColor(src, gray, cv::COLOR_RGB2GRAY);
	equalizeHist(gray, equalize);

	cv::CascadeClassifier objDetector("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");

	std::vector<cv::Rect> objs;
	objDetector.detectMultiScale(equalize,objs,1.2,2,CV_HAAR_SCALE_IMAGE,cv::Size(30,30));

	src.copyTo(dst);
	std::vector<cv::Rect>::const_iterator it = objs.begin();
	for (; it != objs.end(); ++it) {
		rectangle(dst, cv::Point(it->x,it->y),
			cv::Point(it->x + it ->width,it->y + it->height),
			cv::Scalar(0,0,255),2,CV_AA);
	}

	cv::imshow("src",src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\DetectFace.jpg", dst);

	cv::waitKey();
}

int DetectEye() {
	cv::Mat src, gray, equalize, dst;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename).copyTo(src);
	if (src.empty()) {
		throw("Faild open file.");
	}

	cvtColor(src, gray, cv::COLOR_RGB2GRAY);
	equalizeHist(gray, equalize);

	cv::CascadeClassifier objDetector("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml");

	std::vector<cv::Rect> objs;
	objDetector.detectMultiScale(equalize, objs, 1.2, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

	src.copyTo(dst);
	std::vector<cv::Rect>::const_iterator it = objs.begin();
	for (; it != objs.end(); ++it) {
		rectangle(dst, cv::Point(it->x, it->y),
			cv::Point(it->x + it->width, it->y + it->height),
			cv::Scalar(0, 0, 255), 2, CV_AA);
	}

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\DetectEye.jpg", dst);

	cv::waitKey();
}

int Akaze() {
	cv::Mat src1, src2;
	const char* filename = "C:\\Code\\FirstOpenCVProgramming\\lena.jpg";

	cv::imread(filename,cv::IMREAD_COLOR).copyTo(src1);
	if (src1.empty()) {
		throw("Faild open file.");
	}
	cv::imread(filename, cv::IMREAD_COLOR).copyTo(src2);
	if (src2.empty()) {
		throw("Faild open file.");
	}

	std::vector<cv::KeyPoint> keypoint1, keypoint2;
	cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();

	detector->detect(src1, keypoint1);
	detector->detect(src2, keypoint2);

	cv::Mat descriptor1, descriptor2;
	cv::Ptr<cv::FeatureDetector> extractor = cv::AKAZE::create();

	extractor->compute(src1, keypoint1, descriptor1);
	extractor->compute(src2, keypoint2, descriptor2);

	std::vector<cv::DMatch> matches;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

	cv::Mat dst;
	drawMatches(src1,keypoint1,src2,keypoint1,matches,dst);

	cv::imwrite("C:\\Code\\FirstOpenCVProgramming\\Akaze.jpg",dst);
	cv::imshow("dst",dst);
	cv::imshow("src1", src1);
	cv::imshow("src2", src2);

	cv::waitKey();

	return 0;
}

int main()
{
	//ImageShow(); // 显示图片

	//Flip(); // 翻转图片

	//Resize(); //图像扩大缩小

	//Rotate(); // 回转图像

	//RotateCotinue(); //连续回转图像

	//Perspective(); //透视形式的投影

	//Circles(); //画圆圈

	//Lines(); //在图像上画直线

	//DrawRect(); //在图像上画四边形

	//DrawText(); //在图像上添加文字水印

	//GrayScale(); //图像灰度化

	//Equalize(); //灰度平滑化

	//Threshold(); //阈值化处理

	//BitwiseNot(); //图像颜色的反转

	//Blur(); //图像的模糊化处理

	//GaussianBlur(); //图像高斯模糊化处理

	//Laplacian(); //拉普拉斯滤波处理

	//Sobel(); // sobel边缘检测

	//Canny(); // canny边缘检测

	//Dilate(); //图像的膨胀

	//Erode(); //图像的收缩

	//Add(); //图像叠加

	//DispBasic(); //摄像头基本图像展示

	//DetectConers(); // 图像上角检测

	//EliminateObjects(); //图像识别消去1

	//DetectFace();//图像上脸部识别

	//DetectEye();//图像眼睛识别

	//Akaze();//图像的特征点提取

    return 0;
}

