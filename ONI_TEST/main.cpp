#include <fstream>
#include <iostream>
#include <time.h>  

using namespace std;

#include<opencv\cv.h>
#include<opencv\cxcore.h>
#include<opencv\highgui.h>

// OpenNI Header
#include <XnOpenNI.h>
#include <XnTypes.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnPropNames.h>

#define SAMPLE_XML_PATH "C:\\Program Files (x86)\\OpenNI\\Data\\SamplesConfig.xml"
#define mydelay  50

void delay(int m)
{
    clock_t wait = m+ clock();
    while (wait > clock());
}

inline std::string printfstring(const char *message, ...)
{
    static char buf[8*1024];
    va_list va;
    va_start(va, message);
    vsprintf(buf, message, va);
    va_end(va);
    std::string str(buf);
    return str;
}

/// convert depth map to OpenCV
void xdepth2opencv(xn::DepthMetaData &xDepthMap, cv::Mat &im)
{
    int h=xDepthMap.YRes();
    int w=xDepthMap.XRes();
    const cv::Mat tmp(h, w, CV_16U, ( void *)xDepthMap.Data());
    tmp.copyTo(im);
}

/// convert image map to OpenCV
void ximage2opencv(xn::ImageMetaData &xImageMap, cv::Mat &im)
{
    int h=xImageMap.YRes();
    int w=xImageMap.XRes();
    const XnRGB24Pixel * xx=xImageMap.RGB24Data();
    const cv::Mat tmp(h, w, CV_8UC3, ( void *)xx);
    tmp.copyTo(im);
}

int main( int argc, char** argv )
{

	// Initial OpenNI Context	
	xn::Context xContext;
	xContext.Init();

	// Read .oni File
	xn::Player xPlayer;
	xContext.OpenFileRecording("D:\\RGBD-HuDaActA Color-Depth Video Database for Human Daily Activity Recognition DataSet\\USER10\\S15_C13_U10_B1.oni", xPlayer );
	xPlayer.SetRepeat( false );

	xn::ImageGenerator g_image;
	xn::ImageMetaData g_imageMD;
	xn::DepthGenerator g_depth;
	xn::DepthMetaData g_depthMD;
	g_depth.Create( xContext );
	g_image.Create( xContext );

	// get total frame number

	XnUInt32 depthFrames;
	xPlayer.GetNumFrames( g_depth.GetName(), depthFrames);

	std::cout << "Depth Frame: " << depthFrames  << std::endl;

	XnMapOutputMode imageMapMode;
	g_image.GetMapOutputMode(imageMapMode);
	int c = imageMapMode.nXRes;
	int r = imageMapMode.nYRes;

	XnUInt32 uFrames;
	xPlayer.GetNumFrames( g_depth.GetName(), uFrames );

	const XnRGB24Pixel* pImageMap;

	CvVideoWriter *vdo = cvCreateVideoWriter("file.avi", CV_FOURCC('P' , 'I' , 'M' , '1'), 30 , cvSize(c,r) , 1);
	
	xContext.StartGeneratingAll();
	std::cout << "Waiting For Processing ";

	for( unsigned int i = 0; i < uFrames; ++ i )
	{
		g_depth.WaitAndUpdateData();
        g_image.WaitAndUpdateData();
	
		g_image.GetMetaData(g_imageMD);
		//rgbi->imageData = (char *) rgb.data;

		cv::Mat im;
		
		std::string RGBfile = "C:\\Users\\matt-pc\\Desktop\\Photo\\RGB\\" + printfstring("image%05d.png", i);
		
		ximage2opencv(g_imageMD, im);
		IplImage* rgbi = cvCreateImage(cvSize(im.cols,im.rows), IPL_DEPTH_8U, 3);
		IplImage image_temp = im;
		cvCopy(&image_temp, rgbi);

		cv::Mat rgb_image(&image_temp);
		cv::imshow( "Example2", rgb_image );
		cvWaitKey(10);
        cv::imwrite(RGBfile.c_str(), im );
		cvWriteFrame(vdo, &image_temp);
        std::string Depthfile = "C:\\Users\\matt-pc\\Desktop\\Photo\\Depth\\" +  printfstring("depth%05d.png", i);
					
		g_depth.GetMetaData(g_depthMD);
		cv::Mat dim;
        xdepth2opencv(g_depthMD, dim);

        cv::imwrite(Depthfile.c_str(), dim );
		
		if ( i % 100 == 0)
			std::cout << ".";
		std::cout << "_\b";
		delay(mydelay);
		std::cout << "\\\b";
		delay(mydelay);
		std::cout << "|\b";
		delay(mydelay);
		std::cout << "/\b";
		delay(mydelay);
		
		i++;

	 }
	 cvReleaseVideoWriter(&vdo);
	 
	return 0;
}