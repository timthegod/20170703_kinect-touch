//#include "stdafx.h"
#include <iostream>
#include <string>

//為了引用 NuiApi, NuiImageCamera, NuiSensor等標頭檔(kinect 1.8 SDK) 為了消除錯誤而改掉的錯誤 https://social.msdn.microsoft.com/Forums/en-US/46b8a23d-0022-47f2-97ee-03607191d302/error-in-nuisensorh?forum=kinectsdk
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define _WIN32_WINNT 0x0601             // Compile against Windows 7 headers
// Windows Header Files
#include <windows.h>
#include <Shlobj.h>

// NiTE.h Header
#include <NiTE.h>

// OpenNI Header
#include <OpenNI.h>
#include "NIButtons.h"

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>

// OpenCV Header
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//namespace: std, cv, openni
using namespace std;
using namespace cv;
using namespace openni;

// set const varible
const unsigned int XRES = 640;
const unsigned int YRES = 480;
const unsigned int BIN_THRESH_OFFSET = 5;
const unsigned int ROI_OFFSET = 70;
const unsigned int MEDIAN_BLUR_K = 5;
const float        DEPTH_SCALE_FACTOR = 255.0 / 4096.0;
const double       GRASPING_THRESH = 0.9;

// colors
const Scalar COLOR_BLUE = Scalar(240, 40, 0);
const Scalar COLOR_DARK_GREEN = Scalar(0, 128, 0);
const Scalar COLOR_LIGHT_GREEN = Scalar(0, 255, 0);
const Scalar COLOR_YELLOW = Scalar(0, 128, 200);
const Scalar COLOR_RED = Scalar(0, 0, 255);

// conversion from cvConvexityDefect
struct ConvexityDefect
{
    Point start;
    Point end;
    Point depth_point;
    float depth;
};

// Thanks to Jose Manuel Cabrera for part of this C++ wrapper function
void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<ConvexityDefect>& convexDefects)
{
    if (hull.size() > 0 && contour.size() > 0)
    {
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect *defectArray = 0;

        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), strDefects);

        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
        for (int i = 0; i < (int)contour.size(); i++) {
            CvPoint cp = { contour[i].x,  contour[i].y };
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int)hull.size();
        //int hullK[count];
        int* hullK = (int*)malloc(count * sizeof(int));
        for (int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);

        for (int i = 0; i<defects->total; i++) {
            ConvexityDefect def;
            def.start = Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end = Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth = defectArray[i].depth;
            convexDefects.push_back(def);
        }

        // release memory
        cvReleaseMemStorage(&contourStr);
        cvReleaseMemStorage(&strDefects);
        cvReleaseMemStorage(&storage);

    }
}

void average(vector<Mat1s>& frames, Mat1s& mean) {
    Mat1d acc(mean.size());
    Mat1d frame(mean.size());
    acc = 0.0;
    for (unsigned int i=0; i<frames.size(); i++) {
        frames[i].convertTo(frame, CV_64FC1);
        acc = acc + frame;
    }

    acc = acc / frames.size();

    acc.convertTo(mean, CV_16SC1);
}

int main(int argc, char** argv)
{
    const unsigned int nBackgroundTrain = 30;
    const unsigned short touchDepthMin = 10;
    const unsigned short touchDepthMax = 20;
    const unsigned int touchMinArea = 50;

    const double debugFrameMaxDepth = 4000; // maximal distance (in millimeters) for 8 bit debug depth frame quantization
    const char* windowName = "Debug";
    const Scalar debugColor0(0,0,128);
    const Scalar debugColor1(255,0,0);
    const Scalar debugColor2(255,255,255);

    int xMin = 110;
    int xMax = 560;
    int yMin = 120;
    int yMax = 320;

    Mat1s depth(480, 640); // 16 bit depth (in millimeters)
    Mat1b depth8(480, 640); // 8 bit depth
    Mat3b rgb(480, 640); // 8 bit depth

    Mat3b debug(480, 640); // debug visualization

    Mat1s foreground(640, 480);
    Mat1b foreground8(640, 480);

    Mat1b touch(640, 480); // touch mask

    Mat1s background(480, 640);
    vector<Mat1s> buffer(nBackgroundTrain);

    //Initial OpenNI
    if (OpenNI::initialize() != STATUS_OK) {
        cerr << "OpenNI Initial Error: " << OpenNI::getExtendedError() << endl;
        return -1;
    }
    Device mDevice;
    if (mDevice.open(ANY_DEVICE) != STATUS_OK) {
        cerr << "Can't Open Device: " << OpenNI::getExtendedError() << endl;
        return -1;
    }
    //Create depth stream
    VideoStream mDepthStream;
    if (mDevice.hasSensor(SENSOR_DEPTH)) {
        if (mDepthStream.create(mDevice, SENSOR_DEPTH) == STATUS_OK) {
            //set video mode
            VideoMode mMode;
            mMode.setResolution(640, 480);
            mMode.setFps(10);
            mMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
            if (mDepthStream.setVideoMode(mMode) != STATUS_OK) {
                cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;
            }
        } else {
            cerr << "Can't create depth stream on device: " << OpenNI::getExtendedError() << endl;
            return -1;
        }
    } else {
        cerr << "ERROR: This device does not have depth sensor" << endl;
        return -1;
    }
    //Create color stream
    VideoStream mColorStream;
    if (mDevice.hasSensor(SENSOR_COLOR)) {
        if (mColorStream.create(mDevice, SENSOR_COLOR) == STATUS_OK) {
            //set video mode
            VideoMode mMode;
            mMode.setResolution(640, 480);
            mMode.setFps(10);
            mMode.setPixelFormat(PIXEL_FORMAT_RGB888);
            if (mColorStream.setVideoMode(mMode) != STATUS_OK) {
                cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;
            }
            //image registration 設定自動影像校準技術(深度與彩圖整合): http://www.terasoft.com.tw/support/techpdf/Automating%20Image%20Registration%20with%20MATLAB.pdf
            mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
            //if (mDevice.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
            //    mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
            //} else {
            //    cerr << "Can't set ImageRegistration Mode."<< endl;
            //}
        }
        else {
            cerr << "Can't create color stream on device: " << OpenNI::getExtendedError() << endl;
            return -1;
        }
    }
    Mat depthShow(YRES, XRES, CV_8UC1);
    Mat colorShow(YRES, XRES, CV_8UC3);

    //偵測膚色前的轉換中介Matrix
    Mat intermideate;
    Mat skinColor;

    //深度現實圖層
    Mat FusionShow;

    namedWindow("depthFrame", CV_WINDOW_AUTOSIZE);
    namedWindow("colorFrame", CV_WINDOW_AUTOSIZE);
    VideoFrameRef  mDepthFrame;
    VideoFrameRef  mColorFrame;
    mDepthStream.start();
    mColorStream.start();
    vector<pair<int, int>> depth_do;// 儲存凸包點以供深度影像處理

    const openni::DepthPixel* pDepth; //紀錄深度

    vector<AbsNIButton*> vButtons;
    vButtons.push_back(new PressButton("Press",
        cv::Rect(70, 120, 100, 100),
        []() { cout << "Press Button 1" << endl; }));
    vButtons.push_back(new HoldButton("Hold",
        cv::Rect(70, 280, 100, 100),
        []() { cout << "Press Button 2" << endl; }));

    int keyboardKeynum = 0;

    namedWindow(windowName);
    createTrackbar("xMin", windowName, &xMin, 640);
    createTrackbar("xMax", windowName, &xMax, 640);
    createTrackbar("yMin", windowName, &yMin, 480);
    createTrackbar("yMax", windowName, &yMax, 480);
    for (unsigned int i=0; i<nBackgroundTrain; i++) {
        depth.data = (uchar*)mDepthFrame.getData();
        buffer[i] = depth;
    }

    average(buffer, background);
    while (keyboardKeynum != 27 && keyboardKeynum != 'q')
    {
        //get depth frame
        if (mDepthStream.isValid()) {
            if (mDepthStream.readFrame(&mDepthFrame) == STATUS_OK) {
                //convert data to OpenCV format             
                const Mat depthRaw(
                    mDepthFrame.getHeight(), mDepthFrame.getWidth(),
                    CV_16UC1, (void*)mDepthFrame.getData());
                depthRaw.convertTo(depthShow, CV_8U, DEPTH_SCALE_FACTOR);
                //水平翻轉
                cv::flip(depthShow, depthShow, 1);
            }
        }
        //check if color stream is available
        if (mColorStream.isValid()) {
            //get color frame
            if (mColorStream.readFrame(&mColorFrame) == STATUS_OK) {
                //convert data to OpenCV format
                const cv::Mat colorRaw(
                    mColorFrame.getHeight(), mColorFrame.getWidth(),
                    CV_8UC3, (void*)mColorFrame.getData());
                //convert form RGB to BGR
                cvtColor(colorRaw, colorShow, CV_RGB2BGR);
                //水平翻轉
                cv::flip(colorShow, colorShow, 1);
                //深度彩圖融合
                Mat depthShow_BGR;
                cvtColor(depthShow, depthShow_BGR, CV_GRAY2BGR);
                FusionShow = 0.2*colorShow + 0.8*depthShow_BGR;

                //draw buttom
                for (auto itButton = vButtons.begin(); itButton != vButtons.end(); ++itButton) (*itButton)->Draw(FusionShow);
            }
        }

        depth.data = (uchar*)mDepthFrame.getData();
        foreground = background - depth;
        touch = (foreground > touchDepthMin) & (foreground < touchDepthMax);
        Rect roi(xMin, yMin, xMax - xMin, yMax - yMin);
        Mat touchRoi = touch(roi);

        vector< vector<Point2i> > contours_any;
        vector<Point2f> touchPoints;
        findContours(touchRoi, contours_any, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2i(xMin, yMin));
        for (unsigned int i=0; i<contours_any.size(); i++) {
            Mat contourMat(contours_any[i]);
            // find touch points by area thresholding
            if ( contourArea(contourMat) > touchMinArea ) {
                Scalar center = mean(contourMat);
                Point2i touchPoint(center[0], center[1]);
                touchPoints.push_back(touchPoint);
            }
        }
        depth.convertTo(depth8, CV_8U, 255 / debugFrameMaxDepth); // render depth to debug frame
        cvtColor(depth8, debug, CV_GRAY2BGR);
        debug.setTo(debugColor0, touch);  // touch mask
        rectangle(debug, roi, debugColor1, 2); // surface boundaries
        for (unsigned int i=0; i<touchPoints.size(); i++) { // touch points
            circle(debug, touchPoints[i], 5, debugColor2, CV_FILLED);
        }
        imshow(windowName, debug);
        //將原始畫面傳換成比較好分析膚色的畫面
        cv::cvtColor(colorShow, intermideate, CV_BGR2YCrCb);
        cv::inRange(intermideate, Scalar(0, 137, 77), Scalar(256, 177, 127), skinColor);
        #ifdef DEBUG
        cv::imshow("intermideate", intermideate);
        #endif // DEBUG
        //結構簡化
        cv::erode(skinColor, skinColor, Mat(), Point(-1, -1), 5);
        cv::dilate(skinColor, skinColor, Mat(), Point(-1, -1), 5);
        medianBlur(skinColor, skinColor, MEDIAN_BLUR_K);
        Mat contoursCountMat = skinColor.clone(); 
        std::vector< std::vector<Point> > contours;
        findContours(contoursCountMat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
        cvtColor(skinColor, skinColor, CV_GRAY2RGB);//轉彩圖
        cout << contours.size() << endl;
        #ifdef DEBUG
        cv::imshow("contoursMat", contoursCountMat);
        #endif // DEBUG
        //如果輪廓存在
        if (contours.size()) {
            for (int iContours = 0; iContours < contours.size(); iContours++) {
                vector<Point> contour = contours[iContours];
                Mat contourMat = Mat(contour);
                double cArea = contourArea(contourMat);//計算輪廓面積
                if (cArea> 500) {
                    // 计算得到轮廓中心坐标
                    Scalar center = mean(contourMat);
                    Point centerPoint = Point(center.val[0], center.val[1]);

                    // 通过道格拉斯-普克算法得到一个简单曲线（近似的轮廓）
                    vector<Point> approxCurve;
                    approxPolyDP(contourMat, approxCurve, 20.0, true);

                    // 画出轮廓
                    vector< vector<Point> > debugContourV;
                    debugContourV.push_back(approxCurve);
                    drawContours(skinColor, debugContourV, 0, COLOR_DARK_GREEN, 3);
                    // 计算轮廓点的凸包。
                    vector<int> hull;
                    convexHull(Mat(approxCurve), hull, false, false);
                    for (int iHull = 0; iHull < hull.size(); iHull++) {
                        int indexHull = hull[iHull];
                        depth_do.push_back({ approxCurve[indexHull].x, approxCurve[indexHull].y });
                        cout <<"X: " << depth_do[iHull].first <<" Y: "<<depth_do[iHull].second <<  endl;
                         //mDepthStream.readFrame(&mDepthFrame);
                        pDepth  = (const openni::DepthPixel*)mDepthFrame.getData();
                        int idx = depth_do[iHull].second * (mDepthFrame.getWidth()) + depth_do[iHull].first ; // 位置值 =  x + y*width
                        cout << "depth  " << pDepth[idx] << endl; //深度值 
                        for (auto itButton = vButtons.begin(); itButton != vButtons.end(); ++itButton)
                            (*itButton)->CheckHand(depth_do[iHull].first, depth_do[iHull].second, pDepth[idx]);
                       
                        //const Mat depthData = (mDepthFrame.getHeight(), mDepthFrame.getWidth(), CV_8UC1);
                        circle(skinColor, approxCurve[indexHull], 3, COLOR_YELLOW, 2);

                        cv::putText(skinColor,
                            to_string(pDepth[idx]),
                            approxCurve[indexHull], // Coordinates
                            cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                            1.0, // Scale. 2.0 = 2x bigger
                            cv::Scalar(0, 0, 255), // Color
                            1, // Thickness
                            CV_AA); // Anti-alias
                    }
                    depth_do.clear();//清空VECTOR
                    // 查找凸缺陷
                    vector<ConvexityDefect> convexDefects;
                    findConvexityDefects(approxCurve, hull, convexDefects);
                    for (int iConvexDefects = 0; iConvexDefects < convexDefects.size(); iConvexDefects++) {
                        circle(skinColor, convexDefects[iConvexDefects].depth_point, 3, COLOR_BLUE, 2);
                    }
                    // 利用轮廓、凸包、缺陷等点坐标确定指尖等点坐标，并画出
                    vector<Point> hullPoints;
                    for (int iHull = 0; iHull < hull.size(); iHull++) {
                        int curveIndex = hull[iHull];
                        Point pt = approxCurve[curveIndex];

                        hullPoints.push_back(pt);
                    }
                    //是否握拳
                    double hullArea = contourArea(Mat(hullPoints));
                    double curveArea = contourArea(Mat(approxCurve));
                    double handRatio = curveArea / hullArea;
                    if (handRatio > GRASPING_THRESH) circle(skinColor, centerPoint, 5, COLOR_LIGHT_GREEN, 5);//握拳
                    else circle(skinColor, centerPoint, 5, COLOR_RED, 5);
                }
            }
        }
        Mat depthShow_BGR;
        cvtColor(depthShow, depthShow_BGR, CV_GRAY2BGR);
        skinColor = 0.5*skinColor + 0.5*depthShow_BGR;
        //show image
        imshow("skinColor", skinColor);
        imshow("colorFrame", colorShow);
        imshow("FusionShow", FusionShow);
        imshow("depthFrame", depthShow);
        //waitKey裡的值必>0，否則無法使用
        keyboardKeynum = waitKey(1);
    }
    mDepthStream.destroy();
    mColorStream.destroy();
    mDevice.close();
    OpenNI::shutdown();
    return 0;
}
