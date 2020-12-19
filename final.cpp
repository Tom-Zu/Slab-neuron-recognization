#include <opencv2\opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>
//#include <fstream>

using namespace cv;
//using namespace cvflann;
using namespace std;

struct Cell
{
    float x;
    float y;
    float size;
};

void AdaptiveThreshold(InputArray _src, OutputArray _dst, double maxValue,
    int method, int type, int blockSize, double delta, double percent)
{
    //CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    src.type() == CV_16U;
    CV_Assert(blockSize % 2 == 1 && blockSize > 1);
    Size size = src.size();

    _dst.create(size, src.type());
    Mat dst = _dst.getMat();

    if (maxValue < 0)
    {
        dst = Scalar(0);
        return;
    }

    //HAL(adaptiveThreshold, cv_hal_adaptiveThreshold, src.data, src.step, dst.data, dst.step, src.cols, src.rows,
    //maxValue, method, type, blockSize, delta);

    Mat mean;

    if (src.data != dst.data)
        mean = dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
        boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
            Point(-1, -1), true, BORDER_REPLICATE | BORDER_ISOLATED);
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        Mat srcfloat, meanfloat;
        src.convertTo(srcfloat, CV_32F);
        meanfloat = srcfloat;
        GaussianBlur(srcfloat, meanfloat, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE | BORDER_ISOLATED);
        meanfloat.convertTo(mean, src.type());
    }
    else
        //Error(StsBadFlag, "Unknown/unsupported adaptive threshold method");

        int i, j;
    uchar imaxval = saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    uchar tab[768];

    for (int i = 0; i < 768; i++)
        tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);

    if (src.isContinuous() && mean.isContinuous() && dst.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for (int i = 0; i < size.height; i++)
    {
        uchar* sdata = src.ptr(i);
        uchar* mdata = mean.ptr(i);
        uchar* ddata = dst.ptr(i);

        for (int j = 0; j < size.width; j++)
        {
            mdata[j] = mdata[j] * (1 - percent);
            ddata[j] = tab[sdata[j] - mdata[j] + 255];
        }
    }
}

void Process(int, void*);
int blur_size = 1;
int blur_radius = 1;
int fthresh_size = 1;
int fthresh_const = 1;
int fthresh_percent = 1;
int sthresh_size = 1;
int sthresh_const = 1;
int sthresh_percent = 1;
int farea = 20, sarea = 20;
int fcir = 1, scir = 1;
Mat in, green;
Mat blurred, first, second, frame, thresh, bot, ero;
Mat Temp;
vector<vector<Cell>> Cells;

int main()
{
    int count = 0;
    VideoCapture cap("auto.avi");
    VideoCapture cap1("green.avi");
    if (!cap.isOpened())
    {
        return -1;
    }
    if (!cap1.isOpened())
    {
        return -1;
    }
    //double fps = cap.get(CAP_PROP_FPS);
    double fps = cap.get(CAP_PROP_FPS);
    //namedWindow("final", WINDOW_AUTOSIZE);
    while (waitKey(0))
    {
        //Mat in(510, 510, CV_16UC1, Scalar(0, 0, 255));
        cap.read(in);
        if (in.empty())
        {
            break;
        }
        cap1.read(green);
        if (green.empty())
        {
            break;
        }
        
        in.convertTo(Temp, CV_16U);
        cvtColor(in, frame, COLOR_BGR2GRAY);
        //in = imread("example.png", IMREAD_GRAYSCALE);
        //imshow("origional", frame);

        //namedWindow("Gausian_Blur", WINDOW_AUTOSIZE);
        //namedWindow("Adaptive_Threshold", WINDOW_AUTOSIZE);
        createTrackbar("radius:", "Gausian_Blur", &blur_size, 100, Process);
        createTrackbar("size:", "Gausian_Blur", &blur_radius, 20, Process);
        createTrackbar("size:", "First_Adaptive_Threshold", &fthresh_size, 100, Process);
        createTrackbar("const:", "First_Adaptive_Threshold", &fthresh_const, 100, Process);
        createTrackbar("percent:", "First_Adaptive_Threshold", &fthresh_percent, 1000, Process);
        createTrackbar("size:", "Second_Adaptive_Threshold", &sthresh_size, 100, Process);
        createTrackbar("const:", "Second_Adaptive_Threshold", &sthresh_const, 100, Process);
        createTrackbar("percent:", "Second_Adaptive_Threshold", &sthresh_percent, 1000, Process);
        createTrackbar("area:", "first_keypoints", &farea, 100, Process);
        createTrackbar("circularity:", "first_keypoints", &fcir, 100, Process);
        createTrackbar("area:", "second_keypoints", &sarea, 100, Process);
        createTrackbar("circularity:", "second_keypoints", &scir, 100, Process);
        Process(0, 0);

        //imshow("regular blurr", im);
        char c = (char)waitKey(0);

        if (c == 27) {
            break;
        }
        count++;
    }
    return 0;
}

void Process(int, void*)
{
    vector<Cell> layer;
    imshow("origional", in);
    threshold(frame, thresh, 85, 30, THRESH_TRUNC);
    GaussianBlur(thresh, thresh, Size(blur_size * 2 + 1, blur_size * 2 + 1), blur_radius, blur_radius);
    imshow("thresh", thresh);
    GaussianBlur(frame, blurred, Size(blur_size * 2 + 1, blur_size * 2 + 1), blur_radius, blur_radius);
    imshow("Gausian_Blur", blurred);
    //erode(blurred, ero, cv::Mat(), Point(), 1, 2);
    threshold(blurred, ero, 85, 30, THRESH_TOZERO);
    imshow("erode", ero);
    //threshold(frame, bot, 95, 30, THRESH_TRUNC);
    //AdaptiveThreshold(blurred, im, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TRUNC, 35, 2, 0.078);
    AdaptiveThreshold(thresh, first, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TRUNC, fthresh_size * 2 + 1, fthresh_const, static_cast<double>(fthresh_percent) / 1000);
    AdaptiveThreshold(ero, second, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TRUNC, sthresh_size * 2 + 1, sthresh_const, static_cast<double>(sthresh_percent) / 1000);
    
    SimpleBlobDetector::Params params1;

    // Change thresholds
   // params.minThreshold = 10;
    //params.maxThreshold = 200;

    //params.filterByColor = true;
    //params.maxThreshold = 150;

    // Filter by Area.
    params1.filterByArea = true;
    params1.minArea = farea;
    params1.maxArea = 250;

    // Filter by Circularity
    params1.filterByCircularity = true;
    params1.minCircularity = static_cast<double>(fcir) / 100;

    // Filter by Convexity
    params1.filterByConvexity = false;
    params1.minConvexity = 0.3;
    //params.maxConvexity = 0.99;

    // Filter by Inertia
    params1.filterByInertia = false;
    //params.minInertiaRatio = 0.01;

    //params.minDistBetweenBlobs = 0;

    // Storage for blobs

    SimpleBlobDetector::Params params2;

    // Change thresholds
   // params.minThreshold = 10;
    //params.maxThreshold = 200;

    //params.filterByColor = true;
    //params.maxThreshold = 150;

    // Filter by Area.
    params2.filterByArea = true;
    params2.minArea = sarea;
    params2.maxArea = 250;

    // Filter by Circularity
    params2.filterByCircularity = true;
    params2.minCircularity = static_cast<double>(scir) / 100;

    // Filter by Convexity
    params2.filterByConvexity = false;
    params2.minConvexity = 0.3;
    //params.maxConvexity = 0.99;

    // Filter by Inertia
    params2.filterByInertia = false;

    std::vector<KeyPoint> keypoints;
    std::vector<KeyPoint> keypoints1;
    std::vector<KeyPoint> keypoints2;

    // Set up detector with params
    Ptr<SimpleBlobDetector> detector1 = SimpleBlobDetector::create(params1);
    Ptr<SimpleBlobDetector> detector2 = SimpleBlobDetector::create(params2);

    // Detect blobs
    detector1->detect(first, keypoints1);
    detector2->detect(second, keypoints2);

    Mat f_with_keypoints, s_with_keypoints, gr_with_keypoints, in_with_keypoints;
    drawKeypoints(in, keypoints1, f_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(in, keypoints2, s_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(first, keypoints1, first, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(second, keypoints2, second, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("First_Adaptive_Threshold", first);

    imshow("Second_Adaptive_Threshold", second);

    //drawKeypoints(green, keypoints, gr_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("first_keypoints", f_with_keypoints);
    imshow("second_keypoints", s_with_keypoints);

    for (int i = 0; i < keypoints2.size(); i++)
    {
        bool notfound = 1;
        for (int k = 0; k < keypoints1.size(); k++)
        {
            if (keypoints1[k].overlap(keypoints2[i],keypoints1[k])!=0)
                //pow(keypoints1[k].pt.x - keypoints2[i].pt.x, 2) + pow(keypoints1[k].pt.x - keypoints2[i].pt.x, 2) < max(pow(keypoints1[k].size/2, 2), pow(keypoints2[i].size/2, 2))
            {
                notfound = 0;
            }
        }
        if (notfound)
        {
            keypoints1.push_back(keypoints2[i]);
        }
    }

    drawKeypoints(in, keypoints1, in_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("total_keypoints", in_with_keypoints);

    //drawKeypoints(green, keypoints, greenfinal, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Show blobs
    //imshow("THG out", in_with_keypoints);
    //imshow("green", gr_with_keypoints);
    //imshow("CALCIUM", greenfinal);
    //waitKey(0);

    for (int i = 0; i < keypoints.size(); i++)
    {
        Cell a;
        a.x = keypoints[i].pt.x;
        a.y = keypoints[i].pt.y;
        a.size = keypoints[i].size;
        layer.push_back(a);
    }

    Cells.push_back(layer);
}