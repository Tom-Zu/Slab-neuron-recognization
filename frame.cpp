#include "opencv2\opencv.hpp"

using namespace cvflann;

//#include "opencv2/core/openvx/ovx_defs.hpp"

using namespace cv;
using namespace std;

void AdaptiveThreshold(InputArray _src, OutputArray _dst, double maxValue,
    int method, int type, int blockSize, double delta, double percent)
{
    //CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(blockSize % 2 == 1 && blockSize > 1);
    Size size = src.size();

    _dst.create(size, src.type());
    Mat dst = _dst.getMat();

    if (maxValue < 0)
    {
        dst = Scalar(0);
        return;
    }

    //CALL_HAL(adaptiveThreshold, cv_hal_adaptiveThreshold, src.data, src.step, dst.data, dst.step, src.cols, src.rows,
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
        //CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

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


int main(int argc, char** argv)
{
    /*VideoCapture cap("example.mp4");
    if (!cap.isOpened()) {
         cout << "Error opening video stream or file" << endl;
         return -1;
    }
    while (1) {
        Mat blurred, im, in;
        cap >> in;
        if (in.empty())
            break;*/

        // Read image
        Mat blurred, im, in, green, out;
        in = imread("auto4.png", IMREAD_GRAYSCALE);
        green = imread("green3.png");
        imshow("THG in", in);
        
        GaussianBlur(in, im, Size(5, 5), 15, 15);

        //medianBlur(in, blurred, 5);

        //bilateralFilter(in, blurred, 5, 5000, 100);

        //blur(in, blurred, Size(3, 3));

        //imshow("Blurr", blurred);

        //imshow("median blurr", median);

        //imshow("bilateral blurr", bilateral);

        //imshow("regular blurr", regular);


        //blur(in, blurred, Size(5, 5));
        //Canny(Gaussian, out, 0, 50, 5);

        //GaussianBlur(in, blurred, Size(9, 9), 11, 11);

        imshow("Blurr", im);

        //threshold(blurred, im, 81, 255, THRESH_TRUNC);

        //imshow("trunc ", im);

        AdaptiveThreshold(im, im, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TRUNC, 35, 2, 0.078);

        imshow("Threshold ", im);

        //blur(im, blurred, Size(5, 5));

        //imshow("Blurr 2", blurred);

        //AdaptiveThreshold(blurred, im, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 35, 10, 0.15);
       
        //AdaptiveThreshold(blurred, im, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 35, 5, 0.10);

        //threshold(blurred, im, 50, 100, THRESH_TRIANGLE);

        //threshold(blurred, im, 85, 100, THRESH_BINARY);

        //threshold(in, out, 50, 100, CALIB_CB_ADAPTIVE_THRESH);

        

        

        SimpleBlobDetector::Params params;

        // Change thresholds
       // params.minThreshold = 10;
        //params.maxThreshold = 200;

        //params.filterByColor = true;
        params.maxThreshold = 150;

        // Filter by Area.
        params.filterByArea = true;
        params.minArea = 55;
        params.maxArea = 250;

        // Filter by Circularity
        params.filterByCircularity = true;
        params.minCircularity = 0.4;

        // Filter by Convexity
        params.filterByConvexity = false;
        params.minConvexity = 0.3;
        //params.maxConvexity = 0.99;

        // Filter by Inertia
        params.filterByInertia = false;
        //params.minInertiaRatio = 0.01;

        //params.minDistBetweenBlobs = 0;

        // Storage for blobs
        std::vector<KeyPoint> keypoints;

        // Set up detector with params
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // Detect blobs
        detector->detect(im, keypoints);

        Mat im_with_keypoints, in_with_keypoints, greenfinal;
        drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(in, keypoints, in_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(green, keypoints, greenfinal, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Show blobs
        imshow("keypoints", im_with_keypoints);
        imshow("THG out", in_with_keypoints);
        imshow("CALCIUM", greenfinal);
        waitKey(0);
}