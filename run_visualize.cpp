#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "util.h"

using namespace std;

const float cellResolution = .1;
const int ROI_X = 150;
const int ROI_Y = 400;

int main(){
    ifstream poseFile;
    poseFile.open(Kitti::posePath);

    if (!poseFile){
        cerr << "Cannot open pose file" << endl;
        return -1;
    }

    bool hasPose(true);

    string mapFile = "/home/cy/Desktop/x_0_y_0.png";
    cv::Mat map = cv::imread(mapFile);
    if (map.empty()){
        cerr << "Cannot open map file." << endl;
        return -1;
    }

    while(hasPose){
        Eigen::MatrixXf pose(3,4);
        hasPose = Kitti::nextPose(poseFile, pose);

        int x = pose(0, 3) / cellResolution + map.cols/2;
        int y =  map.rows - pose(2,3) / cellResolution;
        cout << "x:" << x << " y:" << y << endl;

        int maxWidth = min(ROI_X, map.cols - x);
        int maxHeight = min(ROI_Y, map.rows - y);

        if (maxHeight == 0 || maxWidth == 0){
            continue;
        }

        cv::Mat centerOnPose(map.clone(), cv::Rect(max(x-ROI_X/2, 0), max(y-ROI_Y/2, 0), maxWidth, maxHeight) );
        cv::circle(centerOnPose, cv::Point(ROI_X/2, ROI_Y/2), 2, cv::Scalar(0, 0, 255), CV_FILLED);

        cv::namedWindow("Localization Results", cv::WINDOW_NORMAL);
        cv::imshow("Localization Results", centerOnPose);
        cv::waitKey(0);
    }

    return 0;
}