#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "ParticleFilter.h"

using namespace std;

const float cellResolution = .1;
const int ROI_X = 150;
const int ROI_Y = 400;

cv::Mat poseOnMap(cv::Mat& map, MatrixXf &gt, ParticleFilter &mcl){
    const cv::Scalar gtColor(0, 0, 255), pColor(255, 0, 0);
    const int pointSize = 1, headingLine = 8;

    float x, y, heading;
    mat2xyh(gt, x, y, heading);

    int cellX = x / cellResolution + map.cols/2;
    int cellY = y / cellResolution;
    heading = atan2(gt(1, 0), gt(0,0));

//    cout << "x:" << x << " y:" << y << endl;

    cv::Mat clone = map.clone();
    cv::Point gtPoint(cellX, cellY);
    int lx = gtPoint.x + headingLine * sin(heading);
    int ly = gtPoint.y + headingLine * cos(heading);

    cv::circle(clone, gtPoint, pointSize, gtColor, 2);
    cv::line(clone, gtPoint, cv::Point(lx, ly), gtColor, 1);

    for (auto &pose : mcl.particles){
        float px, py, heading;
        mat2xyh(pose, px, py, heading);

        int cellX = px / cellResolution + map.cols/2;
        int cellY = py / cellResolution;

        int lx = cellX + headingLine * sin(heading);
        int ly = cellY + headingLine * cos(heading);

        cv::circle(clone, cv::Point(cellX, cellY), pointSize, pColor, 2);
        cv::line(clone, cv::Point(cellX, cellY), cv::Point(lx, ly), pColor, 1);
    }

    int maxWidth = min(ROI_X, map.cols - cellX);
    int maxHeight = min(ROI_Y, map.rows - cellY);
    if (maxHeight == 0 || maxWidth == 0){
        return map.clone();
    }

    cv::Mat centerOnPose(clone, cv::Rect(max(gtPoint.x - ROI_X/2, 0), max(gtPoint.y - ROI_Y/2, 0), maxWidth, maxHeight));

    return centerOnPose;
}

int main(){
    ifstream poseFile;
    poseFile.open(Kitti::posePath);

    if (!poseFile){
        cerr << "Cannot open pose file" << endl;
        return -1;
    }

    bool hasPose = true;

    string mapFile = "/source/map.png";
    cv::Mat map = cv::imread(mapFile);
    if (map.empty()){
        cerr << "Cannot open map file." << endl;
        return -1;
    }

    //cv::Mat mapAreaF = cv::Mat::zeros(map.rows, map.cols, CV_8UC1);
    cv::flip(map, map, 0);

    int frame = 0;
    ParticleFilter mcl(20);

    Matrix4f curPose, prevPose;

    while(hasPose){
        Eigen::MatrixXf pose(3,4);
        hasPose = Kitti::nextPose(poseFile, pose);
        Matrix4f pose4  = Matrix4f::Identity();
        pose4.topLeftCorner<3,4>() = pose;

        if (frame == 0){
            //first pose, initialMCL
            mcl.init(pose4);
            curPose = pose4;
        } else {
            prevPose = curPose;
            curPose = pose4;
            mcl.predict(prevPose, curPose);
        }

        if (mcl.weightsDegenerate()){
            cout << "low effective sample size! performing resampling..." << endl;
            mcl.resample();
        }

        cv::Mat visualize = poseOnMap(map, pose, mcl);
        cv::flip(visualize, visualize, 0);
        cv::namedWindow("Localization Results", cv::WINDOW_NORMAL);
        cv::imshow("Localization Results", visualize);
        cv::waitKey(0);

        frame++;
    }

    return 0;
}