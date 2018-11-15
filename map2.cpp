#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <numeric>

#include "util.h"



using namespace Eigen;
using namespace std;

//Don't use this
VectorXf imageToWorldPoints(float u, float v, const MatrixXf& pose, const MatrixXf& proj){
    Matrix4f pose2 = Matrix4f::Zero();
    pose2.topLeftCorner<3, 4>() = pose;
    pose2(3, 3) = 1;
    MatrixXf pose3 = pose2.inverse();

    Vector3f imagePoint(u, v, 1);

    MatrixXf R = pose3.topLeftCorner<3,3>();
    VectorXf t = pose3.col(3).head(3);

    MatrixXf lMatrix = R.inverse() * proj.topLeftCorner<3,3>().inverse() * imagePoint;
    MatrixXf rMatrix = R.inverse() * t;

//    cout << pose(1,3) << endl;
    float s = (HEIGHT + pose(1, 3) + rMatrix(1, 0)) / lMatrix(1, 0);
    VectorXf realWorld = R.inverse() * (s * proj.topLeftCorner<3,3>().inverse() * imagePoint - t);

    return realWorld;
}


VectorXf imageToWorldPoints2(float u, float v, const MatrixXf& pose, const MatrixXf& proj){
    Vector3f imagePoint(u, v, 1);
    MatrixXf Kinv = proj.topLeftCorner<3,3>().inverse();
    Vector3f L = Kinv * imagePoint;
    double s =  HEIGHT / L(1);

    Vector3f camPoint = s * Kinv * imagePoint;
    Vector4f homoCam  = Vector4f::Ones();
    homoCam.head(3) = camPoint;
    Vector3f worldPoint = pose * homoCam;
    return worldPoint;
}

typedef map<long, map<int, int> > CellAssign;

int main(){
    const int size = 1000;
    const int ysize = 4000;
    const float resolution = 0.1;
    const float  minX = -50;
    const float  minY = -50;

    ifstream pose_file;
    pose_file.open(posePath);

    if (!pose_file){
        cerr << "Cannot open pose file" << endl;
    }

    MatrixXf proj = getPMatrix() ;
//    cout << proj << endl;

    bool hasPose = true;
    int frameNum = 0;

    map<pair<int, int>, CellAssign> patchMapping;
    map<pair<int, int>, CellAssign> grayPatchMapping;

    while (hasPose) {
        MatrixXf pose(3,4);
        hasPose = nextPose(pose_file, pose);
        cout << pose << endl << "------------" << endl;

        int pad = 6 - to_string(frameNum).length();
        stringstream filename, orgFilename;
        filename << labelDir << string(pad, '0') << frameNum << ".png";
        orgFilename << imgDir << string(pad, '0') << frameNum << ".png";

        cout << "Opening filename" << filename.str() << endl;

        cv::Mat img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat orgImg = cv::imread(orgFilename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Size shape = img.size();

        //cout << (int)img.at<uchar>(322, 758) << endl;

        for (int row = 0; row < shape.height; row++){
            for (int col = 0; col < shape.width; col++){
                uint8_t label = img.at<uchar>(row, col);
                uint8_t grayColor = orgImg.at<uchar>(row, col);

//                if (label == 24){
//                if (row > 300 && label==24){
                if  (row > 300){
                    //convert u,v coordinate to world coordinate
                    VectorXf realWorld = imageToWorldPoints2(col, row, pose, proj);

                    //calculate which cell inside map
                    int patchX = realWorld(0) / (resolution * size);
                    int patchY = realWorld(2) / (resolution * ysize);
                    int cellX = fmod(realWorld(0), resolution * size) / resolution;
                    int cellY = fmod(realWorld(2), resolution * ysize) / resolution;
                    cellX =(500 + cellX) % size;
//                    cellY =(500 + cellY) % size;

                    if (cellX < 0){
                        cout << "less 0" << endl;
                        continue;
                    }

                    int blabel = (label ==24)?1:0;

                    pair<int, int> gp = {patchX, patchY};
                    long cellId = cellY * size + cellX;

                    patchMapping.insert({gp, CellAssign()});
                    patchMapping[gp].insert({cellId, map<int, int>()});
                    patchMapping[gp][cellId].insert({blabel, 0});

                    grayPatchMapping.insert({gp, CellAssign()});
                    grayPatchMapping[gp].insert({cellId, map<int, int>()});
                    grayPatchMapping[gp][cellId].insert({blabel, 0});

                    patchMapping[gp][cellId][blabel]++;
                    grayPatchMapping[gp][cellId][grayColor]++;

                }
            }
        }


        //TODO: remove this
        frameNum++;
        if (frameNum > NUM_FRAMES)
            break;
    }

    for (auto &it: patchMapping){
        int x = it.first.first;
        int y = it.first.second;
        CellAssign &ca = it.second;
        CellAssign &grayCa = grayPatchMapping[it.first];

        cv::Mat mapArea = cv::Mat::zeros(ysize, size, CV_8UC3);
//        cv::Mat grayMapArea = cv::Mat::zeros(ysize, size, CV_8UC1);

        cout << mapArea.rows <<  " " << mapArea.cols << mapArea.channels() << endl;
        int maxPts = 0;

        for (auto &it2: grayCa){
            long cellId = it2.first;
            map<int, int> &pointMap = it2.second;

            int cellX = cellId % size;
            int cellY = cellId / size;

            int total = 0;
            int cnt = 0;
            for (auto &it3: pointMap){
                total += it3.first * it3.second;
                cnt += it3.second;
            }

            float avg = total*1./cnt;
            mapArea.at<cv::Vec3b>(cellY, cellX)[0] = avg;
//                    maxLabel = maxIter.first;
//                }
//            }
//
//            //uint8_t label = points[points.size()-1];
//            if (maxLabel == 24) {
//                mapArea.at<uchar>(cellX, cellY) = 255;
//            } else {
//                mapArea.at<uchar>(cellX, cellY) = 128;
//            }
        }

        for (auto &it2: ca){
            long cellId = it2.first;
            map<int, int> &pointMap = it2.second;

            int cellX = cellId % size;
            int cellY = cellId / size;

            float avg = pointMap[1]*1. / (pointMap[0] + pointMap[1]);
            mapArea.at<cv::Vec3b>(cellY, cellX)[1] = (uchar)(avg * 255);

//            int maxLabel = -1;
//            int maxCount = -1;
//
//            for(auto &maxIter: pointMap){
//                if (maxCount < maxIter.second){
//                    maxCount = maxIter.second;
//                    maxLabel = maxIter.first;
//                }
//            }
//
//            //uint8_t label = points[points.size()-1];
//            if (maxLabel == 24) {
//                mapArea.at<uchar>(cellX, cellY) = 255;
//            } else {
//                mapArea.at<uchar>(cellX, cellY) = 128;
//            }
        }

        stringstream filename, gfilename;

        filename << "/home/cy/Desktop/x_" << x << "_y_" << y << ".png";
//        gfilename << "/home/cy/Desktop/g_x_" << x << "_y_" << y << ".png";
        cv::Mat mapAreaF = cv::Mat::zeros(ysize, size, CV_8UC1);
//        cv::Mat gMapAreaF = cv::Mat::zeros(ysize, size, CV_8UC1);
        cv::flip(mapArea, mapAreaF, 0);
//        cv::flip(grayMapArea, gMapAreaF, 0);
        cv::imwrite(filename.str(), mapAreaF);
    }

    return 0;
}

