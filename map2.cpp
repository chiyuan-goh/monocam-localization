#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <numeric>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>

#include "util.h"

#define CELL_RESOLUTION 0.1

using namespace Eigen;
using namespace std;

typedef map<long, map<int, int> > CellAssign;
typedef pcl::PointCloud<pcl::PointXYZI> IPC;

namespace {
    const int size = 1000;
    const int ysize = 4000;
    const float resolution = 0.1;
    const float minX = -50;
    const float minY = -50;
}

//TODO: assign the right height to each cell to see if there is improvement in the map generation.
//TODO: instead of generating map for all frames, try to skip frame according to paper
//TODO: dont hard code map alignment issue/check hyungjin code for references.


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

void worldPointToGrid(float x, float y, int &patchX, int &patchY, int &cellX, int &cellY){
    patchX = x / (resolution * size);
    patchY = y / (resolution * ysize);
    cellX = fmod(x, resolution * size) / resolution;
    cellY = fmod(y, resolution * ysize) / resolution;
    cellX =(500 + cellX) % size;
}

void inline updateMapper(map<pair<int, int>, CellAssign> &mapper, int &patchX, int &patchY, int &cellX, int &cellY, int key){
    pair<int, int> gp = {patchX, patchY};
    long cellId = cellY * size + cellX;

    mapper.insert({gp, CellAssign()});
    mapper[gp].insert({cellId, map<int, int>()});
    mapper[gp][cellId].insert({key, 0});
    mapper[gp][cellId][key]++;
}

void filterPointCloud(IPC::Ptr src, IPC::Ptr dst, MatrixXf &pose){
    dst->clear();

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr outputNorm(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(src);
    ne.setRadiusSearch(.3);
    ne.compute(*outputNorm);

    const float nTol = 0.9, heightTol = 1.35;

    for (int i = 0; i < outputNorm->size(); ++i){
        pcl::Normal &n = outputNorm->points[i];
        if (fabs(n.normal_y) >= nTol && src->points[i].y >= heightTol){
            dst->push_back(src->points[i]);
        }
    }

    Matrix4f P4x4 = Matrix4f::Identity();
    P4x4.topLeftCorner<3,4>() = pose;

    pcl::transformPointCloud(*dst, *dst, P4x4);
}

int main(){


    ifstream pose_file;
    pose_file.open(Kitti::posePath);

    if (!pose_file){
        cerr << "Cannot open pose file" << endl;
    }

    MatrixXf proj = Kitti::getPMatrix() ;
//    cout << proj << endl;

    bool hasPose = true;
    int frameNum = 0;

    map<pair<int, int>, CellAssign> patchMapping, grayPatchMapping, lidarMapping;

    while (hasPose) {
        MatrixXf pose(3,4);
        hasPose = Kitti::nextPose(pose_file, pose);

        int pad = 6 - to_string(frameNum).length();
        stringstream filename, orgFilename, veloFilename;
        filename << Kitti::labelDir << string(pad, '0') << frameNum << ".png";
        orgFilename << Kitti::imgDir << string(pad, '0') << frameNum << ".png";
        veloFilename << Kitti::veloPath << string(pad, '0') << frameNum << ".bin";

        cout << "Opening veloFilename" << veloFilename.str() << endl;

        cv::Mat img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat orgImg = cv::imread(orgFilename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Size shape = img.size();
        IPC::Ptr pc(new IPC()), filteredPC(new IPC());
        Kitti::readBinary(veloFilename.str(), pc);
//        cout << "number of lidar " << pc->size() << endl;
        pcl::transformPointCloud(*pc, *pc, Kitti::getVelodyneToCam());
        filterPointCloud(pc, filteredPC, pose);

        for (auto &point: filteredPC->points){
            int cellX, cellY, patchX, patchY;
            worldPointToGrid(point.x, point.z, patchX, patchY, cellX, cellY);
            if (cellX < 0 || cellX >= size || cellY < 0 || cellY > ysize)
                continue;
            updateMapper(lidarMapping, patchX, patchY, cellX, cellY, point.intensity);
        }

        //get only row 300 and below
        for (int row = 300; row < shape.height; row++){
            for (int col = 0; col < shape.width; col++){
                uint8_t label = img.at<uchar>(row, col);
                uint8_t grayColor = orgImg.at<uchar>(row, col);

                //convert u,v coordinate to world coordinate
                VectorXf realWorld = imageToWorldPoints2(col, row, pose, proj);

                int cellX, cellY, patchX, patchY;
                worldPointToGrid(realWorld(0), realWorld(2), patchX, patchY, cellX, cellY);

                if (cellX < 0){
                    cout << "less 0" << endl;
                    continue;
                }

                int blabel = (label ==24)?1:0;
                updateMapper(patchMapping, patchX, patchY, cellX, cellY, blabel);
                updateMapper(grayPatchMapping, patchX, patchY, cellX, cellY, grayColor);
/*
                    pair<int, int> gp = {patchX, patchY};
                    long cellId = cellY * size + cellX;

                    patchMapping.insert({gp, CellAssign()});
                    patchMapping[gp].insert({cellId, map<int, int>()});
                    patchMapping[gp][cellId].insert({blabel, 0});

                    grayPatchMapping.insert({gp, CellAssign()});
                    grayPatchMapping[gp].insert({cellId, map<int, int>()});
                    grayPatchMapping[gp][cellId].insert({blabel, 0});

                    patchMapping[gp][cellId][blabel]++;
                    grayPatchMapping[gp][cellId][grayColor]++;*/
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
        CellAssign &lidarCa = lidarMapping[it.first];

        cv::Mat mapArea = cv::Mat::zeros(ysize, size, CV_8UC3);
//        cv::Mat grayMapArea = cv::Mat::zeros(ysize, size, CV_8UC1);

        cout << mapArea.rows <<  " " << mapArea.cols << mapArea.channels() << endl;
        int maxPts = 0;

        for (auto &it2: lidarCa){
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
            mapArea.at<cv::Vec3b>(cellY, cellX)[2] = avg;
        }

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
//            mapArea.at<cv::Vec3b>(cellY, cellX)[1] = (uchar)(avg * 255);

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

        filename << "/source/3c_x_" << x << "_y_" << y << ".png";
//        gfilename << "/home/cy/Desktop/g_x_" << x << "_y_" << y << ".png";
        cv::Mat mapAreaF = cv::Mat::zeros(ysize, size, CV_8UC1);
//        cv::Mat gMapAreaF = cv::Mat::zeros(ysize, size, CV_8UC1);
        cv::flip(mapArea, mapAreaF, 0);
//        cv::flip(grayMapArea, gMapAreaF, 0);
        cv::imwrite(filename.str(), mapAreaF);
    }

    return 0;
}

