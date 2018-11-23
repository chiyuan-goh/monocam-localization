//
// Created by cy on 15/11/18.
//

#ifndef MAPGEN_TEST_UTIL_H
#define MAPGEN_TEST_UTIL_H

#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <pcl/common/common_headers.h>

#define HEIGHT 1.65
#define NUM_FRAMES 30

using namespace Eigen;

struct MPoint{
    float x;
    float y;
    float z;
    uint8_t i;
};



Eigen::MatrixXf MPointsToHomoCoordinates(const std::vector<MPoint> &points);


namespace Kitti {
    const std::string labelDir = "/data/Kitti_segmented/resize/results/";
    const std::string imgDir = "/data/odometry/dataset/sequences/04/image_2/";
    const std::string posePath = "/data/odometry/dataset/poses/04.txt";
    const std::string veloPath = "/data/odometry/dataset/sequences/04/velodyne/";

    struct CamerasInfo{
        CamerasInfo();
         Matrix4f T_Cam0Rect_Velodyne;
         Matrix4f T_Cam0Unrect_Velodyne;
         Matrix4f T_Cam0Rect_Cam2Rect;
         MatrixXf P2_Rect;
         Matrix4f R0_Rect;
         Matrix3f K2;
         float height;
         Matrix4f T_Cam0Unrect_Road;
    };

    Eigen::MatrixXf getPMatrix();
    Eigen::MatrixXf getVelodyneToCam();
    bool nextPose(std::ifstream &f, Eigen::MatrixXf &pose);
    void readCSV(std::ifstream &f, std::vector<MPoint> &points);
    bool readBinary(std::string filename, std::vector<MPoint> &points);
    bool readBinary(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr pc);
};

#endif //MAPGEN_TEST_UTIL_H

