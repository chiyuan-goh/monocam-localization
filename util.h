//
// Created by cy on 15/11/18.
//

#ifndef MAPGEN_TEST_UTIL_H
#define MAPGEN_TEST_UTIL_H

#include <string>
#include <Eigen/Dense>
#include <fstream>

#define HEIGHT 1.65
#define NUM_FRAMES 271

const std::string labelDir = "/home/cy/Desktop/Kitti_segmented/resize/results/";
const std::string imgDir = "/ext/data/odometry/dataset/sequences/04/image_2/";
const std::string posePath = "/ext/data/odometry/dataset/poses/04.txt";

Eigen::MatrixXf getPMatrix();
bool nextPose(std::ifstream &f, Eigen::MatrixXf& pose);

#endif //MAPGEN_TEST_UTIL_H

