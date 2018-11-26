//
// Created by cy on 26/11/18.
//

#ifndef MAPGEN_TEST_PARTICLEFILTER_H
#define MAPGEN_TEST_PARTICLEFILTER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

struct ParticleFilter {
    std::vector<Eigen::MatrixXf> particles;
    std::vector<float> weights;
    int nparticles;

    ParticleFilter(int n);

    void init(Eigen::MatrixXf& initialPose);
    void update(cv::Mat& img);
    void predict(Eigen::Matrix4f &prevPose);
};


#endif //MAPGEN_TEST_PARTICLEFILTER_H
