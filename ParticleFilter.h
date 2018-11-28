//
// Created by cy on 26/11/18.
//

#ifndef MAPGEN_TEST_PARTICLEFILTER_H
#define MAPGEN_TEST_PARTICLEFILTER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>


cv::Mat generateLikelihoodField(cv::Mat &map);

struct ParticleFilter {
    std::vector<Eigen::MatrixXf> particles;
    std::vector<double> weights;
    cv::Mat map;
    int nparticles;

    ParticleFilter(int n);

    void init(Eigen::Matrix4f& initialPose);
    void update(cv::Mat& img);
    void resample();
    bool weightsDegenerate();
    void predict(Eigen::Matrix4f &prevPose, Eigen::Matrix4f &curPose,
                 float a1=0.05, float a2 = 0.005, float a3 = 0.005, float a4 = 0.05);
};


#endif //MAPGEN_TEST_PARTICLEFILTER_H
