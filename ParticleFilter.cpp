//
// Created by cy on 26/11/18.
//

#include "ParticleFilter.h"
#include <random>
#include <iostream>

using namespace std;
using namespace Eigen;

void mat2xyzrph(Matrix4f mat, float &x, float &y, float &z, float &r, float &p, float &h){

}

ParticleFilter::ParticleFilter(int n):nparticles(n){
};

void ParticleFilter::init(Eigen::MatrixXf &initialPose) {
    particles = std::vector<MatrixXf>(nparticles);
    weights = std::vector<float>(nparticles);

    float xyBounds = 1.5;
    float yawBound = 45. * M_PI / 180.;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posDis(-xyBounds, xyBounds);
    std::uniform_real_distribution<> headingDis(-yawBound, yawBound);

    Matrix4f hPose = Matrix4f::Identity();
    hPose.topLeftCorner<3,4>() = initialPose;

    for (int i = 0; i < weights.size(); i++){
        weights[i] = 1.;

        float xDiff = posDis(gen);
        float zDiff = posDis(gen);
        float yawDiff = headingDis(gen);
        cout << "Initialized particle with " << xDiff << " " << zDiff << " " << yawDiff * 180/M_PI << endl;

        MatrixXf particle = Matrix4f::Identity(4,4);
        particle(0, 3) = xDiff;
        particle(2, 3) = zDiff;

        AngleAxisf m = AngleAxisf(yawDiff, Vector3f::UnitY());
        particle.topLeftCorner<3,3>() = m.matrix();
        particle = particle * hPose;

        particles[i] = particle;
    }
}

void ParticleFilter::update(cv::Mat &img) {

}

void ParticleFilter::predict(Matrix4f &prevPose){

}