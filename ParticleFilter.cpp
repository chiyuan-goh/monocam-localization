//
// Created by cy on 26/11/18.
//

#include "ParticleFilter.h"
#include "util.h"

#include <random>
#include <iostream>

using namespace std;
using namespace Eigen;

void mat2xyzrph(Matrix4f mat, float &x, float &y, float &z, float &r, float &p, float &h){

}

ParticleFilter::ParticleFilter(int n):nparticles(n){
};

void ParticleFilter::init(Eigen::Matrix4f &initialPose) {
    particles = std::vector<MatrixXf>(nparticles);
    weights = std::vector<float>(nparticles);

    float xyBounds = 1.5;
    float yawBound = 15. * M_PI / 180.;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posDis(-xyBounds, xyBounds);
    std::uniform_real_distribution<> headingDis(-yawBound, yawBound);

    Matrix4f hPose = initialPose;

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

void ParticleFilter::predict(Matrix4f &prevPose, Matrix4f &curPose,
    float a1, float a2, float a3, float a4){
    MatrixXf diff = prevPose.inverse() * curPose;
    float prevx, prevy, prevyaw, curx, cury, curyaw;
    mat2xyh(prevPose, prevx, prevy, prevyaw);
    mat2xyh(curPose, curx, cury, curyaw);
    float diffx = curx - prevx, diffy = cury - prevy, diffyaw = curyaw - prevyaw;

    float delRot1 = atan2(diffx, diffy) - prevyaw;
    float delTrans = sqrt(diffx * diffx + diffy * diffy) - prevyaw;
    float delRot2 = diffyaw - delRot1;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> rot1Gauss(0, a1 * delRot1 * delRot1 + a2 * delTrans * delTrans);
    normal_distribution<> transGauss(0,
            a3  * delTrans * delTrans +  a4 * delRot1 * delRot1 + a4 * delRot2 * delRot2);
    normal_distribution<> rot2Gauss(0, a1 * delRot2 * delRot2 + a2 * delTrans * delTrans);

    for (auto &particle: particles){
        float rot1Hat = delRot1- rot1Gauss(gen);
        float transHat = delTrans - transGauss(gen);
        float rot2Hat = delRot2 - rot2Gauss(gen);

        float px, py, pyaw;
        mat2xyh(particle, px, py, pyaw);

        px += transHat * sin(pyaw + rot1Hat);
        py += transHat * cos(pyaw + rot1Hat);
        pyaw += rot1Hat + rot2Hat;

        particle(0, 3) = px;
        particle(2, 3) = py;
        AngleAxisf heading =  AngleAxisf(pyaw, Vector3f::UnitY());
        particle.topLeftCorner<3,3>() = heading.matrix();
    }

}

bool ParticleFilter::weightsDegenerate() {
    float nEffective = 0;
    for (auto w: weights){
        nEffective += w * w;
    }
    nEffective = 1 / nEffective;

    return nEffective < (2/3 * weights.size());
}

void ParticleFilter::resample() {
    //low variance sampling
    //probabilistic robotics page 110 table 4.4
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> rand(0, 1./weights.size());

    vector<MatrixXf> tmp;

    double c = weights[0];
    double r = rand(gen);
    int i = 0;

    for (int m = 0; m < weights.size(); m++){
        double u = r + m / weights.size();
        while (u > c){
            i++;
            c += weights[i];
        }
        tmp.push_back(particles[i]);
    }

    particles.swap(tmp);
    fill(weights.begin(), weights.end(), 1.);
}