//
// Created by cy on 26/11/18.
//

#include "ParticleFilter.h"
#include "util.h"
#include "KDTreeVectorOfVectorsAdaptor.h"

#include <nanoflann.hpp>
#include <random>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

ParticleFilter::ParticleFilter(int n):nparticles(n){
};

cv::Mat generateLikelihoodField(cv::Mat &map){
    cout << "generating likeligood field" << endl;
    const float pixStd = 8.;


    typedef vector<vector<double> > VOV_t;
    typedef KDTreeVectorOfVectorsAdaptor<VOV_t, double, 2, nanoflann::metric_L2_Simple> my_kd_tree_t;

    const uint8_t thres = 150;
    VOV_t coordinates;
    vector<vector<double> > allQuery;

    for (int r = 0; r < map.rows; r++){
        for (int c = 0; c < map.cols; c++){
            if (map.at<cv::Vec3b>(r, c)[1] > thres){
                vector<double> rc = {1. * r, 1. * c};
                coordinates.push_back(rc);
            }
            vector<double> rc2 = {r * 1. ,c * 1.};
            allQuery.push_back(rc2);
        }
    }

    size_t n = coordinates.size();
    my_kd_tree_t matIndex(2, coordinates, 10);
    matIndex.index->buildIndex();

    const size_t numResults = 1;
    vector<size_t> ret_indexes(numResults);
    vector<double> out_dists_sqr(numResults);
//    nanoflann::KNNResultSet<double> resultSet(numResults);
//    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    cv::Mat lField = cv::Mat::zeros(map.rows, map.cols, CV_8UC1);
    int size_counted = 0;

    for (auto &queryPoint: allQuery){
        matIndex.query(&queryPoint[0], 1, &ret_indexes[0], &out_dists_sqr[0]);
//        cout << "ret_index["<<0<<"]=" << ret_indexes[0] << "x:" << coordinates[ret_indexes[0]][0] << " y:" <<
//        coordinates[ret_indexes[0]][1]  << " out_dist_sqr=" << out_dists_sqr[0] << std::endl;

        if (size_counted++ == 1000){
            cout << "counted " << size_counted << endl;
        }

        float dist = out_dists_sqr[0];
        float p  = exp( (-dist * dist) / (2 * pixStd * pixStd));
        lField.at<uchar>(queryPoint[0], queryPoint[1]) = (uchar) (p * 255);
    }

    cout << "completed likelihood field..." << endl;
    cv::imwrite("/source/likelihood_field.png", lField);
    return lField;
}

void ParticleFilter::init(Eigen::Matrix4f &initialPose) {
    particles = std::vector<MatrixXf>(nparticles);
    weights = std::vector<double>(nparticles);

    float xyBounds = 1.5;
    float yawBound = 15. * M_PI / 180.;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posDis(-xyBounds, xyBounds);
    std::uniform_real_distribution<> headingDis(-yawBound, yawBound);

    Matrix4f hPose = initialPose;

    for (int i = 0; i < weights.size(); i++){
        if (i == 0){
            particles[i] = hPose;
            weights[i] = 1.;
        }
        else {
            weights[i] = 1.;

            float xDiff = posDis(gen);
            float zDiff = posDis(gen);
            float yawDiff = headingDis(gen);
            cout << "Initialized particle with " << xDiff << " " << zDiff << " " << yawDiff * 180 / M_PI << endl;

            MatrixXf particle = Matrix4f::Identity(4, 4);
            particle(0, 3) = 2;//xDiff;
            particle(2, 3) = 2;//zDiff;

            AngleAxisf m = AngleAxisf(45 * M_PI/180., Vector3f::UnitY());
//            AngleAxisf m = AngleAxisf(yawDiff, Vector3f::UnitY());
            particle.topLeftCorner<3, 3>() = m.matrix();
            particle = particle * hPose;

            particles[i] = particle;
        }
    }
}

void ParticleFilter::update(cv::Mat &img) {
    const int lookAhead = 50;
    Kitti::CamerasInfo cams;
    vector<long> xy;

    for (int v = lookAhead; v < img.rows; v++){
        for (int u = 0; u < img.cols; u++){
            if (img.at<uchar>(v, u) == 24){
                xy.push_back(v * img.cols + u);
            }
        }
    }

    MatrixXi compare = MatrixXi::Zero(1500, 2);

    for (int pidx = 0; pidx < particles.size(); pidx++){
        double prob = 1.;
        double lp = .0001;
        Matrix4f pPose = (Matrix4f)particles[pidx];
        int t1c = 0, t2c = 0, t3c = 0;
        int counter = 0;

        cout << pPose << endl;
        for (auto &l: xy){
            float v = l / img.cols;
            float u = l % img.cols;
            Vector3f mapPoint = imageToWorldRoadPoint(u, v, pPose, cams);
            int patchX, patchY, cellX, cellY;
            worldPointToGrid(mapPoint(0), mapPoint(2), patchX, patchY, cellX, cellY);

            if (patchX != 0 || patchY != 0){
                prob += log(lp);
                t1c++;
                cout << mapPoint << endl;
                Vector3f mapPoint = imageToWorldRoadPoint(u, v, pPose, cams);
            }
            else if (cellX < 0 || cellX >= map.cols || cellY < 0 || cellY >= map.rows){
                prob += log(lp);
                t2c++;
            }
            else {
                cout << mapPoint << endl;
                prob += log(map.at<uchar>(cellY, cellX) / 255. +  1./ ((img.rows - lookAhead) * img.cols));
                t3c++;
            }

            if (counter < 1500)
            compare(counter++, pidx) = cellX * 500 + cellY;
//            cout << prob << endl;

        }

        cout << "t1:" << t1c << " t2:" << t2c << " t3:" << t3c << endl;
        cout << "setting particle " << pidx << ": " << prob << endl;
        weights[pidx] += prob;
    }

//    cout << compare << endl;
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

    double maxVal = *std::max_element(weights.begin(), weights.end());
    vector<double> wtmp;
    for (auto &w: weights){
        wtmp.push_back(exp(w - maxVal));
    }

    double total = std::accumulate(wtmp.begin(), wtmp.end(), 0.);


    double c = wtmp[0]/total;
    double r = rand(gen);
    int i = 0;

    for (int m = 0; m < wtmp.size(); m++){
        double u = r + m / wtmp.size();
        while (u > c){
            i++;
            c += wtmp[i]/total;
        }
        tmp.push_back(particles[i]);
    }

    particles.swap(tmp);
    fill(weights.begin(), weights.end(), 1.);
}