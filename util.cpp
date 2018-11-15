//
// Created by cy on 15/11/18.
//

#include "util.h"


Eigen::MatrixXf getPMatrix(){
    Eigen::MatrixXf P(3, 4);
    P << 7.070912000000e+02, 0.000000000000e+00,  6.018873000000e+02, 4.688783000000e+01,
            0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03;
    return P;
}

bool nextPose(std::ifstream &f, Eigen::MatrixXf& pose){
    bool hasPose = false;

    if(f.peek() != EOF) {
        hasPose = true;
        for (int i = 0; i < 12; ++i) {
            float v;
            f >> v;
            int y = i / 4;
            int x = i % 4;
            pose(y, x) = v;
        }
    }
    return hasPose;
}
