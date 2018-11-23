//
// Created by cy on 15/11/18.
//

#include "util.h"
#include <iterator>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
/*
 * For instance, to transform a point in road coordinates to the left color image
(= image_2 for which semantic labels are provided in label_2), the 3D point X
transforms to pixel x as

x = P2 * R0_rect * Tr_cam_to_road^-1 * X

where R0_rect and Tr_cam_to_road have been extended to 4x4 matrices by adding
a fourth row (with 1 as the last element and zeros elsewhere) and a fourth
column (for R0_rect only).
 */

Kitti::CamerasInfo::CamerasInfo() {
    height = HEIGHT;
    const float pitch = 4 * M_PI / 180;

    P2_Rect = MatrixXf(3, 4);
    P2_Rect << 7.070912000000e+02, 0.000000000000e+00,  6.018873000000e+02, 4.688783000000e+01,
                0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01,
                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03;

    T_Cam0Rect_Velodyne << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                            -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                            9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                            0, 0, 0, 1;

//    T_Cam0Unrect_Velodyne << 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
//                            1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
//                            9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01,
//                                       0,           0,              0,             1;
    T_Cam0Unrect_Velodyne << 7.027555e-03, -9.999753e-01, 2.599616e-05,  -7.137748e-03,
                            -2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02,
                            9.999728e-01, 7.027479e-03, -2.255075e-03,   -3.336324e-01,
                                       0,           0,              0,             1;

    T_Cam0Unrect_Road = Matrix4f::Identity();
    T_Cam0Unrect_Road(1, 3) = height;
    AngleAxisf m = AngleAxisf(pitch, Vector3f::UnitX());
    T_Cam0Unrect_Road.topLeftCorner<3,3>() = m.matrix();

    T_Cam0Rect_Cam2Rect = Matrix4f::Identity();
    T_Cam0Rect_Cam2Rect(0, 3) = -1 * P2_Rect(0, 3) / P2_Rect(0, 0);

    K2 = P2_Rect.topLeftCorner<3,3>();

//    R0_Rect << 9.999239e-01, 9.837760e-03, -7.445048e-03, 0,
//              -9.869795e-03, 9.999421e-01, -4.278459e-03, 0,
//               7.402527e-03, 4.351614e-03,  9.999631e-01, 0,
//                          0,            0,             0, 1;
    R0_Rect << 9.999280e-01, 8.085985e-03, -8.866797e-03, 0,
                -8.123205e-03, 9.999583e-01, -4.169750e-03, 0,
                8.832711e-03, 4.241477e-03, 9.999520e-01, 0,
                          0,            0,             0, 1;
}


Eigen::MatrixXf Kitti::getPMatrix(){
    Eigen::MatrixXf P(3, 4);
    P << 7.070912000000e+02, 0.000000000000e+00,  6.018873000000e+02, 4.688783000000e+01,
            0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03;
    return P;
}

Eigen::MatrixXf Kitti::getVelodyneToCam(){
    Eigen::MatrixXf Tr(4, 4);
    Tr << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
    -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
    9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
    0, 0, 0, 1;
    return Tr;
}

bool Kitti::nextPose(std::ifstream &f, Eigen::MatrixXf& pose){
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

void Kitti::readCSV(std::ifstream &f, vector<MPoint> &points){
    while (f.peek() != EOF) {
        string x, y, z, i;
        string r, g, b;
        string l;

        getline(f, x, ',');
        getline(f, y, ',');
        getline(f, z, ',');
        getline(f, i);

        uint8_t intensity255 = stof(i) * 255;

        MPoint p = {stof(x), stof(y), stof(z), intensity255};
        points.push_back(p);
    }

    f.close();
}

//reads the kitti velodyne bin file
bool Kitti::readBinary(string filename, vector<MPoint> &points){
    ifstream file(filename, std::ifstream::ate | std::ifstream::binary);

    if (!file.is_open()){
        return false;
    } else {
        int size = file.tellg();
        cout << "size is " << size << endl;

        char* memblock = new char[size];
        file.seekg(0, ios::beg);
        file.read(memblock, size);
        file.close();

        float* tofloat = (float*)memblock;

        int numPoints  = size / (4 * sizeof(float));
//        cout << "number of points:" << numPoints << endl;
        for (int i = 0; i < numPoints; i++){
//            std::cout << "point: " << tofloat[i * 4] << " " << tofloat[i*4+1] << " " << tofloat[i*4+2] << " " <<  tofloat[i*4+3] << std::endl;
            MPoint p = {tofloat[i * 4], tofloat[i*4+1], tofloat[i*4+2], uint8_t (tofloat[i*4+3] * 255)};
            points.push_back(p);
        }

        delete memblock;
        return true;
    }
}


bool Kitti::readBinary(string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr pc){
    ifstream file(filename, std::ifstream::ate | std::ifstream::binary);

    if (!file.is_open()){
        return false;
    } else {
        int size = file.tellg();
        cout << "size is " << size << endl;

        char* memblock = new char[size];
        file.seekg(0, ios::beg);
        file.read(memblock, size);
        file.close();

        float* tofloat = (float*)memblock;

        int numPoints  = size / (4 * sizeof(float));
        cout << "number of points:" << numPoints << endl;
        for (int i = 0; i < numPoints; i++){
//            std::cout << "point: " << tofloat[i * 4] << " " << tofloat[i*4+1] << " " << tofloat[i*4+2] << " " <<  tofloat[i*4+3] << std::endl;
            pcl::PointXYZI p;
            p.x = tofloat[i * 4];
            p.y = tofloat[i*4+1];
            p.z = tofloat[i*4+2];
            p.intensity = tofloat[i*4+3] * 255;
            pc->push_back(p);
        }

        delete memblock;
        return true;
    }
}




Eigen::MatrixXf MPointsToHomoCoordinates(const std::vector<MPoint> &points){
    Eigen::MatrixXf homoPoints = Eigen::MatrixXf::Ones(4, points.size());
    for (int i = 0; i < points.size(); i++){
        const MPoint &p = points[i];
        homoPoints(0, i) = p.x;
        homoPoints(1, i) = p.y;
        homoPoints(2, i) = p.z;
    }
    return homoPoints;
}