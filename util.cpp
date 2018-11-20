//
// Created by cy on 15/11/18.
//

#include "util.h"
#include <iterator>
#include <iostream>

using namespace std;

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