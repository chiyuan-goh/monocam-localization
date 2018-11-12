#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <numeric>

#define HEIGHT 1.65

using namespace Eigen;
using namespace std;

bool nextPose(ifstream &f, Eigen::MatrixXf& pose){
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

MatrixXf getPMatrix(const string& filename){
    MatrixXf P(3, 4);
    P << 7.070912000000e+02, 0.000000000000e+00,  6.018873000000e+02, 4.688783000000e+01,
        0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01,
        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03;
    return P;
}


VectorXf imageToWorldPoints(float u, float v, const MatrixXf& pose, const MatrixXf& proj){
    Matrix4f pose2 = Matrix4f::Zero();
    pose2.topLeftCorner<3, 4>() = pose;
    pose2(3, 3) = 1;
    MatrixXf pose3 = pose2.inverse();


    Vector3f imagePoint(u, v, 1);

    MatrixXf R = pose3.topLeftCorner<3,3>();
    VectorXf t = pose3.col(3).head(3);

    MatrixXf lMatrix = R.inverse() * proj.topLeftCorner<3,3>().inverse() * imagePoint;
    MatrixXf rMatrix = R.inverse() * t;

    float s = (HEIGHT + rMatrix(1, 0)) / lMatrix(1, 0);
    VectorXf realWorld = R.inverse() * (s * proj.topLeftCorner<3,3>().inverse() * imagePoint - t);

    return realWorld;
}


typedef map<long, vector<uint8_t> > CellAssign;

int main(){
    const int size = 1000;
    const float resolution = 0.1;
    const float  minX = -50;
    const float  minY = -50;

    string labelDir = "/home/cy/Desktop/Kitti_segmented/resize/results/";

    ifstream pose_file;
    pose_file.open("/ext/data/odometry/dataset/poses/04.txt");

    if (!pose_file){
        cerr << "Cannot open pose file" << endl;
    }

    MatrixXf proj = getPMatrix("/ext/data/odometry/dataset/sequences/04/calib.txt") ;
//    cout << proj << endl;

    bool hasPose = true;
    int frameNum = 0;

    map<pair<int, int>, CellAssign> patchMapping;

    while (hasPose) {
        MatrixXf pose(3,4);
        hasPose = nextPose(pose_file, pose);
        cout << pose << endl << "------------" << endl;

        int pad = 6 - to_string(frameNum).length();
        stringstream filename;
        filename << labelDir << string(pad, '0') << frameNum << ".png";
        cout << "Opening filename" << filename.str() << endl;

        cv::Mat img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        cv::Size shape = img.size();

        //cout << (int)img.at<uchar>(322, 758) << endl;

        for (int row = 0; row < shape.height; row++){
            for (int col = 0; col < shape.width; col++){
                uint8_t label = img.at<uchar>(row, col);

//                if (label == 24){
                    //convert u,v coordinate to world coordinate
                    VectorXf realWorld = imageToWorldPoints(col, row, pose, proj);

                    //calculate which cell inside map
                    int patchX = realWorld(0) / (resolution * size);
                    int patchY = realWorld(2) / (resolution * size);
                    int cellX = fmod(realWorld(0), resolution * size) / resolution;
                    int cellY = fmod(realWorld(2), resolution * size) / resolution;
                    cellX =(500 + cellX) % size;
//                    cellY =(500 + cellY) % size;

                    if (cellX < 0){
                        cout << "less 0" << endl;
                        continue;
                    }

                    pair<int, int> gp = {patchX, patchY};
                    long cellId = cellX * size + cellY;

                    patchMapping.insert({gp, CellAssign()});
                    patchMapping[gp].insert({cellId, vector<uint8_t>()});
                    patchMapping[gp][cellId].push_back(label);

//                }
            }
        }


        //TODO: remove this
        frameNum++;
        if (frameNum > 1)
            break;
    }

    for (const auto &it: patchMapping){
        int x = it.first.first;
        int y = it.first.second;
        const CellAssign &ca = it.second;

        cv::Mat mapArea = cv::Mat::zeros(size, size, CV_8UC1);

        int maxPts = 0;
        for (const auto &it2: ca){
            const vector<uint8_t> &points = it2.second;
            if (points.size() > maxPts){
                maxPts = points.size();
            }
        }


        for (const auto &it2: ca){
            long cellId = it2.first;
            const vector<uint8_t> &points = it2.second;

            int cellX = cellId % size;
            int cellY = cellId / size;

            uint8_t label = points[points.size()-1];
            if (label == 24) {
                mapArea.at<uchar>(cellX, cellY) = 255;
            } else {
                mapArea.at<uchar>(cellX, cellY) = 128;
            }
        }

        stringstream filename;

        filename << "/home/cy/Desktop/x" << x << "_y_" << y << ".png";
        cv::Mat mapAreaF = cv::Mat::zeros(size, size, CV_8UC1);
        cv::flip(mapArea, mapAreaF, 0);
        cv::imwrite(filename.str(), mapAreaF);
    }

    return 0;
}

