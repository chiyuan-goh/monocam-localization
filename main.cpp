#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <math.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

struct MPoint{
    float x;
    float y;
    float z;
    uint8_t i;
};

using namespace std;
using namespace cv;

typedef map<long, vector<uint8_t>> CellAssign;

void readCSV(ifstream &f, vector<MPoint> &points){
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

int main() {
    //try 1000 by 1000 first.
    const int size = 1000;
    const float resolution = 0.1;
    const float min_x = -79.78397146554197;
    const float min_y = -79.9274194808028;

    cout << CV_MAJOR_VERSION << " " << CV_MINOR_VERSION << endl;

    ifstream infile;
    infile.open("/ext/data/odometry/dataset/all_noskip_xyzi.xyz");
//    infile.open("/ext/data/odometry/dataset/all_noskip_xyzi_10rows.xyz");

    if (!infile){
        cerr << "fucking cannot open the file" << endl;
        return -1;
    }

    //map<long, vector<uint8_t > > cell_alloc;
    map<pair<int, int>, CellAssign> patch_mapping;

    vector<MPoint> points;
    readCSV(infile, points);
    cout << "number of points: " << points.size() << endl;
    infile.close();

    float height_threshold = -1000;
    vector<MPoint> groundPoints;
    for (auto &p: points){
        if (p.y > height_threshold){
            groundPoints.push_back(p);
        }
    }

    cout << "Num ground points: " << groundPoints.size() << endl;

    for (auto &p: groundPoints){
        int patch_x = (p.x - min_x) / (resolution * size);
        int patch_y = (p.z - min_y) / (resolution * size);

        int cell_x = fmod(p.x - min_x, resolution * size) / resolution;
        int cell_y = fmod(p.z - min_y, resolution * size) / resolution;

        pair<int, int> gp ={patch_x, patch_y};

        long cell_id = cell_x * size + cell_y;

        patch_mapping.insert({gp, CellAssign()});
        CellAssign &cell_mapping = patch_mapping[gp];

        cell_mapping.insert({cell_id, vector<uint8_t>()});
        cell_mapping[cell_id].push_back(p.i);
    }

    for (map<pair<int, int>, CellAssign>::const_iterator patch_it=patch_mapping.begin(); patch_it != patch_mapping.end(); ++patch_it){
        pair<int, int> pair = patch_it->first;
        int patch_x = pair.first;
        int patch_y = pair.second;

        cv::Mat meanImg =  cv::Mat::zeros(size, size, CV_8UC1);
//        Eigen::MatrixXd meanImg = Eigen::MatrixXd::Zero(size, size);

        const CellAssign &cell_mapping = patch_it->second;

        for (CellAssign::const_iterator it = cell_mapping.begin(); it != cell_mapping.end(); ++it){
            long cell_id = it->first;
            int cell_x = cell_id % size;
            int cell_y = cell_id / size;

            const vector<uint8_t> &intens = it->second;

            if (intens.size() != 0) {
                long sum = accumulate(intens.begin(), intens.end(), 0);
                long avg = sum / intens.size();

                long sq_sum = std::inner_product(intens.begin(), intens.end(), intens.begin(), 0);
                double stdev = std::sqrt(sq_sum / intens.size() * .1 - avg * avg);

                cout << cell_y << " " << cell_x << " " << avg << endl;
                meanImg.at<uchar>(cell_y, cell_x) = (uint8_t)avg;
//                meanImg(cell_y, cell_x) = (int)avg;
            }
        }

        stringstream filename;
        filename << "/home/cy/Desktop/x_" << patch_x << "_y_" << patch_y << ".png";
        imwrite(filename.str(), meanImg);
//        meanImg.release();
    }

//    cv::Mat meanImg =  cv::Mat::zeros(size, size, CV_8UC1);
//
//    for (auto &it:gridAlloc){
//        long id = it.first;
//        int x = id % size;
//        int y = id / size;
//
//        vector<uint8_t> &intens = it.second;
//        if (intens.size() == 0)
//            continue;
//        long sum = accumulate(intens.begin(), intens.end(), 0);
//        long avg = sum / intens.size();
//
//        long sq_sum = std::inner_product(intens.begin(), intens.end(), intens.begin(), 0);
//        double stdev = std::sqrt(sq_sum / intens.size()*.1 - avg * avg);
//
//        meanImg.at<uchar>(y, x) = avg;
//
//    }
//
//    imwrite("/home/cy/Desktop/test.png", meanImg);


}