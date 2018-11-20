#include  "util.h"
#include <string>
#include <vector>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PCloud;

boost::shared_ptr<pcl::visualization::PCLVisualizer> envis (
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 30, 0.1, "normals");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

int main(){
    std::string filename = "/data/odometry/dataset/sequences/04/velodyne/000000.bin";
    std::vector<MPoint> points;

    Kitti::readBinary(filename, points);

    PCloud::Ptr ptr(new PCloud());

    for (int i = 0; i < points.size(); i++){
        pcl::PointXYZRGB p;
        p.x = points[i].x;
        p.y = points[i].y;
        p.z = points[i].z;
        p.r = points[i].i;
        p.g = points[i].i;
        p.b = points[i].i;
        ptr->push_back(p);
    }

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr outputNorm(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(ptr);
    ne.setRadiusSearch(0.3);
    ne.compute(*outputNorm);

    pcl::PointCloud<pcl::Normal>::Ptr filterNorm(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>),
                            globalPose(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::transformPointCloud(*ptr, *ptr, Kitti::getVelodyneToCam());
    float tol = 0.9;
    for (int i = 0; i < outputNorm->points.size(); ++i){
        pcl::Normal &opNorm = outputNorm->points[i];
        if (fabs(opNorm.normal_z) >= tol && ptr->points[i].y >= 1.35){
            filterNorm->push_back(outputNorm->points[i]);
            filtered->push_back(ptr->points[i]);
        }
    }

//    for (auto &po:outputNorm->points){
//        if (po.normal_x < tol){
//            filterNorm->push_back(po);
//        }
//    }

    cout << "orioginla size:" << outputNorm->size() << " filtered size:" << filtered->size() << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer =
            envis(filtered, filterNorm);

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return 0;
}

