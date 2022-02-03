#ifndef WHITE_LINE_DETECTION__WHITE_LINE_DETECTION_HPP_
#define WHITE_LINE_DETECTION__WHITE_LINE_DETECTION_HPP_

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <image_transport/image_transport.hpp>


namespace WhiteLineDetection
{
    class WhiteLineDetection : public rclcpp::Node
    {
    public:
        explicit WhiteLineDetection(rclcpp::NodeOptions options);
        void setupOCL();
        void setupWarp();
        void createGUI();
    private:
        //Define variables
        bool connected{false};
        int HEIGHT{}, WIDTH{};
        int upperColor{255}; 
        int highB, highG, highR;
        int lowB, lowG, lowR;
        cv::UMat Utransmtx;
        cv::Rect ROI;
        cv::Mat erosionKernel;

        //Ros Params
        double A, B, C, D;
        double ratio;
        int tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y;
        int lowColor, kernelSize, nthPixel;
        bool enableImShow;

        //Define raw image callback and camera info callback
        void raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        //Define other image pipeline functions
        void getPixelPointCloud(cv::UMat &erodedImage) const;
        void display(cv::UMat &Uinput, cv::UMat &Utransformed, cv::UMat &Uerosion);
        cv::UMat imageFiltering(cv::UMat &warpedImage);
        cv::UMat shiftPerspective(cv::UMat &inputImage);
        cv::UMat ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg);

        //empty callback functions but it is the only way to increment the sliders
        static void lowBlueTrackbar(int, void *){}
        static void highBlueTrackbar(int, void *){}
        static void lowGreenTrackbar(int, void *){}
        static void highGreenTrackbar(int, void *){}
        static void lowRedTrackbar(int, void *){}
        static void highRedTrackbar(int, void *){}


        
        //Define subscriptions
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_subscription_;
        
        //Define publishers
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr camera_cloud_publisher_;
        
    };
}

#endif
