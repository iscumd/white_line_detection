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

        /// Enable the openCv visualization if set by the node param.
        bool enableImShow;

    private:
        // Define variables

        /// True when connected to the camera.
        bool connected{false};

        /// Camera resolution as retrived from the camera_info topic.
        int HEIGHT{}, WIDTH{};

        /// The lower and upper bound for what we define 'white' as.
        int upperColor{255};
        int lowColor;
        int highB, highG, highR;
        int lowB, lowG, lowR;

        /// The 3x3 perspective transform matrix. Should be treated as constant.
        cv::Mat Utransmtx;
        /// The region of intrest.
        cv::Rect ROI;
        /// Kernal used for white pixel filtering.
        cv::Mat erosionKernel;

        // Ros Params

        /// Calibration constants. These are used for finding the offsets of the white pixels from the robot body.
        double A, B, C, D;
        /// width / height of the actual panel on the ground.
        double ratio;
        /// Warp pixel locations that do something
        float tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y;

        /// The size of the erosion kernel.
        int kernelSize;
        /// The nth pixel to sample from the white pixels. Prevents spam to PCL2.
        int nthPixel;

        // Define raw image callback and camera info callback
        void raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        // Define other image pipeline functions
        void getPixelPointCloud(cv::Mat &erodedImage) const;
        void display(cv::Mat &Uinput, cv::Mat &Utransformed, cv::Mat &Uerosion) const;
        cv::Mat imageFiltering(cv::Mat &warpedImage) const;
        cv::Mat shiftPerspective(cv::Mat &inputImage) const;
        cv::Mat ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg) const;

        // empty callback functions but it is the only way to increment the sliders
        static void lowBlueTrackbar(int, void *) {}
        static void highBlueTrackbar(int, void *) {}
        static void lowGreenTrackbar(int, void *) {}
        static void highGreenTrackbar(int, void *) {}
        static void lowRedTrackbar(int, void *) {}
        static void highRedTrackbar(int, void *) {}

        // Define subscriptions
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_subscription_;

        // Define publishers
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr camera_cloud_publisher_;

        // TODO remove test topic that publishes the filtered images
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_test_;
    };
}

#endif
