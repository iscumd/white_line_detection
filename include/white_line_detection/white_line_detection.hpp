#ifndef WHITE_LINE_DETECTION__WHITE_LINE_DETECTION_HPP_
#define WHITE_LINE_DETECTION__WHITE_LINE_DETECTION_HPP_

#include <memory>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "image_geometry/pinhole_camera_model.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace WhiteLineDetection
{
    class WhiteLineDetection : public rclcpp::Node
    {
    public:
        explicit WhiteLineDetection(rclcpp::NodeOptions options);
        void setupOCL();
        void setupWarp();

    private:
        /// True when connected to the camera.
        bool connected{false};

        /// Camera resolution as retrived from the camera_info topic.
        int HEIGHT{}, WIDTH{};

        /// The lower and upper bound for what we define 'white' as.
        int upperColor{255};
        int lowColor;
        int highB, highG, highR;
        int lowB, lowG, lowR;

        /// Warp pixel locations of the raw image.
        float tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y;

        /// Ratio of width/heigth in the warped image. 1 would be square.
        float ratio;

        /// The 3x3 perspective transform matrix. Should be treated as constant.
        cv::Mat transmtx;
        /// The region of intrest. Crops to this area.
        cv::Rect ROI;
        /// Kernal used for white pixel filtering.
        cv::Mat erosionKernel;

        /// Model used for raycasting from the camera to ground.
        image_geometry::PinholeCameraModel cameraModel;

        // Tf stuff
        std::shared_ptr<tf2_ros::TransformListener> transform_listener{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        // Frame to grab camera info from
        std::string camera_frame;

        /// The size of the erosion kernel.
        int kernelSize;
        /// The nth pixel to sample from the white pixels. Prevents spam to PCL2.
        int nthPixel;

        // Define raw image callback and camera info callback
        void raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        // Define other image pipeline functions
        void getPixelPointCloud(cv::Mat &erodedImage) const;
        cv::Mat imageFiltering(cv::Mat &warpedImage) const;
        cv::Mat ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg) const;
        cv::Mat shiftPerspective(cv::Mat &inputImage) const;

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
