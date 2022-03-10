#pragma once

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
#include "white_line_detection/frontend.hpp"

namespace WhiteLineDetection
{
    class WhiteLineDetection : public rclcpp::Node
    {
    public:
        explicit WhiteLineDetection(const rclcpp::NodeOptions& options);
        void setupOCL();

    private:
        /// The abstract thresholder used in the frontend.
        std::shared_ptr<Thresholder> thresholder;

        /// True when connected to the camera.
        bool connected{false};

        /// Camera resolution as retrived from the camera_info topic.
        int HEIGHT{}, WIDTH{};

        /// Model used for raycasting from the camera to ground.
        image_geometry::PinholeCameraModel cameraModel;

        // Tf stuff
        std::shared_ptr<tf2_ros::TransformListener> transform_listener{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        // Frame to grab camera info from
        std::string camera_frame;
        // Base frame
        std::string base_frame;

        /// The nth pixel to sample from the white pixels. Prevents spam to PCL2.
        int nthPixel;

        /// Only displays the test image without points if set.
        bool debugOnly;

        // Define raw image callback and camera info callback
        void raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        // Define other image pipeline functions
        void getPixelPointCloud(cv::UMat &erodedImage) const;
        static cv::UMat ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg) ;

        // Define subscriptions
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_subscription_;

        // Define publishers
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr camera_cloud_publisher_;

        // Test topic that shows the modified image in rviz
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_test_;
    };
}
