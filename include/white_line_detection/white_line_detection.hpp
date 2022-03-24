#pragma once

#include <memory>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include <opencv2/opencv.hpp>

#include "white_line_detection/backend.hpp"
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
        std::shared_ptr<IThresholder> thresholder;

        /// The abstract processor that handles the binary image and publishing.
        std::unique_ptr<IBackend> backend;

        /// Only displays the test image without points if set.
        bool debugOnly;

        /// True when connected to the camera.
        bool connected{false};

        /// Camera resolution as retrived from the camera_info topic.
        int HEIGHT{}, WIDTH{};

        // Define raw image callback and camera info callback
        void raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
        void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        static cv::UMat ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg);

        // Define subscriptions
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_subscription_;

        // Test topic that shows the modified image in rviz
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_test_;
    };
}
