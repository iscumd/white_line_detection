#include "white_line_detection/white_line_detection.hpp"

#include <cstdint>
#include <string>
#include <memory>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace WhiteLineDetection
{
	WhiteLineDetection::WhiteLineDetection(const rclcpp::NodeOptions& options)
		: Node("white_line_detection", options)
	{
		// Define topic subscriptions and publishers
		raw_img_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			"/camera/image_raw", 10,
			std::bind(&WhiteLineDetection::raw_img_callback, this, std::placeholders::_1));

		cam_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			"/camera/camera_info", 10,
			std::bind(&WhiteLineDetection::cam_info_callback, this, std::placeholders::_1));

		img_test_ = this->create_publisher<sensor_msgs::msg::Image>(
			"/camera/test_img", rclcpp::SensorDataQoS());

		// Define Parameters
		auto lowColor = this->declare_parameter("lower_bound_white", 240);

		auto nthPixel = this->declare_parameter("sample_nth_pixel", 5);

		uint8_t kernelSize = this->declare_parameter("kernel_size", 3);

		debugOnly = this->declare_parameter("debug_only", false);

		// Tf stuff
		auto camera_frame = this->declare_parameter("camera_frame", "camera_link");
		auto base_frame = this->declare_parameter("base_frame", "base_footprint");

		// Frontend
		auto thresh_str = this->declare_parameter("thresholder", "isc.dyn_gauss");

		if (thresh_str == "basic") thresholder = std::make_shared<BasicThresholder>(lowColor);
		else if (thresh_str == "isc.dyn_gauss") thresholder = std::make_shared<DynamicGaussThresholder>(kernelSize);

		// Backend
		auto back_str = this->declare_parameter("backend", "pointcloud2");

		if (back_str == "pointcloud2") backend = std::make_unique<PointCloud2Backend>(reinterpret_cast<rclcpp::Node*>(this), nthPixel, base_frame, camera_frame);
	}

	/// Sets up the GPU to run our code using OpenCl.

	void WhiteLineDetection::setupOCL()
	{
		cv::setUseOptimized(true);
		cv::ocl::setUseOpenCL(true);
		if (cv::useOptimized())
		{
			std::cout << "OpenCL optimizations enabled" << std::endl;
		}
		else
		{
			std::cout << "OpenCL optimizations NOT enabled" << std::endl;
		}

		if (!cv::ocl::haveOpenCL())
		{
			std::cout << "no opencl devices detected" << std::endl;
		}

		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_GPU))
		{
			std::cout << "failed to initialize device" << std::endl;
		}
		std::cout << context.ndevices() << " GPU device(s) detected." << std::endl;

		std::cout << "************************" << std::endl;
		for (size_t i = 0; i < context.ndevices(); i++)
		{
			const cv::ocl::Device& device = context.device(i);
			std::cout << "name: " << device.name() << std::endl;
			std::cout << "available: " << device.available() << std::endl;
			std::cout << "img support: " << device.imageSupport() << std::endl;
			std::cout << device.OpenCL_C_Version() << std::endl;
		}
		std::cout << "************************" << std::endl;

		cv::ocl::Device d = cv::ocl::Device::getDefault();
		std::cout << d.OpenCLVersion() << std::endl;
	}


	/// Converts a raw image to an openCv matrix. This function should decode the image properly automatically.
	///
	/// Returns the cv matrix form of the image.
	cv::UMat WhiteLineDetection::ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg)
	{
		auto cvImage = cv_bridge::toCvCopy(imageMsg, "mono8"); // This should decode correctly, but we may need to deal with bayer filters depending on the driver.
		return cvImage->image.getUMat(cv::ACCESS_RW);
	}

	/// Callback passed to the image topic subscription.
	void WhiteLineDetection::raw_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
	{
		if (!connected)
		{
			RCLCPP_ERROR(this->get_logger(), "Received image without receiving camera info first. This should not occur, and is a logic error.");
		}
		else
		{
			// Decode image
			auto cvImg = ptgrey2CVMat(msg);

			// Filter non-white pixels out
			auto filteredImg = cv::UMat(HEIGHT, WIDTH, CV_8UC1);
			thresholder->threshold(cvImg, filteredImg);

			// Outputs an image as a topic for testing
			auto hdr = std_msgs::msg::Header{};
			hdr.frame_id = "camera_link";
			hdr.stamp = this->get_clock()->now();
			auto out = cv_bridge::CvImage{hdr, "mono8", filteredImg.getMat(cv::ACCESS_READ)};
			auto img = sensor_msgs::msg::Image{};
			out.toImageMsg(img);
			this->img_test_->publish(img);

			// Call backend action
			if (!debugOnly) {
				backend->processWhiteLines(filteredImg, msg->header);
			}
		}
	}

	/// Callback passed to the camera info sub.
	void WhiteLineDetection::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
	{
		if (!connected) // attempt to prevent this callback from spamming.
		{
			HEIGHT = msg->height;
			WIDTH = msg->width;

			backend->setCameraInfo(msg);

			RCLCPP_INFO(this->get_logger(), "Connected to camera");
			connected = true;
		}
	}

}

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::executors::SingleThreadedExecutor exec;
	rclcpp::NodeOptions options;
	auto white_line_detection = std::make_shared<WhiteLineDetection::WhiteLineDetection>(options);
	exec.add_node(white_line_detection);

	white_line_detection->setupOCL();

	exec.spin();
	rclcpp::shutdown();
	return 0;
}
