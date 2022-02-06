#include "../include/white_line_detection/white_line_detection.hpp"
#include "../include/white_line_detection/raytrace.hpp"

#include <memory>
#include <string>
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.hpp>
#include "cv_bridge/cv_bridge.h"
#include <cv_bridge/rgb_colors.h>
#include "cv_bridge/cv_bridge_export.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

namespace WhiteLineDetection
{
	WhiteLineDetection::WhiteLineDetection(rclcpp::NodeOptions options)
		: Node("white_line_detection", options)
	{
		// Define topic subscriptions and publishers
		raw_img_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			"/camera/image_raw", 10,
			std::bind(&WhiteLineDetection::raw_img_callback, this, std::placeholders::_1));

		cam_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			"/camera/camera_info", 10,
			std::bind(&WhiteLineDetection::cam_info_callback, this, std::placeholders::_1));

		camera_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
			"/camera/camera_points", rclcpp::SensorDataQoS());

		img_test_ = this->create_publisher<sensor_msgs::msg::Image>(
			"/camera/test_img", rclcpp::SensorDataQoS());

		// Define Parameters
		lowColor = this->declare_parameter("lower_bound_white", 160);
		lowB = lowColor;
		lowG = lowColor;
		lowR = lowColor;
		highB = upperColor;
		highG = upperColor;
		highR = upperColor;

		kernelSize = this->declare_parameter("kernel_size", 5);
		nthPixel = this->declare_parameter("sample_nth_pixel", 5);
		enableImShow = this->declare_parameter("enable_imshow", true);

		// Tf stuff
		camera_frame = this->declare_parameter("camera_frame", "camera_link");
		map_frame = this->declare_parameter("map_frame", "map");
		
		tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

		// Define CV variables
		erosionKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
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
			cv::ocl::Device device = context.device(i);
			std::cout << "name: " << device.name() << std::endl;
			std::cout << "available: " << device.available() << std::endl;
			std::cout << "img support: " << device.imageSupport() << std::endl;
			std::cout << device.OpenCL_C_Version() << std::endl;
		}
		std::cout << "************************" << std::endl;

		cv::ocl::Device d = cv::ocl::Device::getDefault();
		std::cout << d.OpenCLVersion() << std::endl;
	}

	/// Converts the white pixel matrix into a pointcloud, then publishes the pointclouds.
	///
	/// This function works by intersecting the ground plane with a ray cast from each white pixel location, and converting that point to a PCL.
	/// This pointcloud is then broadcast, allowing the nav stack to see the white lines as obsticles.
	void WhiteLineDetection::getPixelPointCloud(cv::Mat &erodedImage) const
	{
		pcl::PointCloud<pcl::PointXYZ> pointcl;
		sensor_msgs::msg::PointCloud2 pcl_msg;
		std::vector<cv::Point> pixelCoordinates;
		auto camera_to_ground_trans = tf_buffer->lookupTransform(camera_frame, map_frame, tf2::TimePointZero);

		cv::findNonZero(erodedImage, pixelCoordinates);
		// Iter through all the white pixels, adding their locations to pointclouds by 'raytracing' their location on the map from the camera.
		for (size_t i = 0; i < pixelCoordinates.size(); i++)
		{
			if (i % nthPixel == 0)
			{
				auto ray = cameraModel.projectPixelTo3dRay(pixelCoordinates[i]); // Get ray out of camera, correcting for tilt and pan
				auto ray_point = camera_to_ground_trans.transform.translation;	 // The position of the camera is its offset from the origin (map frame)
				auto normal = cv::Vec3f{0.0, 0.0, 1.0};							 // Assume flat plane
				auto plane_point = cv::Vec3f{0.0, 0.0, 0.0};					 // Assume 0,0,0 in plane

				// Find the point where the ray intersects the ground ie. the point where the pixel maps to in the map.
				pcl::PointXYZ new_point = raytracing::intersectLineAndPlane(ray, ray_point, normal, plane_point);

				pointcl.points.push_back(new_point);
			}
		}

		pcl_msg.header.frame_id = "map"; // Because we use map_frame->camera_frame translation as our camera point
		pcl::toROSMsg(pointcl, pcl_msg);

		camera_cloud_publisher_->publish(pcl_msg);
	}

	void WhiteLineDetection::createGUI()
	{
		cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("erosion", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("TRACKBARS", cv::WINDOW_AUTOSIZE);
		//*****************GUI related *********************************
		cv::createTrackbar("Low Blue", "TRACKBARS", &lowB, upperColor, lowBlueTrackbar);
		cv::createTrackbar("Low Green", "TRACKBARS", &lowG, upperColor, lowGreenTrackbar);
		cv::createTrackbar("Low Red", "TRACKBARS", &lowR, upperColor, lowRedTrackbar);
		cv::createTrackbar("High Blue", "TRACKBARS", &highB, upperColor, highBlueTrackbar);
		cv::createTrackbar("High Green", "TRACKBARS", &highG, upperColor, highGreenTrackbar);
		cv::createTrackbar("High Red", "TRACKBARS", &highR, upperColor, highRedTrackbar);
	}

	/// Updates the displayed guis with the last processed image.
	void WhiteLineDetection::display(cv::Mat &Uinput, cv::Mat &Uerosion) const
	{
		cv::imshow("original", Uinput);
		cv::imshow("erosion", Uerosion);
	}

	/// Converts a raw image to an openCv matrix. This function should decode the image properly automatically.
	///
	/// Returns the cv matrix form of the image.
	cv::Mat WhiteLineDetection::ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg) const
	{
		auto cvImage = cv_bridge::toCvCopy(imageMsg, "mono8"); // This should decode correctly, but we may need to deal with bayer filters depending on the driver.
		return cvImage->image;
	}

	/// Filters non-white pixels out of the warped image.
	///
	/// Returns the eroded image matrix. The only pixels left should be white.
	cv::Mat WhiteLineDetection::imageFiltering(cv::Mat &warpedImage) const
	{
		auto binaryImage = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
		auto erodedImage = cv::Mat(HEIGHT, WIDTH, CV_8UC1);

		cv::inRange(warpedImage, cv::Scalar(lowB, lowG, lowR), cv::Scalar(highB, highG, highR), binaryImage);
		cv::erode(binaryImage, erodedImage, erosionKernel);

		return erodedImage;
	}

	/// Callback passed to the image topic subscription. This produces a pointcloud for every
	/// image sent on the topic.
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
			auto filteredImg = imageFiltering(cvImg);

			// TODO remove later, outputs an image as a topic
			auto hdr = std_msgs::msg::Header{};
			hdr.frame_id = "camera_link";
			hdr.stamp = this->get_clock()->now();
			auto out = cv_bridge::CvImage{hdr, "mono8", filteredImg};
			auto img = sensor_msgs::msg::Image{};
			out.toImageMsg(img);
			this->img_test_->publish(img);

			// Convert pixels to pointcloud and publish
			getPixelPointCloud(filteredImg);

			// Display to gui if enabled.
			if (enableImShow)
				display(cvImg, filteredImg);
		}
	}

	/// Callback passed to the camera info sub.
	void WhiteLineDetection::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
	{
		if (!connected) // attempt to prevent this callback from spamming.
		{
			HEIGHT = msg->height;
			WIDTH = msg->width;
			ROI = cv::Rect(12, 12, WIDTH - 20, HEIGHT - 20);
			cameraModel.fromCameraInfo(*msg); // Calibrate camera model

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

	// white_line_detection->setupOCL(); TODO make sure this is doing something then re-enable

	// Only enable gui if set as it crashes otherwise
	if (white_line_detection->enableImShow)
	{
		white_line_detection->createGUI();
	}

	exec.spin();
	rclcpp::shutdown();
	return 0;
}
