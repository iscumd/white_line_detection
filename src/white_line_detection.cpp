#include "../include/white_line_detection/white_line_detection.hpp"
#include "../include/white_line_detection/raytrace.hpp"
#include "white_line_detection/frontend.hpp"

#include <cassert>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core/ocl.hpp>
#include <cv_bridge/rgb_colors.h>
#include <pcl_conversions/pcl_conversions.h>
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

		camera_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
			"/camera/camera_points", rclcpp::SensorDataQoS());

		img_test_ = this->create_publisher<sensor_msgs::msg::Image>(
			"/camera/test_img", rclcpp::SensorDataQoS());

		// Define Parameters

		auto lowColor = this->declare_parameter("lower_bound_white", 240);

		nthPixel = this->declare_parameter("sample_nth_pixel", 5);

		// Tf stuff
		camera_frame = this->declare_parameter("camera_frame", "camera_link");
		base_frame = this->declare_parameter("base_frame", "base_footprint");

		tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

		// Frontend
		auto thresh_str = this->declare_parameter("thresholder", "mean"); //TODO document this
        int subConst = this->declare_parameter("adaptive_constant", 2);
        int blockSize = this->declare_parameter("adaptive_blocksize", 11);

		if (thresh_str == "basic") thresholder = std::make_shared<BasicThresholder>(lowColor);
		else if (thresh_str == "otsu") thresholder = std::make_shared<OtsuThresholder>();
		else if (thresh_str == "mean") thresholder = std::make_shared<AdaptiveThresholder>(subConst, blockSize, cv::ADAPTIVE_THRESH_MEAN_C);
		else if (thresh_str == "gaussian") thresholder = std::make_shared<AdaptiveThresholder>(subConst, blockSize, cv::ADAPTIVE_THRESH_GAUSSIAN_C);
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

	/// Converts the white pixel matrix into a pointcloud, then publishes the pointclouds.
	///
	/// This function works by intersecting the ground plane with a ray cast from each white pixel location, and converting that point to a PCL.
	/// This pointcloud is then broadcast, allowing the nav stack to see the white lines as obsticles.
	void WhiteLineDetection::getPixelPointCloud(cv::UMat &erodedImage) const
	{
		pcl::PointCloud<pcl::PointXYZ> pointcl;
		sensor_msgs::msg::PointCloud2 pcl_msg;
		std::vector<cv::Point> pixelCoordinates;

		// The position of the camera is (0,0,height from base_footprint)
		cv::Point3f ray_point;
		tf2::Quaternion camera_rotation;

		try
		{
			auto trans = tf_buffer->lookupTransform(base_frame, camera_frame, tf2::TimePointZero);
			tf2::convert(trans.transform.rotation, camera_rotation);
			ray_point = cv::Vec3f{0, 0, (float)trans.transform.translation.z};
		}
		catch (std::exception &e)
		{
			RCLCPP_ERROR(this->get_logger(), "base_footprint->camera transform failed with: %s", e.what());
			return; // Just early return if error
		}

		// Rotate the image by 90 degrees to orient the image frame with the front of Ohm
		cv::Point2f center((erodedImage.cols - 1) / 2.0, (erodedImage.rows - 1) / 2.0);
		cv::Mat rotation_matix = getRotationMatrix2D(center, -90, 1.0);
		cv::warpAffine(erodedImage, erodedImage, rotation_matix, erodedImage.size());

		// Remove all non white pixels
		cv::findNonZero(erodedImage, pixelCoordinates);

		// If no white lines detected, publish empty cloud so the pointcloud concat sync will still work
		if (pixelCoordinates.size() == 0) {
			pointcl.push_back(pcl::PointXYZ{});
			pcl::toROSMsg(pointcl, pcl_msg);
			pcl_msg.header.frame_id = base_frame;
			pcl_msg.header.stamp = this->now();
			camera_cloud_publisher_->publish(pcl_msg);
			return;
		}

		// Iter through all the white pixels, adding their locations to pointclouds by 'raytracing' their location on the map from the camera.
		for (size_t i = 0; i < pixelCoordinates.size(); i++)
		{
			if (i % nthPixel == 0)
			{
				auto ray_unrotated = raytracing::convertToOpenCvVec3(cameraModel.projectPixelTo3dRay(pixelCoordinates[i]));													  // Get ray out of camera, correcting for tilt and pan
				auto ray = raytracing::convertTfToOpenCvVec3(tf2::quatRotate(camera_rotation.normalized(), tf2::Vector3{ray_unrotated.x, ray_unrotated.y, ray_unrotated.z})); // Rotate ray by trans base_footprint->camera
				const auto normal = cv::Point3f{0.0, 0.0, 1.0};																												  // Assume flat plane
				const auto plane_point = cv::Point3f{0.0, 0.0, 0.0};																										  // Assume 0,0,0 in plane

				// Find the point where the ray intersects the ground ie. the point where the pixel maps to in the map.
				auto new_point = raytracing::intersectLineAndPlane(ray, ray_point, normal, plane_point);

				// Flip all points over y axis cause their initially placed behind ohm for some reason
				new_point.x = -(new_point.x);

				// Moves the line in the air, which causes pcl-ls to actually accept it.
				new_point.z += 1;

				pointcl.points.push_back(new_point);
			}
		}

		pcl::toROSMsg(pointcl, pcl_msg);
		pcl_msg.header.frame_id = base_frame; // Because we use base_footprint->camera_frame translation as our camera point, all points are relative to base_footprint
		pcl_msg.header.stamp = this->now();

		camera_cloud_publisher_->publish(pcl_msg);
	}

	/// Converts a raw image to an openCv matrix. This function should decode the image properly automatically.
	///
	/// Returns the cv matrix form of the image.
	cv::UMat WhiteLineDetection::ptgrey2CVMat(const sensor_msgs::msg::Image::SharedPtr &imageMsg)
	{
		auto cvImage = cv_bridge::toCvCopy(imageMsg, "mono8"); // This should decode correctly, but we may need to deal with bayer filters depending on the driver.
		return cvImage->image.getUMat(cv::ACCESS_RW);
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
			auto filteredImg = cv::UMat(HEIGHT, WIDTH, CV_8UC1);
			thresholder->threshold(cvImg, filteredImg);

			// Outputs an image as a topic for testing
			auto hdr = std_msgs::msg::Header{};
			hdr.frame_id = camera_frame;
			hdr.stamp = this->get_clock()->now();
			auto out = cv_bridge::CvImage{hdr, "mono8", filteredImg.getMat(cv::ACCESS_READ)};
			auto img = sensor_msgs::msg::Image{};
			out.toImageMsg(img);
			this->img_test_->publish(img);

			// Convert pixels to pointcloud and publish
			getPixelPointCloud(filteredImg);
		}
	}

	/// Callback passed to the camera info sub.
	void WhiteLineDetection::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
	{
		if (!connected) // attempt to prevent this callback from spamming.
		{
			HEIGHT = msg->height;
			WIDTH = msg->width;
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

	white_line_detection->setupOCL();

	exec.spin();
	rclcpp::shutdown();
	return 0;
}
