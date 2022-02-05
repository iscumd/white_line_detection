#include "../include/white_line_detection/white_line_detection.hpp"

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
		A = this->declare_parameter("calibration_constants_A", 0.01);
		B = this->declare_parameter("calibration_constants_B", 0.01);
		C = this->declare_parameter("calibration_constants_C", 0.01);
		D = this->declare_parameter("calibration_constants_D", 0.01);

		// Warp params
		tl_x = this->declare_parameter("pixel_coordinates_tl_x", 0.0);
		tl_y = this->declare_parameter("pixel_coordinates_tl_y", 0.0);
		tr_x = this->declare_parameter("pixel_coordinates_tr_x", 320.0);
		tr_y = this->declare_parameter("pixel_coordinates_tr_y", 0.0);
		bl_x = this->declare_parameter("pixel_coordinates_bl_x", 0.0);
		bl_y = this->declare_parameter("pixel_coordinates_bl_y", 240.0);
		br_x = this->declare_parameter("pixel_coordinates_br_x", 320.0);
		br_y = this->declare_parameter("pixel_coordinates_br_y", 240.0);
      
		ratio = this->declare_parameter("pixel_coordinates_ratio", 1.0);

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

    //Define CV variables
    erosionKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    
    }

	///Sets up the GPU to run our code using OpenCl.

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

	/// Sets up the constant perspective transform matrix as defined by node params.
	void WhiteLineDetection::setupWarp()
	{
		cv::Point Q1 = cv::Point2f(tl_x, tl_y); // top left pixel coordinate
		cv::Point Q2 = cv::Point2f(tr_x, tr_y); // top right
		cv::Point Q3 = cv::Point2f(br_x, br_y); // bottom right
		cv::Point Q4 = cv::Point2f(bl_x, bl_y); // bottom left

		// The dimensions of Ohms board(?)
		double boardH = sqrt((Q3.x - Q2.x) * (Q3.x - Q2.x) + (Q3.y - Q2.y) * (Q3.y - Q2.y));
		double boardW = ratio * boardH;

		cv::Rect R(Q1.x, Q1.y, boardW, boardH);

		cv::Point R1 = cv::Point2f(R.x, R.y);
		cv::Point R2 = cv::Point2f(R.x + R.width, R.y);
		cv::Point R3 = cv::Point2f(cv::Point2f(R.x + R.width, R.y + R.height));
		cv::Point R4 = cv::Point2f(cv::Point2f(R.x, R.y + R.height));

		std::vector<cv::Point2f> squarePts{R1, R2, R3, R4};
		std::vector<cv::Point2f> quadPts{Q1, Q2, Q3, Q4};

		// Copy transform to constant feild
		auto transmtx = cv::getPerspectiveTransform(quadPts, squarePts);
		transmtx.copyTo(Utransmtx);
	}

	/// Converts the white pixel matrix into a pointcloud, then publishes the pointclouds.
	///
	/// This function works by offsetting each pixel by some calibrated constants to get the geographical lie of that pixel, which becomes a point
	/// in the pointcloud. This pointcloud is then broadcast, allowing the nav stack to see the white lines as obsticles.
	void WhiteLineDetection::getPixelPointCloud(cv::Mat &erodedImage) const
	{
		sensor_msgs::msg::PointCloud msg;
  		sensor_msgs::msg::PointCloud2::SharedPtr msg2(new sensor_msgs::msg::PointCloud2);
		std::vector<cv::Point> pixelCoordinates;

		cv::findNonZero(erodedImage, pixelCoordinates);
		// Iter through all the white pixels, adding their locations to pointclouds.
		for (size_t i = 0; i < pixelCoordinates.size(); i++) // TODO: Reconfigure this so that it works with PCL2 so we can stop using PCL
		{
			if (i % nthPixel == 0)
			{
				geometry_msgs::msg::Point32 pixelLoc;
				// XY distances of each white pixel relative to robot
				pixelLoc.x = (A * pixelCoordinates[i].x) + B;
				pixelLoc.y = (C * pixelCoordinates[i].y) + D;
				msg.points.push_back(pixelLoc);
			}
		}
		sensor_msgs::convertPointCloudToPointCloud2(msg, *msg2);
		
		msg2->header.frame_id = "camera_link"; //Fix for gazebo
		camera_cloud_publisher_->publish(*msg2);
	}

	void WhiteLineDetection::createGUI()
	{
		cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("erosion", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("warp", cv::WINDOW_AUTOSIZE);
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
	void WhiteLineDetection::display(cv::Mat &Uinput, cv::Mat &Utransformed, cv::Mat &Uerosion) const
	{
		cv::imshow("original", Uinput);
		cv::imshow("warp", Utransformed);
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

	/// Applies the perspective warp to the image.
	///
	/// Returns the warped image matrix.
	cv::Mat WhiteLineDetection::shiftPerspective(cv::Mat &inputImage) const
	{
		// The transformed image
		auto transformed = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
		cv::warpPerspective(inputImage, transformed, Utransmtx, transformed.size());

		return transformed(ROI);
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
            //RCLCPP_INFO(this->get_logger(),"Recived image!");
            
			// Decode image
			auto cvImg = ptgrey2CVMat(msg);
			// Perspective warp
			auto warpedImg = shiftPerspective(cvImg);
			// Filter non-white pixels out
			auto filteredImg = imageFiltering(warpedImg);

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
				display(cvImg, warpedImg, filteredImg);
		}
	}

	/// Callback passed to the camera info sub.
	void WhiteLineDetection::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
	{
		if (!connected) // attempt to prevent this callback from spamming.
		{
			HEIGHT = msg->height;
			WIDTH = msg->width;
			ROI = cv::Rect(12, 12, WIDTH-20, HEIGHT-20);
			RCLCPP_INFO(this->get_logger(),"Connected to camera");
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

	// Only enable gui if set as it crashes otherwise
	if (white_line_detection->enableImShow) {
		white_line_detection->createGUI();
	}

	white_line_detection->setupWarp();
	exec.spin();
	rclcpp::shutdown();
	return 0;
}
