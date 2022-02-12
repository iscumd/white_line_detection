#include "../include/white_line_detection/white_line_detection.hpp"
#include "../include/white_line_detection/raytrace.hpp"

#include <string>
#include <opencv2/core/ocl.hpp>
#include <cv_bridge/rgb_colors.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

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

		// Warp params
		tl_x = this->declare_parameter("pixel_coordinates_tl_x", 80.0);
		tl_y = this->declare_parameter("pixel_coordinates_tl_y", 0.0);
		tr_x = this->declare_parameter("pixel_coordinates_tr_x", 320.0);
		tr_y = this->declare_parameter("pixel_coordinates_tr_y", 0.0);
		bl_x = this->declare_parameter("pixel_coordinates_bl_x", 0.0);
		bl_y = this->declare_parameter("pixel_coordinates_bl_y", 240.0);
		br_x = this->declare_parameter("pixel_coordinates_br_x", 320.0);
		br_y = this->declare_parameter("pixel_coordinates_br_y", 240.0);

		ratio = this->declare_parameter("pixel_coordinates_ratio", 1.0);

		kernelSize = this->declare_parameter("kernel_size", 5);
		nthPixel = this->declare_parameter("sample_nth_pixel", 5);

		// Tf stuff
		camera_frame = this->declare_parameter("camera_frame", "camera_link");

		tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

		// Define CV variables
		erosionKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

		// Create warp matrix.
		setupWarp();
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

	/// Sets up the constant perspective transform matrix as defined by node params.
	void WhiteLineDetection::setupWarp()
	{
		// Points in the raw image to make rect
		cv::Point Q1 = cv::Point2f(tl_x, tl_y); // top left pixel coordinate
		cv::Point Q2 = cv::Point2f(tr_x, tr_y); // top right
		cv::Point Q3 = cv::Point2f(br_x, br_y); // bottom right
		cv::Point Q4 = cv::Point2f(bl_x, bl_y); // bottom left

		// Take the Pythagorean theorem of the right side (which may be a triangle) to find the height of the final image (equal to that triangles hypotanuse).
		float recth = sqrt((Q3.x - Q2.x) * (Q3.x - Q2.x) + (Q3.y - Q2.y) * (Q3.y - Q2.y));
		// Apply image ratio based off above height.
		float rectw = ratio * recth;

		// Create a rectangle with top left corner in the top left coord of the source image, and with width and height calced above.
		cv::Rect R(Q1.x, Q1.y, rectw, recth);

		// The destination coordinates in the warped image.
		cv::Point R1 = cv::Point2f(R.x, R.y);					   // Top left doesn't change
		cv::Point R2 = cv::Point2f(R.x + R.width, R.y);			   // Top right y doesn't change and x is stretched to width.
		cv::Point R3 = cv::Point2f(R.x + R.width, R.y + R.height); // Bottom right stretches to width and moves y down to new height.
		cv::Point R4 = cv::Point2f(R.x, R.y + R.height);		   // Bottom left keeps the same x and moves y down to new height.

		std::vector<cv::Point2f> squarePts{R1, R2, R3, R4};
		std::vector<cv::Point2f> quadPts{Q1, Q2, Q3, Q4};

		// Copy transform to constant feild
		auto transmtx_double = cv::getPerspectiveTransform(quadPts, squarePts);

		// Explicitly convert to floats, will be double and fail an assert otherwise
		transmtx_double.convertTo(transmtx, CV_32FC1);
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

		// The position of the camera is (0,0,height from base_footprint)
		cv::Point3f ray_point;
		tf2::Quaternion camera_rotation;

		try
		{
			auto trans = tf_buffer->lookupTransform("base_footprint", camera_frame, tf2::TimePointZero);
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
				pcl::PointXYZ new_point = raytracing::intersectLineAndPlane(ray, ray_point, normal, plane_point);

				// Flip all points over y axis cause their initially placed behind ohm for some reason
				new_point.x = -(new_point.x);

				pointcl.points.push_back(new_point);
			}
		}

		pcl::toROSMsg(pointcl, pcl_msg);
		pcl_msg.header.frame_id = "base_footprint"; // Because we use base_footprint->camera_frame translation as our camera point, all points are relative to base_footprint
		pcl_msg.header.stamp = this->now();

		camera_cloud_publisher_->publish(pcl_msg);
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

	/// Applies the perspective warp to the image.
	///
	/// Returns the warped image matrix.
	cv::Mat WhiteLineDetection::shiftPerspective(cv::Mat &inputImage) const
	{
		// The transformed image
		auto transformed = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
		cv::warpPerspective(inputImage, transformed, transmtx, transformed.size());

		return transformed(ROI); // Contrain to region
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

			// Correct for angle
			auto warped = shiftPerspective(cvImg);

			// Filter non-white pixels out
			auto filteredImg = imageFiltering(warped);

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

	exec.spin();
	rclcpp::shutdown();
	return 0;
}
