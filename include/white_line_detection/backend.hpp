/**
 * @file backend.hpp
 * @author Andrew Ealovega
 * @brief Backends for white line detection. Each takes the thresholded image from the frontend and performs some action on it.
 * An example could be publishing as a pointcloud, or an occupancy grid.
 */

#pragma once

#include "white_line_detection/raytrace.hpp"

#include <image_geometry/pinhole_camera_model.h>

#include <opencv2/imgproc.hpp>

#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>


class IBackend {
    public: 

    /**
     * @brief Takes a passed matrix of white pixels, and produces some result.
     */
    virtual void processWhiteLines(cv::UMat binaryImg) = 0;

    /**
     * @brief Called by parent node when camera info is received for the first time.
     */
    virtual void setCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) = 0;

    virtual ~IBackend() = default;
};


/**
 * @brief Backend that publishes lines as pointcloud2 messages by raytracing from the camera
 * to ground. 
 */
class PointCloud2Backend final : public IBackend {
    public:

    PointCloud2Backend(
        rclcpp::Node* parent, 
        int nthPixel, 
        std::string base_frame, 
        std::string camera_frame) 
        : parent(parent), nthPixel(nthPixel), base_frame(base_frame), camera_frame(camera_frame) 
    {
        tf_buffer = std::make_unique<tf2_ros::Buffer>(parent->get_clock());
		transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        camera_cloud_publisher_ = parent->create_publisher<sensor_msgs::msg::PointCloud2>(
			"/camera/camera_points", rclcpp::SensorDataQoS());
    };

    virtual ~PointCloud2Backend() = default;

    /// Converts the white pixel matrix into a pointcloud, then publishes the pointclouds.
	///
	/// This function works by intersecting the ground plane with a ray cast from each white pixel location, and converting that point to a PCL.
	/// This pointcloud is then broadcast, allowing the nav stack to see the white lines as obsticles.
    virtual void processWhiteLines(cv::UMat erodedImage) {
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
			ray_point = cv::Vec3f{0, 0, static_cast<float>(trans.transform.translation.z)};
		}
		catch (std::exception &e)
		{
			RCLCPP_ERROR(parent->get_logger(), "base_footprint->camera transform failed with: %s", e.what());
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
			pcl_msg.header.stamp = parent->now();
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
		pcl_msg.header.stamp = parent->now();

		camera_cloud_publisher_->publish(pcl_msg);
    }

    virtual void setCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        cameraModel.fromCameraInfo(*msg); // Calibrate camera model
    }

    private:
    /// Reference to parent node, to allow for access to clock and publishers. The parent node must outlive this class.
    rclcpp::Node* parent; 

    // Tf stuff
    std::shared_ptr<tf2_ros::TransformListener> transform_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;

    /// Model used for raycasting from the camera to ground.
    image_geometry::PinholeCameraModel cameraModel;

    /// The nth pixel to sample from the white pixels. Prevents spam to PCL2.
    int nthPixel;

    // Define publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr camera_cloud_publisher_;

    //Params
    std::string base_frame;
    std::string camera_frame;
};
