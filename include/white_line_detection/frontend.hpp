/**
 * @file frontend.hpp
 * @author Andrew Ealovega
 * @brief Contains abstractions for the frontend of WLD. These are the structures that give us the matrix of white pixels to pass to the 
 * backend, which actually publishes the pointclouds.
 */

#pragma once

#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>


/**
 * @brief Interface for thresholders. 
 */
class IThresholder {
    public:
    
    /**
     * @brief Thresholds the input image, copying only the white line points into the output image.
     * 
     * @param in The input image matrix.
     * @param out The output image matrix to be written to.
     */
    virtual void threshold(cv::UMat &in, cv::UMat &out) = 0;
    virtual ~IThresholder() = default;
};


/**
 * @brief Thresholder that performs a standard global threshold.  
 * 
 * Needs a lower bound value for construction.
 */
class BasicThresholder final : public IThresholder {
    public:
    explicit BasicThresholder(int lowerBoundWhite) : lowerBoundWhite(lowerBoundWhite) {};
    virtual ~BasicThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) override {
        cv::threshold(in, out, lowerBoundWhite, 255, cv::THRESH_BINARY);
    }

    private:
        int lowerBoundWhite;
};


/**
 * @brief First calculates the mean luminance of the image, then defines the tolerance to be some std dev away from that mean.
 * Then performs a global threshold using that value. This strategy will fail if there are shadows in the image.
 */
class DynamicGaussThresholder final : public IThresholder {
    public:
    DynamicGaussThresholder(uint8_t kernelSize) {
        erosionKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    }

    virtual ~DynamicGaussThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) override {

        //Recalculate every 5 frames
        if (cycle > 5) {
            cv::Scalar mean, dev;

            cv::meanStdDev(in, mean, dev);
            auto mean_val = mean[0]; //Assuming mono8, only one chanel to get the mean of
            auto dev_val = dev[0]; //Assuming mono8

            lowerBoundWhite = static_cast<int>(mean_val + 3 * dev_val); // Set treshold to be the top 3%ish of luminance

            //RCLCPP_INFO(rclcpp::get_logger("dynGaussThresholder"), "using mean: %f | stdDev: %f | threshold: %i", mean_val, dev_val, lowerBoundWhite);

            cycle = 0;
        }

        cv::threshold(in, out, lowerBoundWhite, 255, cv::THRESH_BINARY);

        //Erode the image slightly to remove small pebbles
        cv::erode(out, out, erosionKernal);

        cycle++;
    }

    private:
        int lowerBoundWhite;
        uint8_t cycle{};
        cv::Mat erosionKernal;
};
