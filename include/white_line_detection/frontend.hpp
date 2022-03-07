/**
 * @file frontend.hpp
 * @brief Contains abstractions for the frontend of WLD. These are the structures that give us the matrix of white pixels to pass to the 
 * backend, which actually publishes the pointclouds.
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>


/**
 * @brief Interface for thresholders. 
 */
class Thresholder {
    public:
    
    /**
     * @brief Thresholds the input image, copying only the white line points into the output image.
     * 
     * @param in The input image matrix.
     * @param out The output image matrix to be written to.
     */
    virtual void threshold(cv::UMat &in, cv::UMat &out) = 0;
    virtual ~Thresholder() = default;
};


/**
 * @brief Thresholder that performs a standard global threshold.  
 * 
 * Needs a lower bound value for construction.
 */
class BasicThresholder : public Thresholder {
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
class DynamicGaussThresholder : public Thresholder {
    public:
    virtual ~DynamicGaussThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) override {
        cv::Mat mean, dev;

        cv::meanStdDev(in, mean, dev);
        auto mean_val = mean.data[0]; //Assuming mono8, only one chanel to get the mean of
        auto dev_val = dev.data[0]; //Assuming mono8

        lowerBoundWhite = static_cast<int>(mean_val + 2 * dev_val); // Set treshold to be the top 3%ish of luminance

        RCLCPP_INFO(rclcpp::get_logger("dynGaussThresholder"), "using mean: %d | stdDev: %d | threshold: %i", mean_val, dev_val, lowerBoundWhite);

        cv::threshold(in, out, lowerBoundWhite, 255, cv::THRESH_BINARY);
    }

    private:
        mutable int lowerBoundWhite; //TODO add a variable so that this is only recalculated so many frames at a time to save cycles
};

/**
 * @brief Thresholder that performs an adaptive threshold.
 */
class AdaptiveThresholder : public Thresholder {
    public:
    explicit AdaptiveThresholder(int subConst, int blockSize, cv::AdaptiveThresholdTypes adaptiveKind) 
    : subConst(subConst), blockSize(blockSize), adaptiveKind(adaptiveKind) {};
    virtual ~AdaptiveThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) override {
        cv::adaptiveThreshold(in, out, 255, adaptiveKind, cv::THRESH_BINARY, blockSize, subConst);
    }

    private:
        /// A constant subtracted from the sum of an area to get the threshold bound.
        int subConst;
        /// The size of the region to sample for each sum.
        int blockSize;
        /// The cv enum of cv.ADAPTIVE_THRESH_MEAN_C or cv.ADAPTIVE_THRESH_GAUSSIAN_C
        cv::AdaptiveThresholdTypes adaptiveKind;
};


/**
 * @brief Thresholder that performs otsu's method.
 * This thresholder chooses a threshold value itself.
 */
class OtsuThresholder : public Thresholder {
    public:
    virtual ~OtsuThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) override {
        cv::threshold(in, out, 1337, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
};