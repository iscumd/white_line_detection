/**
 * @file frontend.hpp
 * @brief Contains abstractions for the frontend of WLD. These are the structures that give us the matrix of white pixels to pass to the 
 * backend, which actually publishes the pointclouds.
 */

#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>


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
    virtual void threshold(cv::UMat &in, cv::UMat &out) const = 0;
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

    virtual void threshold(cv::UMat &in, cv::UMat &out) const override {
        cv::threshold(in, out, lowerBoundWhite, 255, cv::THRESH_BINARY);
    }

    private:
        int lowerBoundWhite;
};

/**
 * @brief Thresholder that performs an adaptive threshold.
 */
class AdaptiveThresholder : public Thresholder {
    public:
    explicit AdaptiveThresholder(int subConst, int blockSize, cv::AdaptiveThresholdTypes adaptiveKind) 
    : subConst(subConst), blockSize(blockSize), adaptiveKind(adaptiveKind) {};
    virtual ~AdaptiveThresholder() = default;

    virtual void threshold(cv::UMat &in, cv::UMat &out) const override {
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

    virtual void threshold(cv::UMat &in, cv::UMat &out) const override {
        cv::threshold(in, out, 1337, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
};