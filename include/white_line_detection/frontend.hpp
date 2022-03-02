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
    virtual ~Thresholder() {};
};


/**
 * @brief Thresholder that performs a standard global threshold.  
 * 
 * Needs a lower bound value for construction.
 */
class BasicThresholder : public Thresholder {
    public:
    explicit BasicThresholder(int lowerBoundWhite) : lowerBoundWhite(lowerBoundWhite) {};
    virtual ~BasicThresholder() {};

    virtual void threshold(cv::UMat &in, cv::UMat &out) const override {
        cv::threshold(in, out, lowerBoundWhite, 255, cv::THRESH_BINARY);
    }

    private:
        int lowerBoundWhite;
};