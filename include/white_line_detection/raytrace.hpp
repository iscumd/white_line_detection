#pragma once

#include "pcl/point_types.h"
#include "opencv2/core.hpp"

// The generics in this file are somewhat useless rn, but in the case we get C++20 concepts, making a concept that bounds x,y,z members to require a static_cast<float> would let any vector3 type work.

/// Implements 'raytracing' operations.
namespace raytracing
{

    // Formats matrix to string in R formatting.
    inline auto matToString(const cv::Mat &mat) noexcept -> std::string
    {
        std::stringstream ss;
        ss << mat;
        return ss.str();
    }

    /// Converts any type with x, y, and z members that can be casted to a float to a
    /// cv::Point3f. This is to deal with C++14 generics being a mess.
    template <typename Vec3>
    auto convertToOpenCvVec3(Vec3 vec3) noexcept -> cv::Point3f
    {
        auto cv = cv::Point3f{};
        cv.x = static_cast<float>(vec3.x);
        cv.y = static_cast<float>(vec3.y);
        cv.z = static_cast<float>(vec3.z);

        return cv;
    }

    /// Converts any type with x(), y(), and z() member functions that can be casted to a float to a
    /// cv::Point3f. This is to deal with C++14 generics being a mess.
    template <typename Vec3>
    auto convertTfToOpenCvVec3(Vec3 vec3) noexcept -> cv::Point3f
    {
        auto cv = cv::Point3f{};
        cv.x = static_cast<float>(vec3.x());
        cv.y = static_cast<float>(vec3.y());
        cv.z = static_cast<float>(vec3.z());

        return cv;
    }

    // Define some private generic vector ops. Something about these is implimented wrong, bc the lines get warped when I replace them with the opencv equivelent.
    // But that makes it work so... 
    namespace impl_
    {
        /// Dot product
        template <typename Vec3a, typename Vec3b>
        auto dot(const Vec3a lhs, const Vec3b rhs) -> float
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }

        /// Vector subtraction
        template <typename Vec3a, typename Vec3b>
        auto sub(const Vec3a lhs, const Vec3b rhs) -> Vec3a
        {
            auto out = Vec3a{};
            out.x = lhs.x - rhs.x;
            out.y = lhs.y - rhs.y;
            out.z = lhs.z - rhs.z;

            return out;
        }

        /// Vector addition
        template <typename Vec3a, typename Vec3b>
        auto add(const Vec3a lhs, const Vec3b rhs) -> Vec3a
        {
            auto out = Vec3a{};
            out.x = lhs.x + rhs.x;
            out.y = lhs.y + rhs.y;
            out.z = lhs.z + rhs.z;

            return out;
        }

        /// Vector-Scalar multiplication
        template <typename Vec3>
        auto mult(const Vec3 lhs, float rhs) -> Vec3
        {
            auto out = Vec3{};
            out.x = lhs.x * rhs;
            out.y = lhs.y * rhs;
            out.z = lhs.z * rhs;

            return out;
        }

        /// Vector norm
        template <typename Vec3>
        auto norm(const Vec3 v) -> Vec3
        {
            float len = std::sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));

            auto out = Vec3{};
            out.x = v.x / len;
            out.y = v.y / len;
            out.z = v.z / len;

            return out;
        }
    }

    /// Finds the point of intersection of a ray with starting location and some plane. Ignores the case where the two either don't intersect, or are in each other,
    /// as that cannot occur as it is being used.
    ///
    /// \tparam Vec3 Any type with x, y, and z float members (potentially different between params).
    /// \tparam Point Any type with x, y, and z float members. pcl by default.
    /// \see [credit for algorithm](https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#C.2B.2B)
    template <typename Vec3a, typename Vec3b, typename Vec3c, typename Vec3d, typename Point = pcl::PointXYZ>
    auto intersectLineAndPlane(const Vec3a ray, const Vec3b rayPoint, const Vec3c planeNormal, const Vec3d planePoint) -> Point
    {

        Vec3a diff = impl_::sub(rayPoint, planePoint);
        float prod1 = impl_::dot(diff, planeNormal);
        float prod2 = impl_::dot(ray, planeNormal);
        float prod3 = prod1 / prod2;
        Vec3a final = impl_::sub(rayPoint, impl_::mult(ray, prod3));

        Point out{};
        out.x = final.x;
        out.y = final.y;
        out.z = final.z;

        return out;
    }

}
