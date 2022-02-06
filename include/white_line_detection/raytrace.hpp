#pragma once

#include "pcl/point_types.h"

namespace raytracing
{

    // Define some generic vector ops
    namespace impl_
    {
        /// Dot product
        template <typename Vec3>
        auto dot(const Vec3 lhs, const Vec3 rhs) -> float
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }

        /// Vector subtraction
        template <typename Vec3>
        auto sub(const Vec3 lhs, const Vec3 rhs) -> Vec3
        {
            auto out = Vec3{};
            out.x = lhs.x - rhs.x;
            out.y = lhs.y - rhs.y;
            out.z = lhs.z - rhs.z;

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
        Vec3a diff = sub(rayPoint, planePoint);
        float prod1 = dot(diff, planeNormal);
        float prod2 = dot(ray, planeNormal);
        float prod3 = prod1 / prod2;
        Vec3a final = sub(rayPoint, mult(ray, prod3));

        Point out{};
        out.x = final.x;
        out.y = final.y;
        out.z = final.z;

        return out;
    }

}
