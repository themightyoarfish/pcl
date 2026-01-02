/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2024, Point Cloud Library contributors
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/filters/filter_indices.h>
#include <vector>
#include <cstdint>
#include <cmath>

namespace pcl
{
  /** \brief @b AngularDensitySampling subsamples organized LIDAR point clouds
   * using angular geometry to an approximate voxel size. It typically creates a
   * bit coarser clouds than VoxelGrid for the same voxel size
   *
   * The @b AngularDensitySampling class operates on organized HxW point clouds from rotating LIDARs where the beam angle increments are known.
   * It uses precomputed tangent factors to efficiently determine pixel neighborhoods at any range,
   * removing points that are less than the approximate voxel size away from other points.
   *
   * \author R. Diederichsen
   * \ingroup filters
   */
  template <typename PointT>
  class AngularDensitySampling : public FilterIndices<PointT>
  {
    using PointCloud = typename FilterIndices<PointT>::PointCloud;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    using Filter<PointT>::filter_name_;
    using Filter<PointT>::input_;
    using Filter<PointT>::indices_;
    using Filter<PointT>::removed_indices_;
    using Filter<PointT>::extract_removed_indices_;
    using Filter<PointT>::getClassName;
    using FilterIndices<PointT>::negative_;

    public:
      using Ptr = shared_ptr<AngularDensitySampling<PointT> >;
      using ConstPtr = shared_ptr<const AngularDensitySampling<PointT> >;

      /** \brief Constructor.
        * \param[in] extract_removed_indices Set to true if you want to be able to extract the indices of points being removed (default = false).
        */
      AngularDensitySampling (bool extract_removed_indices = false) :
        FilterIndices<PointT> (extract_removed_indices),
        azimuth_increment_ (0.0f),
        elevation_increment_ (0.0f),
        approx_voxel_size_ (0.0f),
        cloud_width_ (0),
        cloud_height_ (0),
        tan_half_azimuth_inc_ (0.0f),
        tan_half_elevation_inc_ (0.0f)
      {
        filter_name_ = "AngularDensitySampling";
      }

      /** \brief Destructor. */
      ~AngularDensitySampling () override = default;

      /** \brief Provide a pointer to the input dataset.
        * \param[in] cloud the const boost shared pointer to a PointCloud message
        */
      void
      setInputCloud (const PointCloudConstPtr &cloud) override
      {
        Filter<PointT>::setInputCloud (cloud);
        if (cloud)
        {
          cloud_width_ = cloud->width;
          cloud_height_ = cloud->height;
        }
        else
        {
          cloud_width_ = 0;
          cloud_height_ = 0;
        }
      }

      /** \brief Set the angular increments for azimuth and elevation.
        * \param[in] azimuth_inc angular step between columns
        * \param[in] elevation_inc angular step between rows
        * \param[in] degrees if true, angles are in degrees; if false, angles are in radians (default: false)
        */
      inline void
      setAngleIncrements (float azimuth_inc, float elevation_inc, bool degrees = false)
      {
        if (degrees)
        {
          constexpr float deg_to_rad = static_cast<float>(M_PI) / 180.0f;
          azimuth_increment_ = azimuth_inc * deg_to_rad;
          elevation_increment_ = elevation_inc * deg_to_rad;
        }
        else
        {
          azimuth_increment_ = azimuth_inc;
          elevation_increment_ = elevation_inc;
        }
        tan_half_azimuth_inc_ = std::tan (azimuth_increment_ * 0.5f);
        tan_half_elevation_inc_ = std::tan (elevation_increment_ * 0.5f);
      }

      /** \brief Set the minimum Euclidean distance between kept points.
        * \param[in] spacing minimum spacing in meters
        */
      inline void
      setApproxVoxelSize (float approx_voxel_size)
      {
        approx_voxel_size_ = approx_voxel_size;
      }

      /** \brief Get the minimum spacing between kept points.
        * \return minimum spacing in meters
        */
      inline float
      getApproxVoxelSize () const
      {
        return approx_voxel_size_;
      }


    protected:
      using Filter<PointT>::initCompute;
      using Filter<PointT>::deinitCompute;

      /** \brief Angular step between columns in radians. */
      float azimuth_increment_;

      /** \brief Angular step between rows in radians. */
      float elevation_increment_;

      /** \brief Minimum Euclidean distance between kept points. */
      float approx_voxel_size_;

      /** \brief Number of columns (W). */
      std::uint32_t cloud_width_;

      /** \brief Number of rows (H). */
      std::uint32_t cloud_height_;

      /** \brief Precomputed tan(azimuth_increment / 2). */
      float tan_half_azimuth_inc_;

      /** \brief Precomputed tan(elevation_increment / 2). */
      float tan_half_elevation_inc_;

      /** \brief Mask tracking which points are kept during filtering. */
      std::vector<bool> kept_mask_;

      /** \brief Mask tracking which points are removed during filtering. */
      std::vector<bool> removed_mask_;

      /** \brief Filtered results are indexed by an indices array.
        * \param[out] indices The resultant indices.
        */
      void
      applyFilter (Indices &indices) override;
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/filters/impl/angular_density_sampling.hpp>
#endif

