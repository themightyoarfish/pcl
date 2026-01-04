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

#ifndef PCL_FILTERS_IMPL_ANGULAR_DENSITY_SAMPLING_H_
#define PCL_FILTERS_IMPL_ANGULAR_DENSITY_SAMPLING_H_

#include <pcl/common/point_tests.h>
#include <pcl/console/print.h>
#include <pcl/filters/angular_density_sampling.h>

#include <algorithm>
#include <cmath>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void
pcl::AngularDensitySampling<PointT>::applyFilter(Indices& indices)
{
  if (!initCompute()) {
    indices.clear();
    return;
  }

  // Validate that required parameters are set
  if (azimuth_increment_ == 0.0f || elevation_increment_ == 0.0f) {
    PCL_ERROR("[pcl::%s::applyFilter] Angle increments not set! Call "
              "setAngleIncrements() first.\n",
              getClassName().c_str());
    indices.clear();
    return;
  }

  if (approx_voxel_size_ <= 0.0f) {
    PCL_ERROR("[pcl::%s::applyFilter] Approximate voxel size not set or invalid! Call "
              "setApproxVoxelSize() with a positive value.\n",
              getClassName().c_str());
    indices.clear();
    return;
  }

  if (cloud_width_ == 0 || cloud_height_ == 0) {
    PCL_ERROR("[pcl::%s::applyFilter] Cloud dimensions not set! Make sure "
              "setInputCloud() was called with a valid organized cloud.\n",
              getClassName().c_str());
    indices.clear();
    return;
  }

  // When keep_organized_ is true, we need extract_removed_indices_ to be true
  // The base class will set it, but we need it set before we build removed_indices_
  if (keep_organized_ && !extract_removed_indices_) {
    extract_removed_indices_ = true;
  }

  const float approx_voxel_size_sq = approx_voxel_size_ * approx_voxel_size_;

  removed_mask_.assign(cloud_width_ * cloud_height_, false);
  indices.clear();
  indices.reserve(cloud_width_ * cloud_height_);
  removed_indices_->clear();
  removed_indices_->reserve(cloud_width_ * cloud_height_);

  for (std::uint32_t h = 0; h < cloud_height_; ++h) {
    for (std::uint32_t w = 0; w < cloud_width_; ++w) {
      const std::uint32_t idx = h * cloud_width_ + w;

      // blacklisted by previous iterations
      if (removed_mask_[idx]) {
        continue;
      }

      const PointT& pt = (*input_)[idx];

      // Skip invalid points (0,0,0)
      if (pt.x == 0 && pt.y == 0 && pt.z == 0) {
        continue;
      }

      // TODO: can sqrt be avoided? since sqrt(a/b) = sqrt(a) / sqrt(b) and sqrt(a*b) =
      // sqrt(a) * sqrt(b), we should be able to compute the pixel spacing without the
      // sqrt. But could not get it to work.
      const float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);

      // tan(azimuth_increment / 2) = (d /  2) / range
      // d = 2 * range * tan(azimuth_increment / 2)

      // Compute pixel neighborhood size at this distance
      const float azimuth_pixel_spacing_m = range * 2.0f * tan_half_azimuth_inc_;
      const float elevation_pixel_spacing_m = range * 2.0f * tan_half_elevation_inc_;

      // If pixel spacing is larger than approx_voxel_size, all neighbors are further
      // away so we can immediately keep this point without checking neighbors, as there
      // cannot be any inside this voxel
      if (azimuth_pixel_spacing_m > approx_voxel_size_ / 2 &&
          elevation_pixel_spacing_m > approx_voxel_size_ / 2) {
        indices.push_back(idx);
        continue;
      }

      const int delta_w =
          static_cast<int>(std::ceil(approx_voxel_size_ / azimuth_pixel_spacing_m));
      const int delta_h =
          static_cast<int>(std::ceil(approx_voxel_size_ / elevation_pixel_spacing_m));

      const int h_min = std::max(0, static_cast<int>(h) - delta_h);
      const int h_max =
          std::min(static_cast<int>(cloud_height_) - 1, static_cast<int>(h) + delta_h);
      const int w_min = std::max(0, static_cast<int>(w) - delta_w);
      const int w_max =
          std::min(static_cast<int>(cloud_width_) - 1, static_cast<int>(w) + delta_w);

      // Keep this point, and mark violating neighbors as removed
      indices.push_back(idx);

      for (int nh = h_min; nh <= h_max; ++nh) {
        for (int nw = w_min; nw <= w_max; ++nw) {
          const std::uint32_t neighbor_idx = nh * cloud_width_ + nw;

          if (removed_mask_[neighbor_idx])
            continue;

          if (neighbor_idx == idx)
            continue;

          const PointT& neighbor = (*input_)[neighbor_idx];

          if (neighbor.x == 0 && neighbor.y == 0 && neighbor.z == 0)
            continue;

          const float dist_sq =
              (neighbor.getVector3fMap() - pt.getVector3fMap()).squaredNorm();

          if (dist_sq < approx_voxel_size_sq) {
            removed_mask_[neighbor_idx] = true;
            if (extract_removed_indices_)
              removed_indices_->push_back(neighbor_idx);
          }
        }
      }
    }
  }

  // Handle negative_ flag
  // When negative_ is true, we swap indices and removed_indices_ so that:
  // - indices contains points that should be kept (normally removed)
  // - removed_indices_ contains points that should be removed (normally kept)
  // This works for both keep_organized_ true and false cases:
  // - When keep_organized_ is false: swapped indices are used for output
  // - When keep_organized_ is true: base class sets removed_indices_ to
  // user_filter_value_
  if (negative_) {
    indices.swap(*removed_indices_);
  }

  deinitCompute();
}

#define PCL_INSTANTIATE_AngularDensitySampling(T)                                      \
  template class PCL_EXPORTS pcl::AngularDensitySampling<T>;

#endif // PCL_FILTERS_IMPL_ANGULAR_DENSITY_SAMPLING_H_
