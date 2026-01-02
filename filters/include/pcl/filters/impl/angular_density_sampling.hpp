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

#include <pcl/filters/angular_density_sampling.h>
#include <pcl/common/point_tests.h>
#include <pcl/console/print.h>
#include <cmath>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::AngularDensitySampling<PointT>::applyFilter (Indices &indices)
{
  if (!initCompute ())
  {
    indices.clear ();
    return;
  }

  // Validate that required parameters are set
  if (azimuth_increment_ == 0.0f || elevation_increment_ == 0.0f)
  {
    PCL_ERROR ("[pcl::%s::applyFilter] Angle increments not set! Call setAngleIncrements() first.\n", getClassName ().c_str ());
    indices.clear ();
    return;
  }

  if (min_spacing_ <= 0.0f)
  {
    PCL_ERROR ("[pcl::%s::applyFilter] Minimum spacing not set or invalid! Call setMinSpacing() with a positive value.\n", getClassName ().c_str ());
    indices.clear ();
    return;
  }

  if (cloud_width_ == 0 || cloud_height_ == 0)
  {
    PCL_ERROR ("[pcl::%s::applyFilter] Cloud dimensions not set! Make sure setInputCloud() was called with a valid organized cloud.\n", getClassName ().c_str ());
    indices.clear ();
    return;
  }

  const float spacing_sq = min_spacing_ * min_spacing_;

  kept_mask_.assign (cloud_width_ * cloud_height_, false);
  indices.clear ();
  removed_indices_->clear ();

  for (std::uint32_t h = 0; h < cloud_height_; ++h)
  {
    for (std::uint32_t w = 0; w < cloud_width_; ++w)
    {
      const std::uint32_t idx = h * cloud_width_ + w;
      const PointT& pt = (*input_)[idx];

      // Skip invalid points (0,0,0)
      if (pt.x == 0 && pt.y == 0 && pt.z == 0)
      {
        continue;
      }

      // Compute range
      const float range = std::sqrt (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);

      // Compute pixel neighborhood size
      const float azimuth_pixel_spacing = 2.0f * range * tan_half_azimuth_inc_;
      const float elevation_pixel_spacing = 2.0f * range * tan_half_elevation_inc_;

      // If pixel spacing is larger than min_spacing, all neighbors are further away
      // so we can immediately keep this point without checking neighbors
      if (azimuth_pixel_spacing > min_spacing_ && elevation_pixel_spacing > min_spacing_)
      {
        kept_mask_[idx] = true;
        indices.push_back (idx);
        continue;
      }

      const int delta_w = static_cast<int> (std::ceil (min_spacing_ / azimuth_pixel_spacing));
      const int delta_h = static_cast<int> (std::ceil (min_spacing_ / elevation_pixel_spacing));

      // Check neighborhood for any kept point within min_spacing
      bool should_keep = true;

      const int h_min = std::max (0, static_cast<int> (h) - delta_h);
      const int h_max = std::min (static_cast<int> (cloud_height_) - 1, static_cast<int> (h) + delta_h);
      const int w_min = std::max (0, static_cast<int> (w) - delta_w);
      const int w_max = std::min (static_cast<int> (cloud_width_) - 1, static_cast<int> (w) + delta_w);

      for (int nh = h_min; nh <= h_max && should_keep; ++nh)
      {
        for (int nw = w_min; nw <= w_max && should_keep; ++nw)
        {
          const std::uint32_t neighbor_idx = nh * cloud_width_ + nw;

          if (!kept_mask_[neighbor_idx])
            continue;

          const PointT& neighbor = (*input_)[neighbor_idx];

          // Squared Euclidean distance check
          const float dist_sq = (neighbor.getVector3fMap() - pt.getVector3fMap()).squaredNorm();

          if (dist_sq < spacing_sq)
          {
            should_keep = false;
            goto after_loop;
          }
        }
      }
  after_loop:

      if (should_keep)
      {
        kept_mask_[idx] = true;
        indices.push_back (idx);
      }
      else if (extract_removed_indices_)
      {
        removed_indices_->push_back (idx);
      }
    }
  }

  // Handle negative_ flag
  if (negative_)
  {
    indices.swap (*removed_indices_);
  }

  deinitCompute ();
}

#define PCL_INSTANTIATE_AngularDensitySampling(T) template class PCL_EXPORTS pcl::AngularDensitySampling<T>;

#endif    // PCL_FILTERS_IMPL_ANGULAR_DENSITY_SAMPLING_H_

