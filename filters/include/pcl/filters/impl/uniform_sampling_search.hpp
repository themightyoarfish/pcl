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

#ifndef PCL_FILTERS_UNIFORM_SAMPLING_SEARCH_IMPL_H_
#define PCL_FILTERS_UNIFORM_SAMPLING_SEARCH_IMPL_H_

#include <pcl/common/point_tests.h>
#include <pcl/console/print.h>
#include <pcl/filters/uniform_sampling_search.h>

#include <algorithm>
#include <cmath>
#include <queue>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
std::size_t
pcl::UniformSamplingSearch<PointT>::getVoxelIndex(const PointT& point) const
{
  Eigen::Vector4i ijk = Eigen::Vector4i::Zero();
  ijk[0] = static_cast<int>(std::floor(point.x * inverse_leaf_size_[0]));
  ijk[1] = static_cast<int>(std::floor(point.y * inverse_leaf_size_[1]));
  ijk[2] = static_cast<int>(std::floor(point.z * inverse_leaf_size_[2]));

  // Compute the linear voxel index
  Eigen::Vector4i relative_ijk = ijk - min_b_;
  return static_cast<std::size_t>(relative_ijk.dot(divb_mul_));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void
pcl::UniformSamplingSearch<PointT>::applyFilter(Indices& indices)
{
  // Call base class implementation
  UniformSampling<PointT>::applyFilter(indices);

  // Build mapping from voxel index to filtered cloud index
  // The indices array contains original cloud indices in the order they will appear
  // in the filtered cloud, so we iterate over it to build the correct mapping
  voxel_to_filtered_idx_.clear();
  auto input_cloud = search::Search<PointT>::getInputCloud();

  for (index_t filtered_idx = 0; filtered_idx < static_cast<index_t>(indices.size());
       ++filtered_idx) {
    const index_t orig_idx = indices[filtered_idx];
    const PointT& pt = (*input_cloud)[orig_idx];

    // Compute voxel index for this point
    Eigen::Vector4i ijk = Eigen::Vector4i::Zero();
    ijk[0] = static_cast<int>(std::floor(pt.x * inverse_leaf_size_[0]));
    ijk[1] = static_cast<int>(std::floor(pt.y * inverse_leaf_size_[1]));
    ijk[2] = static_cast<int>(std::floor(pt.z * inverse_leaf_size_[2]));

    Eigen::Vector4i relative_ijk = ijk - min_b_;
    std::size_t voxel_idx = static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

    voxel_to_filtered_idx_[voxel_idx] = filtered_idx;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
int
pcl::UniformSamplingSearch<PointT>::radiusSearch(const PointT& point,
                                                 double radius,
                                                 Indices& k_indices,
                                                 std::vector<float>& k_sqr_distances,
                                                 unsigned int max_nn) const
{
  k_indices.clear();
  k_sqr_distances.clear();

  if (radius <= 0.0)
    return 0;

  // Check if filter has been applied
  if (leaves_.empty()) {
    PCL_WARN("[pcl::UniformSamplingSearch::radiusSearch] Filter has not been applied "
             "yet. Call filter() first.\n");
    return 0;
  }

  // Check if point is valid
  if (!pcl::isXYZFinite(point)) {
    PCL_WARN("[pcl::UniformSamplingSearch::radiusSearch] Invalid (NaN, Inf) point "
             "coordinates given.\n");
    return 0;
  }

  // Compute effective radius (limited by max_search_radius_)
  const double effective_radius = std::min(radius, max_search_radius_);
  const double radius_sq = effective_radius * effective_radius;
  const float leaf_size = leaf_size_[0]; // Assuming uniform voxel size

  // Compute query voxel coordinates
  Eigen::Vector4i query_ijk = Eigen::Vector4i::Zero();
  query_ijk[0] = static_cast<int>(std::floor(point.x * inverse_leaf_size_[0]));
  query_ijk[1] = static_cast<int>(std::floor(point.y * inverse_leaf_size_[1]));
  query_ijk[2] = static_cast<int>(std::floor(point.z * inverse_leaf_size_[2]));

  // Compute search extent in voxels
  const int delta = static_cast<int>(std::ceil(effective_radius / leaf_size));

  // Reserve space for results
  std::size_t reserve_size = max_nn > 0 ? max_nn : leaves_.size();
  k_indices.reserve(reserve_size);
  k_sqr_distances.reserve(reserve_size);

  auto input_cloud = search::Search<PointT>::getInputCloud();

  // Iterate over voxel neighborhood
  for (int di = -delta; di <= delta; ++di) {
    for (int dj = -delta; dj <= delta; ++dj) {
      for (int dk = -delta; dk <= delta; ++dk) {
        Eigen::Vector4i neighbor_ijk = query_ijk + Eigen::Vector4i(di, dj, dk, 0);
        Eigen::Vector4i relative_ijk = neighbor_ijk - min_b_;

        // Check bounds
        if (relative_ijk[0] < 0 || relative_ijk[0] >= div_b_[0] ||
            relative_ijk[1] < 0 || relative_ijk[1] >= div_b_[1] ||
            relative_ijk[2] < 0 || relative_ijk[2] >= div_b_[2])
          continue;

        // Compute linear voxel index
        std::size_t voxel_idx = static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

        // Check if voxel exists
        auto it = leaves_.find(voxel_idx);
        if (it == leaves_.end())
          continue;

        // Get point index from voxel
        const index_t point_idx = static_cast<index_t>(it->second.idx);
        if (point_idx < 0 || point_idx >= static_cast<index_t>(input_cloud->size()))
          continue;

        const PointT& neighbor = (*input_cloud)[point_idx];

        // Check if neighbor point is valid
        if (!pcl::isXYZFinite(neighbor))
          continue;

        // Compute squared distance
        const float dist_sq =
            (neighbor.getVector3fMap() - point.getVector3fMap()).squaredNorm();

        // Check if within radius
        if (dist_sq <= radius_sq) {
          // Get filtered cloud index
          auto filtered_it = voxel_to_filtered_idx_.find(voxel_idx);
          if (filtered_it == voxel_to_filtered_idx_.end())
            continue;

          k_indices.push_back(filtered_it->second);
          k_sqr_distances.push_back(dist_sq);

          // Check max_nn limit
          if (max_nn > 0 && k_indices.size() >= max_nn) {
            if (sorted_results_)
              this->sortResults(k_indices, k_sqr_distances);
            return static_cast<int>(k_indices.size());
          }
        }
      }
    }
  }

  // Sort results if requested
  if (sorted_results_)
    this->sortResults(k_indices, k_sqr_distances);

  return static_cast<int>(k_indices.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
int
pcl::UniformSamplingSearch<PointT>::nearestKSearch(
    const PointT& point,
    int k,
    Indices& k_indices,
    std::vector<float>& k_sqr_distances) const
{
  k_indices.clear();
  k_sqr_distances.clear();

  if (k < 1)
    return 0;

  // Check if filter has been applied
  if (leaves_.empty()) {
    PCL_WARN("[pcl::UniformSamplingSearch::nearestKSearch] Filter has not been applied "
             "yet. Call filter() first.\n");
    return 0;
  }

  // Check if point is valid
  if (!pcl::isXYZFinite(point)) {
    PCL_WARN("[pcl::UniformSamplingSearch::nearestKSearch] Invalid (NaN, Inf) point "
             "coordinates given.\n");
    return 0;
  }

  const float leaf_size = leaf_size_[0]; // Assuming uniform voxel size

  auto input_cloud = search::Search<PointT>::getInputCloud();

  // Compute query voxel coordinates
  Eigen::Vector4i query_ijk = Eigen::Vector4i::Zero();
  query_ijk[0] = static_cast<int>(std::floor(point.x * inverse_leaf_size_[0]));
  query_ijk[1] = static_cast<int>(std::floor(point.y * inverse_leaf_size_[1]));
  query_ijk[2] = static_cast<int>(std::floor(point.z * inverse_leaf_size_[2]));

  // Structure to hold candidate neighbors
  struct Candidate {
    std::size_t voxel_idx;
    float dist_sq;

    Candidate(std::size_t v, float d) : voxel_idx(v), dist_sq(d) {}

    bool
    operator<(const Candidate& other) const
    {
      return dist_sq < other.dist_sq;
    }
  };

  std::vector<Candidate> candidates;
  candidates.reserve(k);

  // Expand search shell by shell
  int shell = 0;
  const int max_shell =
      max_search_radius_ < std::numeric_limits<double>::max()
          ? static_cast<int>(std::ceil(max_search_radius_ / leaf_size))
          : std::numeric_limits<int>::max();

  while (true) {
    // Limit shell expansion by max_search_radius_
    if (shell > max_shell)
      break;

    // Collect all voxels in this shell (Chebyshev distance = shell)
    // For shell n, we want voxels where max(|di|, |dj|, |dk|) == n
    for (int di = -shell; di <= shell; ++di) {
      for (int dj = -shell; dj <= shell; ++dj) {
        for (int dk = -shell; dk <= shell; ++dk) {
          // Skip inner shells (already processed)
          // For shell n, we only want voxels where max(|di|, |dj|, |dk|) == n
          if (shell > 0) {
            const int max_dist = std::max({std::abs(di), std::abs(dj), std::abs(dk)});
            if (max_dist != shell)
              continue;
          }

          Eigen::Vector4i neighbor_ijk = query_ijk + Eigen::Vector4i(di, dj, dk, 0);
          Eigen::Vector4i relative_ijk = neighbor_ijk - min_b_;

          // Check bounds
          if (relative_ijk[0] < 0 || relative_ijk[0] >= div_b_[0] ||
              relative_ijk[1] < 0 || relative_ijk[1] >= div_b_[1] ||
              relative_ijk[2] < 0 || relative_ijk[2] >= div_b_[2])
            continue;

          // Compute linear voxel index
          std::size_t voxel_idx = static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

          // Check if voxel exists
          auto it = leaves_.find(voxel_idx);
          if (it == leaves_.end())
            continue;

          // Get point index from voxel
          const index_t point_idx = static_cast<index_t>(it->second.idx);
          if (point_idx < 0 || point_idx >= static_cast<index_t>(input_cloud->size()))
            continue;

          const PointT& neighbor = (*input_cloud)[point_idx];

          // Check if neighbor point is valid
          if (!pcl::isXYZFinite(neighbor))
            continue;

          // Compute squared distance
          const float dist_sq =
              (neighbor.getVector3fMap() - point.getVector3fMap()).squaredNorm();

          // Add candidate with voxel index
          candidates.push_back(Candidate(voxel_idx, dist_sq));
        }
      }
    }

    // After processing this shell, check if we can stop
    if (candidates.size() >= static_cast<std::size_t>(k)) {
      // Sort candidates to get k-th best distance
      std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
      const float kth_best_dist_sq = candidates[k - 1].dist_sq;

      // Compute minimum possible distance from query point to next shell
      // For shell n+1, minimum distance is approximately (n+0.5) * leaf_size - 0.5 *
      // sqrt(3) * leaf_size
      const float min_dist_to_next_shell =
          (shell + 0.5f) * leaf_size - 0.5f * std::sqrt(3.0f) * leaf_size;
      const float min_dist_to_next_shell_sq =
          min_dist_to_next_shell * min_dist_to_next_shell;

      // If next shell's minimum distance is greater than k-th best, we're done
      if (min_dist_to_next_shell_sq > kth_best_dist_sq)
        break;
    }

    ++shell;
  }

  // Sort candidates and take top k
  std::partial_sort(candidates.begin(),
                    candidates.begin() +
                        std::min(static_cast<std::size_t>(k), candidates.size()),
                    candidates.end());

  const std::size_t result_size =
      std::min(static_cast<std::size_t>(k), candidates.size());
  k_indices.resize(result_size);
  k_sqr_distances.resize(result_size);

  for (std::size_t i = 0; i < result_size; ++i) {
    // Convert voxel index to filtered cloud index
    auto filtered_it = voxel_to_filtered_idx_.find(candidates[i].voxel_idx);
    if (filtered_it != voxel_to_filtered_idx_.end()) {
      k_indices[i] = filtered_it->second;
      k_sqr_distances[i] = candidates[i].dist_sq;
    }
  }

  return static_cast<int>(result_size);
}

#define PCL_INSTANTIATE_UniformSamplingSearch(T)                                       \
  template class PCL_EXPORTS pcl::UniformSamplingSearch<T>;

#endif // PCL_FILTERS_UNIFORM_SAMPLING_SEARCH_IMPL_H_
