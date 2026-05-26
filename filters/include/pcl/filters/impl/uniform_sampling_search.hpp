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
bool
pcl::UniformSamplingSearch<PointT>::lookupVoxel(std::size_t voxel_idx,
                                              VoxelSearchEntry& entry) const
{
  if (use_dense_voxel_grid_) {
    if (voxel_idx >= voxel_search_grid_.size()) {
      return false;
    }
    entry = voxel_search_grid_[voxel_idx];
    return entry.orig_idx >= 0;
  }

  const auto it = voxel_search_map_.find(voxel_idx);
  if (it == voxel_search_map_.end()) {
    return false;
  }
  entry = it->second;
  return entry.orig_idx >= 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
float
pcl::UniformSamplingSearch<PointT>::minSquaredDistanceToVoxel(const PointT& point,
                                                            const Eigen::Vector4i& ijk,
                                                            float leaf_size)
{
  const float vx0 = static_cast<float>(ijk[0]) * leaf_size;
  const float vy0 = static_cast<float>(ijk[1]) * leaf_size;
  const float vz0 = static_cast<float>(ijk[2]) * leaf_size;
  const float vx1 = vx0 + leaf_size;
  const float vy1 = vy0 + leaf_size;
  const float vz1 = vz0 + leaf_size;

  const float dx =
      (point.x < vx0) ? vx0 - point.x : ((point.x > vx1) ? point.x - vx1 : 0.f);
  const float dy =
      (point.y < vy0) ? vy0 - point.y : ((point.y > vy1) ? point.y - vy1 : 0.f);
  const float dz =
      (point.z < vz0) ? vz0 - point.z : ((point.z > vz1) ? point.z - vz1 : 0.f);
  return dx * dx + dy * dy + dz * dz;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void
pcl::UniformSamplingSearch<PointT>::applyFilter(Indices& indices)
{
  UniformSampling<PointT>::applyFilter(indices);

  voxel_search_grid_.clear();
  voxel_search_map_.clear();
  use_dense_voxel_grid_ = false;

  const std::size_t bx = static_cast<std::size_t>(div_b_[0]);
  const std::size_t by = static_cast<std::size_t>(div_b_[1]);
  const std::size_t bz = static_cast<std::size_t>(div_b_[2]);
  std::size_t total_voxels = 0;
  if (bx > 0 && by > 0 && bz > 0) {
    const std::size_t bxy = bx * by;
    if (bxy / bx == by && bxy * bz / bxy == bz) {
      total_voxels = bxy * bz;
    }
  }

  if (total_voxels > 0 && total_voxels <= dense_grid_max_voxels_) {
    use_dense_voxel_grid_ = true;
    voxel_search_grid_.assign(total_voxels, VoxelSearchEntry{});
  }
  else {
    voxel_search_map_.reserve(leaves_.size());
  }

  auto input_cloud = search::Search<PointT>::getInputCloud();

  for (index_t filtered_idx = 0; filtered_idx < static_cast<index_t>(indices.size());
       ++filtered_idx) {
    const index_t orig_idx = indices[filtered_idx];
    const PointT& pt = (*input_cloud)[orig_idx];

    Eigen::Vector4i ijk = Eigen::Vector4i::Zero();
    ijk[0] = static_cast<int>(std::floor(pt.x * inverse_leaf_size_[0]));
    ijk[1] = static_cast<int>(std::floor(pt.y * inverse_leaf_size_[1]));
    ijk[2] = static_cast<int>(std::floor(pt.z * inverse_leaf_size_[2]));

    const Eigen::Vector4i relative_ijk = ijk - min_b_;
    const std::size_t voxel_idx =
        static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

    const VoxelSearchEntry entry{orig_idx, filtered_idx};
    if (use_dense_voxel_grid_) {
      if (voxel_idx < voxel_search_grid_.size()) {
        voxel_search_grid_[voxel_idx] = entry;
      }
    }
    else {
      voxel_search_map_[voxel_idx] = entry;
    }
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
        const Eigen::Vector4i neighbor_ijk =
            query_ijk + Eigen::Vector4i(di, dj, dk, 0);

        // Sphere culling: skip voxels whose AABB is entirely outside the search sphere
        if (minSquaredDistanceToVoxel(point, neighbor_ijk, leaf_size) > radius_sq)
          continue;

        const Eigen::Vector4i relative_ijk = neighbor_ijk - min_b_;

        if (relative_ijk[0] < 0 || relative_ijk[0] >= div_b_[0] ||
            relative_ijk[1] < 0 || relative_ijk[1] >= div_b_[1] ||
            relative_ijk[2] < 0 || relative_ijk[2] >= div_b_[2])
          continue;

        const std::size_t voxel_idx =
            static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

        VoxelSearchEntry entry;
        if (!lookupVoxel(voxel_idx, entry))
          continue;

        if (entry.orig_idx < 0 ||
            entry.orig_idx >= static_cast<index_t>(input_cloud->size()))
          continue;

        const PointT& neighbor = (*input_cloud)[entry.orig_idx];

        if (!pcl::isXYZFinite(neighbor))
          continue;

        const float dist_sq =
            (neighbor.getVector3fMap() - point.getVector3fMap()).squaredNorm();

        if (dist_sq <= radius_sq) {
          k_indices.push_back(entry.filtered_idx);
          k_sqr_distances.push_back(dist_sq);

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

  struct Candidate {
    index_t filtered_idx;
    float dist_sq;

    Candidate(index_t f, float d) : filtered_idx(f), dist_sq(d) {}

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

          const Eigen::Vector4i neighbor_ijk =
              query_ijk + Eigen::Vector4i(di, dj, dk, 0);

          if (max_shell < std::numeric_limits<int>::max()) {
            const float max_radius_sq =
                static_cast<float>(max_search_radius_ * max_search_radius_);
            if (minSquaredDistanceToVoxel(point, neighbor_ijk, leaf_size) >
                max_radius_sq)
              continue;
          }

          const Eigen::Vector4i relative_ijk = neighbor_ijk - min_b_;

          if (relative_ijk[0] < 0 || relative_ijk[0] >= div_b_[0] ||
              relative_ijk[1] < 0 || relative_ijk[1] >= div_b_[1] ||
              relative_ijk[2] < 0 || relative_ijk[2] >= div_b_[2])
            continue;

          const std::size_t voxel_idx =
              static_cast<std::size_t>(relative_ijk.dot(divb_mul_));

          VoxelSearchEntry entry;
          if (!lookupVoxel(voxel_idx, entry))
            continue;

          if (entry.orig_idx < 0 ||
              entry.orig_idx >= static_cast<index_t>(input_cloud->size()))
            continue;

          const PointT& neighbor = (*input_cloud)[entry.orig_idx];

          if (!pcl::isXYZFinite(neighbor))
            continue;

          const float dist_sq =
              (neighbor.getVector3fMap() - point.getVector3fMap()).squaredNorm();

          candidates.push_back(Candidate(entry.filtered_idx, dist_sq));
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
    k_indices[i] = candidates[i].filtered_idx;
    k_sqr_distances[i] = candidates[i].dist_sq;
  }

  return static_cast<int>(result_size);
}

#define PCL_INSTANTIATE_UniformSamplingSearch(T)                                       \
  template class PCL_EXPORTS pcl::UniformSamplingSearch<T>;

#endif // PCL_FILTERS_UNIFORM_SAMPLING_SEARCH_IMPL_H_
