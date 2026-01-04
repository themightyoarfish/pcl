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

#include <pcl/search/search.h>
#include <pcl/filters/uniform_sampling.h>
#include <limits>

namespace pcl
{
  /** \brief @b UniformSamplingSearch combines UniformSampling filtering with efficient neighbor search.
    *
    * The @b UniformSamplingSearch class uses the voxel grid structure created by UniformSampling
    * to efficiently find nearest neighbors. It searches by expanding through voxel shells
    * (for k-nearest) or checking voxels within a radius (for radius search).
    *
    * \note The filter must be applied (via filter() or applyFilter()) before search queries can be performed.
    *
    * \author Point Cloud Library contributors
    * \ingroup filters
    */
  template <typename PointT>
  class UniformSamplingSearch : public UniformSampling<PointT>,
                                 public search::Search<PointT>
  {
    using PointCloud = typename search::Search<PointT>::PointCloud;
    using PointCloudConstPtr = typename search::Search<PointT>::PointCloudConstPtr;
    using PointCloudPtr = typename search::Search<PointT>::PointCloudPtr;
    using IndicesPtr = pcl::IndicesPtr;
    using IndicesConstPtr = pcl::IndicesConstPtr;

    using UniformSampling<PointT>::leaves_;
    using UniformSampling<PointT>::leaf_size_;
    using UniformSampling<PointT>::inverse_leaf_size_;
    using UniformSampling<PointT>::min_b_;
    using UniformSampling<PointT>::max_b_;
    using UniformSampling<PointT>::div_b_;
    using UniformSampling<PointT>::divb_mul_;
    using search::Search<PointT>::sorted_results_;

  public:
    using Ptr = shared_ptr<UniformSamplingSearch<PointT> >;
    using ConstPtr = shared_ptr<const UniformSamplingSearch<PointT> >;

    PCL_MAKE_ALIGNED_OPERATOR_NEW

    /** \brief Constructor.
      * \param[in] extract_removed_indices Set to true if you want to extract removed indices (default: false)
      * \param[in] sorted_results Set to true if search results should be sorted by distance (default: false)
      */
    UniformSamplingSearch (bool extract_removed_indices = false, bool sorted_results = false)
      : UniformSampling<PointT> (extract_removed_indices),
        search::Search<PointT> ("UniformSamplingSearch", sorted_results),
        max_search_radius_ (std::numeric_limits<double>::max ())
    {
    }

    /** \brief Destructor. */
    ~UniformSamplingSearch () override = default;

    /** \brief Set the input cloud for both filtering and searching.
      * \param[in] cloud the input point cloud
      */
    void
    setInputCloud (const PointCloudConstPtr& cloud) override
    {
      // Set input for UniformSampling (via Filter base)
      UniformSampling<PointT>::setInputCloud (cloud);
      // Set input for Search base
      search::Search<PointT>::setInputCloud (cloud);
    }

    /** \brief Set the maximum search radius for limiting voxel neighborhood expansion.
      * \param[in] radius maximum search radius (default: unlimited)
      */
    inline void
    setMaxSearchRadius (double radius)
    {
      max_search_radius_ = radius;
    }

    /** \brief Get the maximum search radius.
      * \return maximum search radius
      */
    inline double
    getMaxSearchRadius () const
    {
      return max_search_radius_;
    }

    /** \brief Search for the k-nearest neighbors for the given query point.
      * \param[in] point the given query point
      * \param[in] k the number of neighbors to search for
      * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
      * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k a priori!)
      * \return number of neighbors found
      */
    int
    nearestKSearch (const PointT &point, int k,
                    Indices &k_indices, std::vector<float> &k_sqr_distances) const override;

    /** \brief Search for all the nearest neighbors of the query point in a given radius.
      * \param[in] point the given query point
      * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
      * \param[out] k_indices the resultant indices of the neighboring points
      * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
      * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
      * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
      * returned.
      * \return number of neighbors found in radius
      */
    int
    radiusSearch (const PointT& point, double radius,
                  Indices& k_indices, std::vector<float>& k_sqr_distances,
                  unsigned int max_nn = 0) const override;

  protected:
    /** \brief Maximum search radius for limiting voxel neighborhood expansion. */
    double max_search_radius_;

    /** \brief Helper function to compute voxel index from a point.
      * \param[in] point the query point
      * \return linear voxel index
      */
    std::size_t
    getVoxelIndex (const PointT& point) const;
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/filters/impl/uniform_sampling_search.hpp>
#endif

