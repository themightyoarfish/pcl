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

#include <pcl/test/gtest.h>
#include <pcl/filters/uniform_sampling_search.h>
#include <pcl/search/brute_force.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/distances.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <set>

using namespace pcl;

/** \brief Generate a spherical point cloud
  * \param[out] cloud output point cloud
  * \param[in] center center of the sphere
  * \param[in] radius radius of the sphere
  * \param[in] num_points number of points to generate
  * \param[in] seed random seed
  */
void
generateSphericalCloud (PointCloud<PointXYZ>::Ptr& cloud,
                        const PointXYZ& center,
                        float radius,
                        unsigned int num_points,
                        unsigned int seed = 42)
{
  cloud->clear ();
  cloud->resize (num_points);
  cloud->width = num_points;
  cloud->height = 1;
  cloud->is_dense = true;

  std::mt19937 rng (seed);
  std::uniform_real_distribution<float> u01 (0.0f, 1.0f);

  for (unsigned int i = 0; i < num_points; ++i)
  {
    // Generate uniform random points on sphere surface using spherical coordinates
    // For uniform distribution on sphere: theta in [0, 2*pi], phi in [0, pi]
    float theta = 2.0f * static_cast<float> (M_PI) * u01 (rng);
    float phi = std::acos (1.0f - 2.0f * u01 (rng));  // acos(1-2u) gives uniform distribution
    float r = radius * std::cbrt (u01 (rng));  // cbrt for uniform volume distribution

    (*cloud)[i].x = center.x + r * std::sin (phi) * std::cos (theta);
    (*cloud)[i].y = center.y + r * std::sin (phi) * std::sin (theta);
    (*cloud)[i].z = center.z + r * std::cos (phi);
  }
}

/** \brief Brute force k-nearest neighbor search
  * \param[in] cloud input point cloud
  * \param[in] query_point query point
  * \param[in] k number of neighbors
  * \param[out] indices output indices
  * \param[out] distances output squared distances
  */
void
bruteForceKSearch (const PointCloud<PointXYZ>::Ptr& cloud,
                   const PointXYZ& query_point,
                   int k,
                   Indices& indices,
                   std::vector<float>& distances)
{
  indices.clear ();
  distances.clear ();

  struct Candidate
  {
    index_t idx;
    float dist_sq;

    Candidate (index_t i, float d) : idx (i), dist_sq (d) {}

    bool operator< (const Candidate& other) const
    {
      return dist_sq < other.dist_sq;
    }
  };

  std::vector<Candidate> candidates;
  candidates.reserve (cloud->size ());

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    float dist_sq = squaredEuclideanDistance (query_point, (*cloud)[i]);
    candidates.push_back (Candidate (static_cast<index_t> (i), dist_sq));
  }

  std::partial_sort (candidates.begin (),
                     candidates.begin () + std::min (static_cast<std::size_t> (k), candidates.size ()),
                     candidates.end ());

  const std::size_t result_size = std::min (static_cast<std::size_t> (k), candidates.size ());
  indices.resize (result_size);
  distances.resize (result_size);

  for (std::size_t i = 0; i < result_size; ++i)
  {
    indices[i] = candidates[i].idx;
    distances[i] = candidates[i].dist_sq;
  }
}

/** \brief Brute force radius search
  * \param[in] cloud input point cloud
  * \param[in] query_point query point
  * \param[in] radius search radius
  * \param[out] indices output indices
  * \param[out] distances output squared distances
  */
void
bruteForceRadiusSearch (const PointCloud<PointXYZ>::Ptr& cloud,
                        const PointXYZ& query_point,
                        double radius,
                        Indices& indices,
                        std::vector<float>& distances)
{
  indices.clear ();
  distances.clear ();

  const double radius_sq = radius * radius;

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    float dist_sq = squaredEuclideanDistance (query_point, (*cloud)[i]);
    if (dist_sq <= radius_sq)
    {
      indices.push_back (static_cast<index_t> (i));
      distances.push_back (dist_sq);
    }
  }

  // Sort by distance
  std::vector<std::pair<index_t, float> > sorted_results;
  sorted_results.reserve (indices.size ());
  for (std::size_t i = 0; i < indices.size (); ++i)
  {
    sorted_results.push_back (std::make_pair (indices[i], distances[i]));
  }
  std::sort (sorted_results.begin (), sorted_results.end (),
             [] (const std::pair<index_t, float>& a, const std::pair<index_t, float>& b)
             { return a.second < b.second; });

  indices.resize (sorted_results.size ());
  distances.resize (sorted_results.size ());
  for (std::size_t i = 0; i < sorted_results.size (); ++i)
  {
    indices[i] = sorted_results[i].first;
    distances[i] = sorted_results[i].second;
  }
}

TEST (PCL, UniformSamplingSearch_NearestKSearch)
{
  constexpr unsigned int num_points = 5000;
  constexpr float sphere_radius = 5.0f;
  constexpr float voxel_size = 0.2f;
  constexpr unsigned int num_tests = 10;

  PointCloud<PointXYZ>::Ptr original_cloud (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr filtered_cloud (new PointCloud<PointXYZ>);

  // Generate spherical point cloud centered at origin
  PointXYZ center (0.0f, 0.0f, 0.0f);
  generateSphericalCloud (original_cloud, center, sphere_radius, num_points);

  // Create UniformSamplingSearch
  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud (original_cloud);
  search.setRadiusSearch (voxel_size);

  // Apply filter to populate leaves_
  Indices filtered_indices;
  search.filter (filtered_indices);

  // Extract filtered cloud for brute force comparison
  pcl::copyPointCloud (*original_cloud, filtered_indices, *filtered_cloud);

  // Test multiple queries
  std::mt19937 rng (12345);
  std::uniform_real_distribution<float> coord_dist (-sphere_radius * 1.5f, sphere_radius * 1.5f);

  for (unsigned int test = 0; test < num_tests; ++test)
  {
    // Generate random query point
    PointXYZ query_point (coord_dist (rng), coord_dist (rng), coord_dist (rng));

    // Random k between 1 and 20
    const int k = 1 + (rng () % 20);

    // Search using UniformSamplingSearch
    Indices search_indices;
    std::vector<float> search_distances;
    int num_found = search.nearestKSearch (query_point, k, search_indices, search_distances);

    // Brute force search on filtered cloud
    Indices bf_indices;
    std::vector<float> bf_distances;
    bruteForceKSearch (filtered_cloud, query_point, k, bf_indices, bf_distances);

    // Verify results
    ASSERT_EQ (num_found, static_cast<int> (bf_indices.size ()));
    ASSERT_EQ (search_indices.size (), bf_indices.size ());
    ASSERT_EQ (search_distances.size (), bf_distances.size ());

    // Check that distances match and points are the same
    // Note: search_indices are indices into original_cloud, bf_indices are indices into filtered_cloud
    // filtered_cloud[i] corresponds to original_cloud[filtered_indices[i]]
    ASSERT_EQ (search_indices.size (), bf_indices.size ());

    for (std::size_t i = 0; i < search_indices.size (); ++i)
    {
      // Verify distance matches
      EXPECT_NEAR (search_distances[i], bf_distances[i], 1e-4f);

      // Verify that the point at search_indices[i] in original cloud matches
      // the point at bf_indices[i] in filtered cloud
      const PointXYZ& search_pt = (*original_cloud)[search_indices[i]];
      const PointXYZ& bf_pt = (*filtered_cloud)[bf_indices[i]];

      EXPECT_NEAR (search_pt.x, bf_pt.x, 1e-4f);
      EXPECT_NEAR (search_pt.y, bf_pt.y, 1e-4f);
      EXPECT_NEAR (search_pt.z, bf_pt.z, 1e-4f);
    }

    // Verify distances are sorted
    for (std::size_t i = 1; i < search_distances.size (); ++i)
    {
      EXPECT_LE (search_distances[i - 1], search_distances[i]);
    }
  }
}

TEST (PCL, UniformSamplingSearch_RadiusSearch)
{
  constexpr unsigned int num_points = 5000;
  constexpr float sphere_radius = 5.0f;
  constexpr float voxel_size = 0.2f;
  constexpr unsigned int num_tests = 10;

  PointCloud<PointXYZ>::Ptr original_cloud (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr filtered_cloud (new PointCloud<PointXYZ>);

  // Generate spherical point cloud centered at origin
  PointXYZ center (0.0f, 0.0f, 0.0f);
  generateSphericalCloud (original_cloud, center, sphere_radius, num_points);

  // Create UniformSamplingSearch
  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud (original_cloud);
  search.setRadiusSearch (voxel_size);
  search.setSortedResults (true);

  // Apply filter to populate leaves_
  Indices filtered_indices;
  search.filter (filtered_indices);

  // Extract filtered cloud for brute force comparison
  pcl::copyPointCloud (*original_cloud, filtered_indices, *filtered_cloud);

  // Test multiple queries
  std::mt19937 rng (54321);
  std::uniform_real_distribution<float> coord_dist (-sphere_radius * 1.5f, sphere_radius * 1.5f);
  std::uniform_real_distribution<float> radius_dist (0.5f, 2.0f);

  for (unsigned int test = 0; test < num_tests; ++test)
  {
    // Generate random query point
    PointXYZ query_point (coord_dist (rng), coord_dist (rng), coord_dist (rng));

    // Random radius
    const double search_radius = radius_dist (rng);

    // Search using UniformSamplingSearch
    Indices search_indices;
    std::vector<float> search_distances;
    int num_found = search.radiusSearch (query_point, search_radius, search_indices, search_distances);

    // Brute force search on filtered cloud
    Indices bf_indices;
    std::vector<float> bf_distances;
    bruteForceRadiusSearch (filtered_cloud, query_point, search_radius, bf_indices, bf_distances);

    // Verify results
    ASSERT_EQ (num_found, static_cast<int> (bf_indices.size ()));
    ASSERT_EQ (search_indices.size (), bf_indices.size ());
    ASSERT_EQ (search_distances.size (), bf_distances.size ());

    // Verify all points are within radius
    for (std::size_t i = 0; i < search_indices.size (); ++i)
    {
      const PointXYZ& pt = (*original_cloud)[search_indices[i]];
      float dist_sq = squaredEuclideanDistance (query_point, pt);
      EXPECT_LE (dist_sq, search_radius * search_radius + 1e-4f);
    }

    // Verify completeness: all points within radius are found
    // Create a set of found indices for efficient lookup
    std::set<index_t> found_indices_set (search_indices.begin (), search_indices.end ());
    const double radius_sq = search_radius * search_radius;

    // Check all filtered points - if they're within radius, they should be in results
    for (std::size_t i = 0; i < filtered_indices.size (); ++i)
    {
      const index_t orig_idx = filtered_indices[i];
      const PointXYZ& pt = (*original_cloud)[orig_idx];
      float dist_sq = squaredEuclideanDistance (query_point, pt);

      if (dist_sq <= radius_sq + 1e-4f)  // Within radius (with small tolerance)
      {
        EXPECT_TRUE (found_indices_set.find (orig_idx) != found_indices_set.end ())
          << "Point at index " << orig_idx << " is within radius but not found in search results. "
          << "Distance: " << std::sqrt (dist_sq) << ", Radius: " << search_radius;
      }
    }

    // Verify distances match (allowing for small differences due to filtering)
    // Since we're comparing filtered vs filtered, distances should match closely
    for (std::size_t i = 0; i < std::min (search_distances.size (), bf_distances.size ()); ++i)
    {
      EXPECT_NEAR (search_distances[i], bf_distances[i], 1e-3f);
    }

    // Verify distances are sorted
    for (std::size_t i = 1; i < search_distances.size (); ++i)
    {
      EXPECT_LE (search_distances[i - 1], search_distances[i]);
    }
  }
}

TEST (PCL, UniformSamplingSearch_MaxSearchRadius)
{
  constexpr unsigned int num_points = 3000;
  constexpr float sphere_radius = 5.0f;
  constexpr float voxel_size = 0.2f;
  constexpr double max_search_radius = 1.5;

  PointCloud<PointXYZ>::Ptr original_cloud (new PointCloud<PointXYZ>);

  // Generate spherical point cloud
  PointXYZ center (0.0f, 0.0f, 0.0f);
  generateSphericalCloud (original_cloud, center, sphere_radius, num_points);

  // Create UniformSamplingSearch with max search radius
  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud (original_cloud);
  search.setRadiusSearch (voxel_size);
  search.setMaxSearchRadius (max_search_radius);

  // Apply filter
  Indices filtered_indices;
  search.filter (filtered_indices);

  // Test radius search with radius larger than max_search_radius
  PointXYZ query_point (0.0f, 0.0f, 0.0f);
  const double search_radius = 3.0;  // Larger than max_search_radius

  Indices search_indices;
  std::vector<float> search_distances;
   search.radiusSearch (query_point, search_radius, search_indices, search_distances);

  // Verify that results are limited by max_search_radius
  for (std::size_t i = 0; i < search_indices.size (); ++i)
  {
    const PointXYZ& pt = (*original_cloud)[search_indices[i]];
    float dist = std::sqrt (squaredEuclideanDistance (query_point, pt));
    EXPECT_LE (dist, max_search_radius + 1e-4f);
  }
}

TEST (PCL, UniformSamplingSearch_EmptyCloud)
{
  PointCloud<PointXYZ>::Ptr empty_cloud (new PointCloud<PointXYZ>);
  empty_cloud->width = 0;
  empty_cloud->height = 1;

  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud (empty_cloud);
  search.setRadiusSearch (0.1);

  // Apply filter on empty cloud
  Indices filtered_indices;
  search.filter (filtered_indices);

  // Search should return 0 results
  PointXYZ query_point (0.0f, 0.0f, 0.0f);
  Indices indices;
  std::vector<float> distances;

  int num_found_k = search.nearestKSearch (query_point, 10, indices, distances);
  EXPECT_EQ (num_found_k, 0);

  int num_found_r = search.radiusSearch (query_point, 1.0, indices, distances);
  EXPECT_EQ (num_found_r, 0);
}

TEST (PCL, UniformSamplingSearch_FilterNotApplied)
{
  PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);
  cloud->push_back (PointXYZ (0.0f, 0.0f, 0.0f));
  cloud->push_back (PointXYZ (1.0f, 0.0f, 0.0f));
  cloud->push_back (PointXYZ (0.0f, 1.0f, 0.0f));

  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud (cloud);
  search.setRadiusSearch (0.1);

  // Don't apply filter - leaves_ should be empty
  PointXYZ query_point (0.0f, 0.0f, 0.0f);
  Indices indices;
  std::vector<float> distances;

  int num_found_k = search.nearestKSearch (query_point, 10, indices, distances);
  EXPECT_EQ (num_found_k, 0);

  int num_found_r = search.radiusSearch (query_point, 1.0, indices, distances);
  EXPECT_EQ (num_found_r, 0);
}

/* ---[ */
int
main (int argc, char** argv)
{
  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
/* ]--- */

