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

#include <pcl/common/distances.h>
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/uniform_sampling_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/brute_force.h>
#include <pcl/search/kdtree.h>
#include <pcl/test/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace pcl;

// Global cloud loaded from bun0.pcd
PointCloud<PointXYZ>::Ptr bun0_cloud;

TEST(PCL, UniformSamplingSearch_NearestKSearch)
{
  constexpr float voxel_size = 0.01f;
  constexpr unsigned int num_tests = 100;

  PointCloud<PointXYZ>::Ptr original_cloud = bun0_cloud;
  PointCloud<PointXYZ>::Ptr filtered_cloud(new PointCloud<PointXYZ>);

  // Create UniformSamplingSearch
  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud(original_cloud);
  search.setRadiusSearch(voxel_size);

  // Apply filter to populate leaves_
  Indices filtered_indices;
  search.filter(filtered_indices);

  ASSERT_GT(filtered_indices.size(), 0);

  // Extract filtered cloud for KdTree comparison
  pcl::copyPointCloud(*original_cloud, filtered_indices, *filtered_cloud);
  // Save filtered cloud to "filtered.pcd"
  pcl::io::savePCDFileBinary("filtered.pcd", *filtered_cloud);

  // Create KdTree on filtered cloud
  search::KdTree<PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);

  // Compute bounding box for query point generation
  PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);


  // Test multiple queries
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> x_dist(min_pt.x, max_pt.x);
  std::uniform_real_distribution<float> y_dist(min_pt.y, max_pt.y);
  std::uniform_real_distribution<float> z_dist(min_pt.z, max_pt.z);

  for (unsigned int test = 0; test < num_tests; ++test) {
    // Generate random query point
    PointXYZ query_point(x_dist(rng), y_dist(rng), z_dist(rng));

    const int k = 1;

    // Search using UniformSamplingSearch
    Indices search_indices;
    std::vector<float> search_distances;
    int num_found =
        search.nearestKSearch(query_point, k, search_indices, search_distances);

    // KdTree search on filtered cloud
    Indices kd_indices;
    std::vector<float> kd_distances;
    kdtree.nearestKSearch(query_point, k, kd_indices, kd_distances);
    ASSERT_EQ(kd_indices.size(), num_found);
    ASSERT_EQ(filtered_cloud->at(search_indices[0]).getVector3fMap(), filtered_cloud->at(kd_indices[0]).getVector3fMap());
  }
}

TEST(PCL, UniformSamplingSearch_RadiusSearch)
{
  constexpr float voxel_size = 0.01f;
  constexpr double search_radius = 0.03;
  constexpr unsigned int num_tests = 100;

  PointCloud<PointXYZ>::Ptr original_cloud = bun0_cloud;
  PointCloud<PointXYZ>::Ptr filtered_cloud(new PointCloud<PointXYZ>);

  // Create UniformSamplingSearch
  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud(original_cloud);
  search.setRadiusSearch(voxel_size);
  search.setSortedResults(true);

  // Apply filter to populate leaves_
  Indices filtered_indices;
  search.filter(filtered_indices);

  ASSERT_GT(filtered_indices.size(), 0);

  // Extract filtered cloud for KdTree comparison
  pcl::copyPointCloud(*original_cloud, filtered_indices, *filtered_cloud);

  // Create KdTree on filtered cloud
  search::KdTree<PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);
  kdtree.setSortedResults(true);

  // Compute bounding box for query point generation
  PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

  // Test multiple queries
  std::mt19937 rng(54321);
  std::uniform_real_distribution<float> x_dist(min_pt.x, max_pt.x);
  std::uniform_real_distribution<float> y_dist(min_pt.y, max_pt.y);
  std::uniform_real_distribution<float> z_dist(min_pt.z, max_pt.z);

  for (unsigned int test = 0; test < num_tests; ++test) {
    // Generate random query point
    PointXYZ query_point(x_dist(rng), y_dist(rng), z_dist(rng));

    // Search using UniformSamplingSearch
    Indices search_indices;
    std::vector<float> search_distances;
    int num_found = search.radiusSearch(
        query_point, search_radius, search_indices, search_distances);

    // KdTree search on filtered cloud
    Indices kd_indices;
    std::vector<float> kd_distances;
    int kd_num_found = kdtree.radiusSearch(query_point, search_radius, kd_indices, kd_distances);

    // Verify same number of results
    ASSERT_EQ(num_found, kd_num_found);
    ASSERT_EQ(search_indices.size(), kd_indices.size());

    // Verify all points are within radius
    for (std::size_t i = 0; i < search_indices.size(); ++i) {
      const PointXYZ& pt = (*filtered_cloud)[search_indices[i]];
      float dist_sq = squaredEuclideanDistance(query_point, pt);
      EXPECT_LE(dist_sq, search_radius * search_radius + 1e-4f);
    }

    // Verify distances match
    for (std::size_t i = 0; i < search_distances.size(); ++i) {
      EXPECT_NEAR(search_distances[i], kd_distances[i], 1e-4f);
    }

    // Verify distances are sorted
    for (std::size_t i = 1; i < search_distances.size(); ++i) {
      EXPECT_LE(search_distances[i - 1], search_distances[i]);
    }

    // Verify same points are found (indices should match since both use filtered cloud)
    for (std::size_t i = 0; i < search_indices.size(); ++i) {
      EXPECT_EQ(search_indices[i], kd_indices[i]);
    }
  }
}

TEST(PCL, UniformSamplingSearch_EmptyCloud)
{
  PointCloud<PointXYZ>::Ptr empty_cloud(new PointCloud<PointXYZ>);
  empty_cloud->width = 0;
  empty_cloud->height = 1;

  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud(empty_cloud);
  search.setRadiusSearch(0.1);

  // Apply filter on empty cloud
  Indices filtered_indices;
  search.filter(filtered_indices);

  // Search should return 0 results
  PointXYZ query_point(0.0f, 0.0f, 0.0f);
  Indices indices;
  std::vector<float> distances;

  int num_found_k = search.nearestKSearch(query_point, 10, indices, distances);
  EXPECT_EQ(num_found_k, 0);

  int num_found_r = search.radiusSearch(query_point, 1.0, indices, distances);
  EXPECT_EQ(num_found_r, 0);
}

TEST(PCL, UniformSamplingSearch_FilterNotApplied)
{
  PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
  cloud->push_back(PointXYZ(0.0f, 0.0f, 0.0f));
  cloud->push_back(PointXYZ(1.0f, 0.0f, 0.0f));
  cloud->push_back(PointXYZ(0.0f, 1.0f, 0.0f));

  UniformSamplingSearch<PointXYZ> search;
  search.setInputCloud(cloud);
  search.setRadiusSearch(0.1);

  // Don't apply filter - leaves_ should be empty
  PointXYZ query_point(0.0f, 0.0f, 0.0f);
  Indices indices;
  std::vector<float> distances;

  int num_found_k = search.nearestKSearch(query_point, 10, indices, distances);
  EXPECT_EQ(num_found_k, 0);

  int num_found_r = search.radiusSearch(query_point, 1.0, indices, distances);
  EXPECT_EQ(num_found_r, 0);
}

TEST(PCL, UniformSamplingSearch_FilterOutputMatchesUniformSampling)
{
  constexpr unsigned int num_points = 5000;
  constexpr float sphere_radius = 5.0f;
  constexpr float voxel_size = 0.2f;

  PointCloud<PointXYZ>::Ptr original_cloud = bun0_cloud;

  // Apply UniformSampling filter
  UniformSampling<PointXYZ> uniform_sampling;
  uniform_sampling.setInputCloud(original_cloud);
  uniform_sampling.setRadiusSearch(voxel_size);
  Indices uniform_sampling_indices;
  uniform_sampling.filter(uniform_sampling_indices);
  PointCloud<PointXYZ> uniform_sampling_output;
  uniform_sampling.filter(uniform_sampling_output);

  // Apply UniformSamplingSearch filter
  UniformSamplingSearch<PointXYZ> uniform_sampling_search;
  uniform_sampling_search.setInputCloud(original_cloud);
  uniform_sampling_search.setRadiusSearch(voxel_size);
  Indices uniform_sampling_search_indices;
  uniform_sampling_search.filter(uniform_sampling_search_indices);
  PointCloud<PointXYZ> uniform_sampling_search_output;
  uniform_sampling_search.filter(uniform_sampling_search_output);

  // Verify indices match
  ASSERT_EQ(uniform_sampling_indices.size(), uniform_sampling_search_indices.size());

  // Check that uniform_sampling points are identical to uniform_sampling_search points
  ASSERT_EQ(uniform_sampling_output.size(), uniform_sampling_search_output.size()) << "Output clouds have different sizes!";

  for (std::size_t i = 0; i < uniform_sampling_output.size(); ++i) {
    const auto& p1 = uniform_sampling_output[i];
    const auto& p2 = uniform_sampling_search_output[i];
    ASSERT_EQ(p1.getVector3fMap(), p2.getVector3fMap()) << "Point mismatch at index " << i;
  }

}

int
main(int argc, char** argv)
{
  bun0_cloud.reset(new PointCloud<PointXYZ>);
  if (0 != pcl::io::loadPCDFile<PointXYZ>(argv[1], *bun0_cloud)) {
    std::cerr << "Failed to load test file " << argv[1] << std::endl;
    return (-1);
  }

  testing::InitGoogleTest(&argc, argv);
  return (RUN_ALL_TESTS());
}
