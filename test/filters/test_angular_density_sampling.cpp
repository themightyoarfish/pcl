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
#include <pcl/filters/angular_density_sampling.h>
#include <pcl/point_types.h>
#include <cmath>

// Create a synthetic organized LIDAR point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr createTestCloud(
    std::uint32_t width, std::uint32_t height,
    float azimuth_inc, float elevation_inc, float range)
{
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->width = width;
  cloud->height = height;
  cloud->is_dense = true;
  cloud->resize(width * height);

  for (std::uint32_t h = 0; h < height; ++h) {
    float elevation = h * elevation_inc;
    for (std::uint32_t w = 0; w < width; ++w) {
      float azimuth = w * azimuth_inc;
      std::uint32_t idx = h * width + w;

      // Convert spherical to cartesian
      (*cloud)[idx].x = range * std::cos(elevation) * std::cos(azimuth);
      (*cloud)[idx].y = range * std::cos(elevation) * std::sin(azimuth);
      (*cloud)[idx].z = range * std::sin(elevation);
    }
  }
  return cloud;
}

TEST(AngularDensitySampling, BasicFiltering)
{
  constexpr std::uint32_t WIDTH = 1364;
  constexpr std::uint32_t HEIGHT = 128;
  constexpr float AZIMUTH_INC = 2.0f * static_cast<float>(M_PI) / WIDTH;
  constexpr float ELEVATION_INC = 0.5f / HEIGHT;  // ~28 deg total
  constexpr float RANGE = 10.0f;
  constexpr float MIN_SPACING = 0.5f;

  auto cloud = createTestCloud(WIDTH, HEIGHT, AZIMUTH_INC, ELEVATION_INC, RANGE);

  pcl::AngularDensitySampling<pcl::PointXYZ> filter(true);
  filter.setInputCloud(cloud);
  filter.setAngleIncrements(AZIMUTH_INC, ELEVATION_INC);
  filter.setApproxVoxelSize(MIN_SPACING);

  pcl::Indices indices;
  filter.filter(indices);

  // Verify no two kept points are closer than MIN_SPACING
  const float min_spacing_sq = MIN_SPACING * MIN_SPACING;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    for (std::size_t j = i + 1; j < indices.size(); ++j) {
      const auto& p1 = (*cloud)[indices[i]];
      const auto& p2 = (*cloud)[indices[j]];
      float dx = p1.x - p2.x;
      float dy = p1.y - p2.y;
      float dz = p1.z - p2.z;
      float dist_sq = dx*dx + dy*dy + dz*dz;
      EXPECT_GE(dist_sq, min_spacing_sq * 0.99f);  // Allow small tolerance
    }
  }

  // Should have fewer points than original
  EXPECT_LT(indices.size(), cloud->size());
  EXPECT_GT(indices.size(), 0u);
}

TEST(AngularDensitySampling, KeepOrganized)
{
  constexpr std::uint32_t WIDTH = 1364;
  constexpr std::uint32_t HEIGHT = 128;
  constexpr float AZIMUTH_INC = 2.0f * static_cast<float>(M_PI) / WIDTH;
  constexpr float ELEVATION_INC = 0.5f / HEIGHT;
  constexpr float RANGE = 10.0f;

  auto cloud = createTestCloud(WIDTH, HEIGHT, AZIMUTH_INC, ELEVATION_INC, RANGE);

  pcl::AngularDensitySampling<pcl::PointXYZ> filter(true);
  filter.setInputCloud(cloud);
  filter.setAngleIncrements(AZIMUTH_INC, ELEVATION_INC);
  filter.setApproxVoxelSize(0.5f);
  filter.setKeepOrganized(true);

  pcl::PointCloud<pcl::PointXYZ> output;
  filter.filter(output);

  // Output should have same dimensions
  EXPECT_EQ(output.width, WIDTH);
  EXPECT_EQ(output.height, HEIGHT);
  EXPECT_EQ(output.size(), cloud->size());

  // Removed points should be NaN
  auto removed = filter.getRemovedIndices();
  for (const auto& idx : *removed) {
    EXPECT_TRUE(std::isnan(output[idx].x));
  }
}

TEST(AngularDensitySampling, ZeroPointsSkipped)
{
  constexpr std::uint32_t WIDTH = 100;
  constexpr std::uint32_t HEIGHT = 10;

  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->width = WIDTH;
  cloud->height = HEIGHT;
  cloud->resize(WIDTH * HEIGHT);

  // Fill with zeros (invalid points)
  for (auto& pt : *cloud) {
    pt.x = pt.y = pt.z = 0.0f;
  }

  // Add a few valid points
  (*cloud)[50].x = 10.0f;
  (*cloud)[50].y = 0.0f;
  (*cloud)[50].z = 0.0f;

  (*cloud)[500].x = 20.0f;
  (*cloud)[500].y = 0.0f;
  (*cloud)[500].z = 0.0f;

  pcl::AngularDensitySampling<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud);
  filter.setAngleIncrements(0.01f, 0.01f);
  filter.setApproxVoxelSize(5.0f);  // Large spacing

  pcl::Indices indices;
  filter.filter(indices);

  // Should have exactly 2 valid points kept
  EXPECT_EQ(indices.size(), 2u);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

