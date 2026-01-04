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

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/filters/angular_density_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>

#include <cmath>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

float default_approx_voxel_size = 0.5f;
float default_azimuth_inc = 0.0f; // 0 means auto-derive from cloud width
float default_elevation_inc = 0.0f;

void
printHelp(int, char** argv)
{
  print_error("Syntax is: %s input.pcd output.pcd <options>\n", argv[0]);
  print_info("  where options are:\n");
  print_info("    -voxel_size X     = approximate voxel size (default: ");
  print_value("%f", default_approx_voxel_size);
  print_info(")\n");
  print_info("    -azimuth X     = azimuth angle increment in radians (default: auto "
             "from cloud width)\n");
  print_info("    -elevation X   = elevation angle increment in radians (default: auto "
             "from cloud height)\n");
  print_info(
      "    -organized     = keep organized structure (set removed points to NaN)\n");
  print_info("    -negative      = return removed points instead of kept points\n");
}

int
main(int argc, char** argv)
{
  print_info("Subsample organized LIDAR point cloud using angular density filter.\n");
  print_info("For more information, use: %s -h\n", argv[0]);

  if (argc < 3) {
    printHelp(argc, argv);
    return (-1);
  }

  std::vector<int> p_file_indices = parse_file_extension_argument(argc, argv, ".pcd");
  if (p_file_indices.size() != 2) {
    print_error("Need one input PCD file and one output PCD file.\n");
    return (-1);
  }

  float approx_voxel_size = default_approx_voxel_size;
  float azimuth_inc = default_azimuth_inc;
  float elevation_inc = default_elevation_inc;
  bool keep_organized = find_switch(argc, argv, "-organized");
  bool negative = find_switch(argc, argv, "-negative");

  parse_argument(argc, argv, "-voxel_size", approx_voxel_size);
  parse_argument(argc, argv, "-azimuth", azimuth_inc);
  parse_argument(argc, argv, "-elevation", elevation_inc);

  // Load input cloud
  TicToc tt;
  print_highlight("Loading ");
  print_value("%s ", argv[p_file_indices[0]]);
  tt.tic();

  pcl::PCLPointCloud2 cloud2;
  if (loadPCDFile(argv[p_file_indices[0]], cloud2) < 0)
    return (-1);

  print_info("[done, ");
  print_value("%g", tt.toc());
  print_info(" ms : ");
  print_value("%d", cloud2.width * cloud2.height);
  print_info(" points]\n");

  // Check if organized
  if (cloud2.height == 1) {
    print_error("Input cloud is not organized (height=1). This filter requires "
                "organized clouds.\n");
    return (-1);
  }

  // Auto-derive angle increments if not specified
  if (azimuth_inc == 0.0f) {
    azimuth_inc = static_cast<float>(2.0 * M_PI / cloud2.width);
    print_info("Auto-derived azimuth increment: ");
    print_value("%f rad\n", azimuth_inc);
  }
  if (elevation_inc == 0.0f) {
    elevation_inc = static_cast<float>(0.5 / cloud2.height); // Assume ~28 deg FOV
    print_info("Auto-derived elevation increment: ");
    print_value("%f rad\n", elevation_inc);
  }

  // Convert to PointXYZ for filtering
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(cloud2, *cloud);

  // Apply filter
  print_highlight("Filtering with approx_voxel_size=");
  print_value("%f", approx_voxel_size);
  print_info(" ...\n");
  tt.tic();

  pcl::AngularDensitySampling<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud);
  filter.setAngleIncrements(azimuth_inc, elevation_inc);
  filter.setApproxVoxelSize(approx_voxel_size);
  filter.setKeepOrganized(keep_organized);
  filter.setNegative(negative);

  pcl::PointCloud<pcl::PointXYZ> output;
  filter.filter(output);

  print_info("[done, ");
  print_value("%g", tt.toc());
  print_info(" ms : ");
  print_value("%d", output.width * output.height);
  print_info(" points]\n");

  // Save output
  print_highlight("Saving ");
  print_value("%s ", argv[p_file_indices[1]]);
  tt.tic();

  pcl::PCLPointCloud2 output2;
  pcl::toPCLPointCloud2(output, output2);
  PCDWriter w;
  w.writeBinaryCompressed(argv[p_file_indices[1]], output2);

  print_info("[done, ");
  print_value("%g", tt.toc());
  print_info(" ms]\n");

  return (0);
}
