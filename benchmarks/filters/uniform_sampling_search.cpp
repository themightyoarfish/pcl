#include <pcl/common/io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <random>

#include <benchmark/benchmark.h>

void
print_help()
{
  std::cout << "Usage: benchmark_uniform_sampling_search <pcd_filename> <voxel_radius>\n";
  std::cout << "  pcd_filename: Path to a PCD file\n";
  std::cout << "  voxel_radius: Voxel size for uniform sampling (e.g., 0.01)\n";
}

static void
BM_UniformSamplingSearch_ConstructAndFilter(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius)
{
    pcl::Indices filtered_indices;
  for (auto _ : state) {
    // Construct UniformSamplingSearch
    pcl::UniformSamplingSearch<pcl::PointXYZ> search;
    search.setInputCloud(cloud_in);
    search.setRadiusSearch(voxel_radius);
    search.filter(filtered_indices);
  }
}

static void
BM_UniformSamplingSearch_KSearchOnly(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius,
    const int k)
{
  int radiusSearchIdx = 0;

    // Construct UniformSamplingSearch
    pcl::UniformSamplingSearch<pcl::PointXYZ> search;
    search.setInputCloud(cloud_in);
    search.setRadiusSearch(voxel_radius);
    // Apply filter (required before search)
    pcl::Indices filtered_indices;
    search.filter(filtered_indices);
    state.ResumeTiming();
    // Search for k nearest neighbors
    pcl::Indices k_indices;
    std::vector<float> k_sqr_distances;

  for (auto _ : state) {
      search.nearestKSearch((*cloud_in)[radiusSearchIdx++ % filtered_indices.size()], k, k_indices, k_sqr_distances);
  }
}

static void
BM_KdTree_ConstructAndFilter(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius)
{
  state.PauseTiming();
    // First filter using UniformSamplingSearch
    pcl::UniformSamplingSearch<pcl::PointXYZ> uniform_sampling;
    uniform_sampling.setInputCloud(cloud_in);
    uniform_sampling.setRadiusSearch(voxel_radius);
    pcl::Indices filtered_indices;
    uniform_sampling.filter(filtered_indices);

    // Extract filtered cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_in, filtered_indices, *filtered_cloud);

  state.ResumeTiming();

  for (auto _ : state) {
    // Build KdTree on filtered cloud
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(filtered_cloud);
  }
}

static void
BM_KdTree_KSearchOnly(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius,
    const int k)
{
  state.PauseTiming();

  // First filter using UniformSamplingSearch
  pcl::UniformSamplingSearch<pcl::PointXYZ> uniform_sampling;
  uniform_sampling.setInputCloud(cloud_in);
  uniform_sampling.setRadiusSearch(voxel_radius);
  pcl::Indices filtered_indices;
  uniform_sampling.filter(filtered_indices);

  // Extract filtered cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud_in, filtered_indices, *filtered_cloud);

  // Build KdTree on filtered cloud
  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);

  pcl::Indices k_indices;
  std::vector<float> k_sqr_distances;

  int radiusSearchIdx = 0;

  state.ResumeTiming();

  for (auto _ : state) {
    // Run nearestKSearch on a subset of points
    kdtree.nearestKSearch((*cloud_in)[radiusSearchIdx++ % filtered_cloud->size()], k, k_indices, k_sqr_distances);
  }
}

static void
BM_UniformSamplingSearch_RadiusSearchOnly(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius,
    const double search_radius)
{
  int radiusSearchIdx = 0;

  state.PauseTiming();
  // Construct UniformSamplingSearch
  pcl::UniformSamplingSearch<pcl::PointXYZ> search;
  search.setInputCloud(cloud_in);
  search.setRadiusSearch(voxel_radius);
  // Apply filter (required before search)
  pcl::Indices filtered_indices;
  search.filter(filtered_indices);
  state.ResumeTiming();
  // Search for neighbors within radius
  pcl::Indices r_indices;
  std::vector<float> r_sqr_distances;

  for (auto _ : state) {
    search.radiusSearch((*cloud_in)[radiusSearchIdx++ % filtered_indices.size()], search_radius, r_indices, r_sqr_distances);
  }
}

static void
BM_KdTree_RadiusSearchOnly(
    benchmark::State& state,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    const double voxel_radius,
    const double search_radius)
{
  state.PauseTiming();

  // First filter using UniformSamplingSearch
  pcl::UniformSamplingSearch<pcl::PointXYZ> uniform_sampling;
  uniform_sampling.setInputCloud(cloud_in);
  uniform_sampling.setRadiusSearch(voxel_radius);
  pcl::Indices filtered_indices;
  uniform_sampling.filter(filtered_indices);

  // Extract filtered cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud_in, filtered_indices, *filtered_cloud);

  // Build KdTree on filtered cloud
  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);

  pcl::Indices r_indices;
  std::vector<float> r_sqr_distances;

  int radiusSearchIdx = 0;

  state.ResumeTiming();

  for (auto _ : state) {
    // Run radiusSearch on a subset of points
    kdtree.radiusSearch((*cloud_in)[radiusSearchIdx++ % filtered_cloud->size()], search_radius, r_indices, r_sqr_distances);
  }
}

int
main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Error: Missing arguments.\n";
    print_help();
    return -1;
  }

  // Load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud_in) < 0) {
    std::cerr << "Error: Failed to load PCD file: " << argv[1] << std::endl;
    return -1;
  }

  // Parse voxel radius
  double voxel_radius = 0.01;
  try {
    voxel_radius = std::abs(std::stod(argv[2]));
  } catch (const std::invalid_argument&) {
    std::cerr << "Error: Invalid voxel radius. Using default: 0.01\n";
  }

  std::cout << "Loaded cloud with " << cloud_in->size() << " points\n";
  std::cout << "Voxel radius: " << voxel_radius << "\n";

  constexpr int k = 5;
  const double search_radius = voxel_radius * 3.0; // Use 3x voxel radius as search radius

  // Benchmark: construct + filter (full pipeline)
  benchmark::RegisterBenchmark("USS_ConstructAndFilter",
                               &BM_UniformSamplingSearch_ConstructAndFilter,
                               cloud_in,
                               voxel_radius)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  // Benchmark: k-nearest search only (pre-constructed)
  benchmark::RegisterBenchmark("USS_KSearchOnly",
                               &BM_UniformSamplingSearch_KSearchOnly,
                               cloud_in,
                               voxel_radius,
                               k)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  // Benchmark: radius search only (pre-constructed)
  benchmark::RegisterBenchmark("USS_RadiusSearchOnly",
                               &BM_UniformSamplingSearch_RadiusSearchOnly,
                               cloud_in,
                               voxel_radius,
                               search_radius)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  // Benchmark: KdTree construct + filter (full pipeline)
  benchmark::RegisterBenchmark("KdTree_ConstructAndFilter",
                               &BM_KdTree_ConstructAndFilter,
                               cloud_in,
                               voxel_radius)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  // Benchmark: KdTree k-nearest search only (pre-constructed)
  benchmark::RegisterBenchmark("KdTree_KSearchOnly",
                               &BM_KdTree_KSearchOnly,
                               cloud_in,
                               voxel_radius,
                               k)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  // Benchmark: KdTree radius search only (pre-constructed)
  benchmark::RegisterBenchmark("KdTree_RadiusSearchOnly",
                               &BM_KdTree_RadiusSearchOnly,
                               cloud_in,
                               voxel_radius,
                               search_radius)
      ->Unit(benchmark::kMillisecond)
      ->MinWarmUpTime(0.5)->MinTime(2.0);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
