#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/point_tests.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

void
print_help()
{
  std::cout << "Usage: benchmark_uniform_sampling_search <pcd_filename> <voxel_radius>\n";
  std::cout << "  pcd_filename: Path to a PCD file\n";
  std::cout << "  voxel_radius: Voxel size for uniform sampling (e.g., 0.01)\n";
}

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;
using QueryPoints = std::vector<pcl::PointXYZ>;

constexpr std::size_t kNumQueryPoints = 100;
constexpr unsigned kQueryRngSeed = 12345;

static bool
isValidQueryPoint(const pcl::PointXYZ& pt)
{
  if (!pcl::isFinite(pt)) {
    return false;
  }
  return pt.x != 0.f || pt.y != 0.f || pt.z != 0.f;
}

static PointCloudPtr
filterCloud(const PointCloudPtr& cloud_in, double voxel_radius)
{
  pcl::UniformSamplingSearch<pcl::PointXYZ> uniform_sampling;
  uniform_sampling.setInputCloud(cloud_in);
  uniform_sampling.setRadiusSearch(voxel_radius);
  pcl::Indices filtered_indices;
  uniform_sampling.filter(filtered_indices);

  PointCloudPtr filtered_cloud(new PointCloud);
  pcl::copyPointCloud(*cloud_in, filtered_indices, *filtered_cloud);
  return filtered_cloud;
}

static QueryPoints
generateQueryPoints(const PointCloudPtr& filtered_cloud,
                    std::size_t count,
                    unsigned seed)
{
  pcl::PointXYZ min_pt;
  pcl::PointXYZ max_pt;
  pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

  QueryPoints queries;
  queries.reserve(count);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> x_dist(min_pt.x, max_pt.x);
  std::uniform_real_distribution<float> y_dist(min_pt.y, max_pt.y);
  std::uniform_real_distribution<float> z_dist(min_pt.z, max_pt.z);

  const std::size_t max_attempts = count * 1000;
  for (std::size_t attempts = 0; queries.size() < count && attempts < max_attempts;
       ++attempts) {
    const pcl::PointXYZ candidate(x_dist(rng), y_dist(rng), z_dist(rng));
    if (isValidQueryPoint(candidate)) {
      queries.push_back(candidate);
    }
  }

  return queries;
}

static const pcl::PointXYZ&
nextQueryPoint(const QueryPoints& query_points, std::size_t& cursor)
{
  return query_points[cursor++ % query_points.size()];
}

static void
warmupKdTree(pcl::search::KdTree<pcl::PointXYZ>& kdtree,
             const QueryPoints& query_points)
{
  pcl::Indices indices;
  std::vector<float> sqr_distances;
  kdtree.nearestKSearch(query_points.front(), 1, indices, sqr_distances);
  benchmark::DoNotOptimize(indices.size());
}

static void
BM_UniformSamplingSearch_ConstructAndFilter(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius)
{
  pcl::Indices filtered_indices;
  for (auto _ : state) {
    pcl::UniformSamplingSearch<pcl::PointXYZ> search;
    search.setInputCloud(cloud_in);
    search.setRadiusSearch(voxel_radius);
    search.filter(filtered_indices);
    benchmark::DoNotOptimize(filtered_indices.size());
  }
}

static void
BM_UniformSamplingSearch_KSearchOnly(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius,
    const double search_radius,
    const int k,
    const QueryPoints query_points)
{
  if (query_points.empty()) {
    state.SkipWithError("query_points is empty");
    return;
  }

  pcl::UniformSamplingSearch<pcl::PointXYZ> search;
  search.setInputCloud(cloud_in);
  search.setRadiusSearch(voxel_radius);
  search.setMaxSearchRadius(search_radius);
  pcl::Indices filtered_indices;
  search.filter(filtered_indices);

  if (filtered_indices.empty()) {
    state.SkipWithError("filtered_indices is empty");
    return;
  }

  pcl::Indices k_indices;
  std::vector<float> k_sqr_distances;
  std::size_t query_cursor = 0;

  for (auto _ : state) {
    search.nearestKSearch(
        nextQueryPoint(query_points, query_cursor), k, k_indices, k_sqr_distances);
    benchmark::DoNotOptimize(k_indices.size());
    benchmark::DoNotOptimize(k_sqr_distances.size());
  }
}

static void
BM_KdTree_ConstructAndFilter(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius,
    const QueryPoints query_points)
{
  if (query_points.empty()) {
    state.SkipWithError("query_points is empty");
    return;
  }

  const PointCloudPtr filtered_cloud = filterCloud(cloud_in, voxel_radius);
  if (filtered_cloud->empty()) {
    state.SkipWithError("filtered cloud is empty");
    return;
  }

  std::size_t query_cursor = 0;
  for (auto _ : state) {
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(filtered_cloud);
    pcl::Indices indices;
    std::vector<float> sqr_distances;
    kdtree.nearestKSearch(
        nextQueryPoint(query_points, query_cursor), 1, indices, sqr_distances);
    benchmark::DoNotOptimize(indices.size());
    benchmark::DoNotOptimize(sqr_distances.size());
  }
}

static void
BM_KdTree_KSearchOnly(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius,
    const int k,
    const QueryPoints query_points)
{
  if (query_points.empty()) {
    state.SkipWithError("query_points is empty");
    return;
  }

  const PointCloudPtr filtered_cloud = filterCloud(cloud_in, voxel_radius);
  if (filtered_cloud->empty()) {
    state.SkipWithError("filtered cloud is empty");
    return;
  }

  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);
  warmupKdTree(kdtree, query_points);

  pcl::Indices k_indices;
  std::vector<float> k_sqr_distances;
  std::size_t query_cursor = 0;

  for (auto _ : state) {
    kdtree.nearestKSearch(
        nextQueryPoint(query_points, query_cursor), k, k_indices, k_sqr_distances);
    benchmark::DoNotOptimize(k_indices.size());
    benchmark::DoNotOptimize(k_sqr_distances.size());
  }
}

static void
BM_UniformSamplingSearch_RadiusSearchOnly(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius,
    const double search_radius,
    const QueryPoints query_points)
{
  if (query_points.empty()) {
    state.SkipWithError("query_points is empty");
    return;
  }

  pcl::UniformSamplingSearch<pcl::PointXYZ> search;
  search.setInputCloud(cloud_in);
  search.setRadiusSearch(voxel_radius);
  search.setMaxSearchRadius(search_radius);
  pcl::Indices filtered_indices;
  search.filter(filtered_indices);

  if (filtered_indices.empty()) {
    state.SkipWithError("filtered_indices is empty");
    return;
  }

  pcl::Indices r_indices;
  std::vector<float> r_sqr_distances;
  std::size_t query_cursor = 0;

  for (auto _ : state) {
    search.radiusSearch(nextQueryPoint(query_points, query_cursor),
                        search_radius,
                        r_indices,
                        r_sqr_distances);
    benchmark::DoNotOptimize(r_indices.size());
    benchmark::DoNotOptimize(r_sqr_distances.size());
  }
}

static void
BM_KdTree_RadiusSearchOnly(
    benchmark::State& state,
    const PointCloudPtr cloud_in,
    const double voxel_radius,
    const double search_radius,
    const QueryPoints query_points)
{
  if (query_points.empty()) {
    state.SkipWithError("query_points is empty");
    return;
  }

  const PointCloudPtr filtered_cloud = filterCloud(cloud_in, voxel_radius);
  if (filtered_cloud->empty()) {
    state.SkipWithError("filtered cloud is empty");
    return;
  }

  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(filtered_cloud);
  warmupKdTree(kdtree, query_points);

  pcl::Indices r_indices;
  std::vector<float> r_sqr_distances;
  std::size_t query_cursor = 0;

  for (auto _ : state) {
    kdtree.radiusSearch(nextQueryPoint(query_points, query_cursor),
                        search_radius,
                        r_indices,
                        r_sqr_distances);
    benchmark::DoNotOptimize(r_indices.size());
    benchmark::DoNotOptimize(r_sqr_distances.size());
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

  PointCloudPtr cloud_in(new PointCloud);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud_in) < 0) {
    std::cerr << "Error: Failed to load PCD file: " << argv[1] << std::endl;
    return -1;
  }

  double voxel_radius = 0.01;
  try {
    voxel_radius = std::abs(std::stod(argv[2]));
  } catch (const std::invalid_argument&) {
    std::cerr << "Error: Invalid voxel radius. Using default: 0.01\n";
  }

  std::cout << "Loaded cloud with " << cloud_in->size() << " points\n";
  std::cout << "Voxel radius: " << voxel_radius << "\n";

  const PointCloudPtr filtered_cloud = filterCloud(cloud_in, voxel_radius);
  if (filtered_cloud->empty()) {
    std::cerr << "Error: filtered cloud is empty\n";
    return -1;
  }

  const QueryPoints query_points =
      generateQueryPoints(filtered_cloud, kNumQueryPoints, kQueryRngSeed);
  if (query_points.size() < kNumQueryPoints) {
    std::cerr << "Error: could only generate " << query_points.size() << " of "
              << kNumQueryPoints << " valid query points\n";
    return -1;
  }

  std::cout << "Using " << query_points.size()
            << " random query points (seed=" << kQueryRngSeed << ")\n";

  constexpr int k = 20;
  const double search_radius = voxel_radius * 3.0;
  constexpr double kMinTime = 2.0;
  constexpr double kMinWarmUpTime =  0.5;

  auto register_ms_benchmark = [&](const char* name, auto* func, auto... args) {
    benchmark::RegisterBenchmark(name, func, args...)
        ->Unit(benchmark::kMillisecond)
        ->MinWarmUpTime(kMinWarmUpTime)
        ->MinTime(kMinTime);
  };

  register_ms_benchmark("USS_ConstructAndFilter",
                        &BM_UniformSamplingSearch_ConstructAndFilter,
                        cloud_in,
                        voxel_radius);

  register_ms_benchmark("USS_KSearchOnly",
                        &BM_UniformSamplingSearch_KSearchOnly,
                        cloud_in,
                        voxel_radius,
                        search_radius,
                        k,
                        query_points);

  register_ms_benchmark("USS_RadiusSearchOnly",
                        &BM_UniformSamplingSearch_RadiusSearchOnly,
                        cloud_in,
                        voxel_radius,
                        search_radius,
                        query_points);

  register_ms_benchmark("KdTree_ConstructAndFilter",
                        &BM_KdTree_ConstructAndFilter,
                        cloud_in,
                        voxel_radius,
                        query_points);

  register_ms_benchmark("KdTree_KSearchOnly",
                        &BM_KdTree_KSearchOnly,
                        cloud_in,
                        voxel_radius,
                        k,
                        query_points);

  register_ms_benchmark("KdTree_RadiusSearchOnly",
                        &BM_KdTree_RadiusSearchOnly,
                        cloud_in,
                        voxel_radius,
                        search_radius,
                        query_points);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
