// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/random/zipf_distribution.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/config.h"
#include "absl/log/log.h"
#include "absl/random/internal/chi_square.h"
#include "absl/random/internal/pcg_engine.h"
#include "absl/random/internal/sequence_urbg.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace random_internal {

struct ZipfDistributionTestPeer {
  template <typename IntType>
  static double max_kd(const zipf_distribution<IntType>& d) {
    return d.param().max_kd_;
  }
  template <typename IntType>
  static double s(const zipf_distribution<IntType>& d) {
    return d.param().s_;
  }
  template <typename IntType>
  static double d0x5(const zipf_distribution<IntType>& d) {
    return d.param().d0x5_;
  }
  template <typename IntType>
  static double total_hat(const zipf_distribution<IntType>& d) {
    return d.param().total_hat_;
  }
  template <typename IntType>
  static double tail_integral(const zipf_distribution<IntType>& d, double x) {
    return d.param().tail_integral(x);
  }
  template <typename IntType>
  static double tail_integral_inv(const zipf_distribution<IntType>& d,
                                  double u) {
    return d.param().tail_integral_inv(u);
  }
};

}  // namespace random_internal
ABSL_NAMESPACE_END
}  // namespace absl

namespace {

using ::absl::random_internal::kChiSquared;
using ::absl::random_internal::ZipfDistributionTestPeer;
using ::testing::AllOf;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::Le;
using ::testing::Lt;

template <typename IntType>
class ZipfDistributionTypedTest : public ::testing::Test {};

using IntTypes = ::testing::Types<int, int8_t, int16_t, int32_t, int64_t,
                                  uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(ZipfDistributionTypedTest, IntTypes);

TYPED_TEST(ZipfDistributionTypedTest, SerializeTest) {
  using param_type = typename absl::zipf_distribution<TypeParam>::param_type;

  constexpr int kCount = 1000;
  absl::InsecureBitGen gen;
  for (const auto& param : {
           param_type(),
           param_type(0),
           param_type(32),
           param_type(100, 3, 2),
           param_type(std::numeric_limits<TypeParam>::max(), 4, 3),
           param_type(std::numeric_limits<TypeParam>::max() / 2),
       }) {
    // Validate parameters.
    const auto k = param.k();
    const auto q = param.q();
    const auto v = param.v();

    absl::zipf_distribution<TypeParam> before(k, q, v);
    EXPECT_EQ(before.k(), param.k());
    EXPECT_EQ(before.q(), param.q());
    EXPECT_EQ(before.v(), param.v());

    {
      absl::zipf_distribution<TypeParam> via_param(param);
      EXPECT_EQ(via_param, before);
    }

    // Validate stream serialization.
    std::stringstream ss;
    ss << before;
    absl::zipf_distribution<TypeParam> after(4, 5.5, 4.4);

    EXPECT_NE(before.k(), after.k());
    EXPECT_NE(before.q(), after.q());
    EXPECT_NE(before.v(), after.v());
    EXPECT_NE(before.param(), after.param());
    EXPECT_NE(before, after);

    ss >> after;

    EXPECT_FALSE(ss.fail());
    EXPECT_EQ(before.k(), after.k());
    EXPECT_EQ(before.q(), after.q());
    EXPECT_EQ(before.v(), after.v());
    EXPECT_EQ(before.param(), after.param());
    EXPECT_EQ(before, after);

    // Smoke test.
    auto sample_min = after.max();
    auto sample_max = after.min();
    for (int i = 0; i < kCount; i++) {
      auto sample = after(gen);
      EXPECT_GE(sample, after.min());
      EXPECT_LE(sample, after.max());
      if (sample > sample_max) sample_max = sample;
      if (sample < sample_min) sample_min = sample;
    }
    LOG(INFO) << "Range: " << +sample_min << ", " << +sample_max;
  }
}

class ZipfModel {
 public:
  ZipfModel(uint64_t k, double q, double v) : k_(k), q_(q), v_(v) {}

  double mean() const { return mean_; }

  // For the other moments of the Zipf distribution, see, for example,
  // http://mathworld.wolfram.com/ZipfDistribution.html

  // PMF(i) = (1 + i / v)^(-q) / Z
  // Returns the probability that any single invocation returns i.
  double PMF(uint64_t i) {
    return i >= hnq_.size() ? 0.0 : hnq_[static_cast<size_t>(i)] / sum_hnq_;
  }

  // CDF(i) = P(X < i) = sum_{j=0}^{i-1} PMF(j).
  double CDF(uint64_t i) {
    if (i == 0) {
      return 0.0;
    }
    if (i > cdf_.size()) {
      return 1.0;
    }
    if (i == cdf_.size() && !truncated_) {
      return 1.0;
    }
    return cdf_[static_cast<size_t>(i - 1)] / sum_hnq_;
  }

  // The InverseCDF returns the k values which bound p on the upper and lower
  // bound. Since there is no closed-form solution, this is implemented as a
  // bisection of the cdf.
  std::pair<uint64_t, uint64_t> InverseCDF(double p) {
    uint64_t min = 0;
    uint64_t max = static_cast<uint64_t>(cdf_.size()) + (truncated_ ? 1 : 0);
    while (max > min + 1) {
      uint64_t target = (max + min) >> 1;
      double x = CDF(target);
      if (x > p) {
        max = target;
      } else {
        min = target;
      }
    }
    return {min, max};
  }

  // Compute the unnormalized probability totals based on (1 + i / v)^(-q).
  void Init() {
    if (!hnq_.empty()) {
      return;
    }
    const size_t reserve_n =
        (k_ < uint64_t{1000}) ? static_cast<size_t>(k_ + 1) : size_t{1001};
    hnq_.reserve(reserve_n);
    cdf_.reserve(reserve_n);

    sum_hnq_ = 0.0;
    double weighted_i_sum = 0.0;
    truncated_ = false;
    uint64_t i = 0;
    for (;; ++i) {
      const double lp = std::log1p(static_cast<double>(i) / v_);
      const double hnq = std::exp(-q_ * lp);
      sum_hnq_ += hnq;
      weighted_i_sum += static_cast<double>(i) * hnq;
      hnq_.push_back(hnq);
      cdf_.push_back(sum_hnq_);
      if (i == k_) {
        break;
      }
      if (i >= 10000 && hnq <= 1e-12 * sum_hnq_) {
        truncated_ = true;
        break;
      }
    }
    if (truncated_) {
      constexpr double kMaxU64 = 18446744073709549568.0;
      const double kd = (static_cast<double>(k_) > kMaxU64)
                            ? kMaxU64
                            : static_cast<double>(k_);
      const double z_lo =
          -(q_ - 1.0) * std::log1p((static_cast<double>(i) + 0.5) / v_);
      const double z_hi = -(q_ - 1.0) * std::log1p((kd + 0.5) / v_);
      const double tail =
          v_ * std::exp(z_lo) * (-std::expm1(z_hi - z_lo)) / (q_ - 1.0);
      sum_hnq_ += tail;
    }
    assert(sum_hnq_ > 0);
    mean_ = weighted_i_sum / sum_hnq_;
  }

 private:
  const uint64_t k_;
  const double q_;
  const double v_;

  double mean_ = 0.0;
  std::vector<double> hnq_;
  std::vector<double> cdf_;
  double sum_hnq_ = 0.0;
  bool truncated_ = false;
};

using zipf_u64 = absl::zipf_distribution<uint64_t>;

class ZipfTest : public testing::TestWithParam<zipf_u64::param_type>,
                 public ZipfModel {
 public:
  ZipfTest() : ZipfModel(GetParam().k(), GetParam().q(), GetParam().v()) {}

  // We use a fixed bit generator for distribution accuracy tests.  This allows
  // these tests to be deterministic, while still testing the quality of the
  // implementation.
  absl::random_internal::pcg64_2018_engine rng_{0x2B7E151628AED2A6};
};

TEST_P(ZipfTest, ChiSquaredTest) {
  const auto& param = GetParam();
  Init();

  size_t trials = 10000;

  // Find the split-points for the buckets.
  std::vector<uint64_t> points;
  std::vector<double> expected;
  {
    double last_cdf = 0.0;
    double min_p = 1.0;
    for (double p = 0.01; p < 1.0; p += 0.01) {
      auto x = InverseCDF(p);
      if (points.empty() || points.back() < x.second) {
        const double cdf_at_split = CDF(x.second);
        if (cdf_at_split <= last_cdf) {
          continue;
        }
        points.push_back(x.second);
        double q = cdf_at_split - last_cdf;
        expected.push_back(q);
        last_cdf = cdf_at_split;
        if (q < min_p) {
          min_p = q;
        }
      }
    }
    if (last_cdf < 0.999 || (points.size() == 1 && last_cdf < 1.0)) {
      points.push_back(std::numeric_limits<uint64_t>::max());
      double q = 1.0 - last_cdf;
      expected.push_back(q);
      if (q < min_p) {
        min_p = q;
      }
    } else {
      points.back() = std::numeric_limits<uint64_t>::max();
      expected.back() += (1.0 - last_cdf);
    }
    // The Chi-Squared score is not completely scale-invariant; it works best
    // when the small values are in the small digits.
    trials = static_cast<size_t>(8.0 / min_p);
  }
  ASSERT_GT(points.size(), 1);

  // Generate n variates and fill the counts vector with the count of their
  // occurrences.
  std::vector<int64_t> buckets(points.size(), 0);
  double avg = 0;
  {
    zipf_u64 dis(param);
    for (size_t i = 0; i < trials; i++) {
      uint64_t x = dis(rng_);
      ASSERT_LE(x, dis.max());
      ASSERT_GE(x, dis.min());
      avg += static_cast<double>(x);
      auto it = std::upper_bound(points.begin(), points.end() - 1, x);
      buckets[static_cast<size_t>(std::distance(points.begin(), it))]++;
    }
    avg = avg / static_cast<double>(trials);
  }

  // Validate the output using the Chi-Squared test.
  for (auto& e : expected) {
    e *= static_cast<double>(trials);
  }

  // The null-hypothesis is that the distribution is a Zipf distribution with
  // the provided k, q, and v (not estimated from the data).
  const int dof = static_cast<int>(expected.size()) - 1;
  ASSERT_GT(dof, 0);

  // NOTE: This test runs about 17x per invocation, so a value of 0.9999 gives
  // a low suite-wide false positive rate.
  const double threshold = absl::random_internal::ChiSquareValue(dof, 0.9999);

  const double chi_square = absl::random_internal::ChiSquare(
      std::begin(buckets), std::end(buckets), std::begin(expected),
      std::end(expected));

  const double p_actual =
      absl::random_internal::ChiSquarePValue(chi_square, dof);

  // Log if the chi_squared value is above the threshold.
  if (chi_square > threshold) {
    LOG(INFO) << "values";
    for (size_t i = 0; i < expected.size(); i++) {
      LOG(INFO) << points[i] << ": " << buckets[i] << " vs. E=" << expected[i];
    }
    LOG(INFO) << "trials " << trials;
    LOG(INFO) << "mean " << avg << " vs. expected " << mean();
    LOG(INFO) << kChiSquared << "(data, " << dof << ") = " << chi_square << " ("
              << p_actual << ")";
    LOG(INFO) << kChiSquared << " @ 0.9999 = " << threshold;
    FAIL() << kChiSquared << " value of " << chi_square
           << " is above the threshold.";
  }
}

std::vector<zipf_u64::param_type> GenParams() {
  using param = zipf_u64::param_type;
  const auto k = param().k();
  const auto q = param().q();
  const auto v = param().v();
  const uint64_t k2 = 1 << 10;
  return std::vector<zipf_u64::param_type>{
      // Default
      param(k, q, v),
      // vary K
      param(4, q, v), param(1 << 4, q, v), param(k2, q, v),
      // vary V
      param(k2, q, 0.5), param(k2, q, 1.5), param(k2, q, 2.5), param(k2, q, 10),
      // vary Q
      param(k2, 1.5, v), param(k2, 3, v), param(k2, 5, v), param(k2, 10, v),
      // Vary V & Q, including q -> 1+ and large v
      param(k2, 1.5, 0.5), param(k2, 3, 1.5), param(k, 10, 10),
      param(100, 1.0 + 1e-15, 10.0), param(100, 2.5, 1e16)};
}

std::string ParamName(
    const ::testing::TestParamInfo<zipf_u64::param_type>& info) {
  const auto& p = info.param;
  std::string name = absl::StrCat("k_", p.k(), "__q_", absl::SixDigits(p.q()),
                                  "__v_", absl::SixDigits(p.v()));
  return absl::StrReplaceAll(name, {{"+", "_"}, {"-", "_"}, {".", "_"}});
}

INSTANTIATE_TEST_SUITE_P(All, ZipfTest, ::testing::ValuesIn(GenParams()),
                         ParamName);

// NOTE: absl::zipf_distribution is not guaranteed to be stable.
TEST(ZipfDistributionTest, StabilityTest) {
  // absl::zipf_distribution stability relies on
  // absl::uniform_real_distribution, std::log, std::exp, std::log1p
  absl::random_internal::sequence_urbg urbg(
      {0x0003eb76f6f7f755ull, 0xFFCEA50FDB2F953Bull, 0xC332DDEFBE6C5AA5ull,
       0x6558218568AB9702ull, 0x2AEF7DAD5B6E2F84ull, 0x1521B62829076170ull,
       0xECDD4775619F1510ull, 0x13CCA830EB61BD96ull, 0x0334FE1EAA0363CFull,
       0xB5735C904C70A239ull, 0xD59E9E0BCBAADE14ull, 0xEECC86BC60622CA7ull});

  std::vector<int> output(10);

  {
    absl::zipf_distribution<int32_t> dist;
    std::generate(std::begin(output), std::end(output),
                  [&] { return dist(urbg); });
    EXPECT_THAT(output, ElementsAre(10031, 0, 0, 3, 6, 0, 7, 47, 0, 0));
  }
  urbg.reset();
  {
    absl::zipf_distribution<int32_t> dist(std::numeric_limits<int32_t>::max(),
                                          3.3);
    std::generate(std::begin(output), std::end(output),
                  [&] { return dist(urbg); });
    EXPECT_THAT(output, ElementsAre(44, 0, 0, 0, 0, 1, 0, 1, 3, 0));
  }
}

TEST(ZipfDistributionTest, AlgorithmBounds) {
  absl::zipf_distribution<int32_t> dist;

  // Small values from absl::uniform_real_distribution map to larger Zipf
  // distribution values.
  const std::pair<uint64_t, int32_t> kInputs[] = {
      {0xffffffffffffffff, 0x0}, {0x7fffffffffffffff, 0x0},
      {0x3ffffffffffffffb, 0x1}, {0x1ffffffffffffffd, 0x4},
      {0xffffffffffffffe, 0x9},  {0x7ffffffffffffff, 0x12},
      {0x3ffffffffffffff, 0x25}, {0x1ffffffffffffff, 0x4c},
      {0xffffffffffffff, 0x99},  {0x7fffffffffffff, 0x132},
      {0x3fffffffffffff, 0x265}, {0x1fffffffffffff, 0x4cc},
      {0xfffffffffffff, 0x999},  {0x7ffffffffffff, 0x1332},
      {0x3ffffffffffff, 0x2665}, {0x1ffffffffffff, 0x4ccc},
      {0xffffffffffff, 0x9998},  {0x7fffffffffff, 0x1332f},
      {0x3fffffffffff, 0x2665a}, {0x1fffffffffff, 0x4cc9e},
      {0xfffffffffff, 0x998e0},  {0x7ffffffffff, 0x133051},
      {0x3ffffffffff, 0x265ae4}, {0x1ffffffffff, 0x4c9ed3},
      {0xffffffffff, 0x98e223},  {0x7fffffffff, 0x13058c4},
      {0x3fffffffff, 0x25b178e}, {0x1fffffffff, 0x4a062b2},
      {0xfffffffff, 0x8ee23b8},  {0x7ffffffff, 0x10b21642},
      {0x3ffffffff, 0x1d89d89d}, {0x1ffffffff, 0x2fffffff},
      {0xffffffff, 0x45d1745d},  {0x7fffffff, 0x5a5a5a5a},
      {0x3fffffff, 0x69ee5846},  {0x1fffffff, 0x73ecade3},
      {0xfffffff, 0x79a9d260},   {0x7ffffff, 0x7cc0532b},
      {0x3ffffff, 0x7e5ad146},   {0x1ffffff, 0x7f2c0bec},
      {0xffffff, 0x7f95adef},    {0x7fffff, 0x7fcac0da},
      {0x3fffff, 0x7fe55ae2},    {0x1fffff, 0x7ff2ac0e},
      {0xfffff, 0x7ff955ae},     {0x7ffff, 0x7ffcaac1},
      {0x3ffff, 0x7ffe555b},     {0x1ffff, 0x7fff2aac},
      {0xffff, 0x7fff9556},      {0x7fff, 0x7fffcaab},
      {0x3fff, 0x7fffe555},      {0x1fff, 0x7ffff2ab},
      {0xfff, 0x7ffff955},       {0x7ff, 0x7ffffcab},
      {0x3ff, 0x7ffffe55},       {0x1ff, 0x7fffff2b},
      {0xff, 0x7fffff95},        {0x7f, 0x7fffffcb},
      {0x3f, 0x7fffffe5},        {0x1f, 0x7ffffff3},
      {0xf, 0x7ffffff9},         {0x7, 0x7ffffffd},
      {0x3, 0x7ffffffe},         {0x1, 0x7fffffff},
  };

  for (const auto& instance : kInputs) {
    absl::random_internal::sequence_urbg urbg({instance.first});
    EXPECT_EQ(instance.second, dist(urbg));
  }
}

template <typename IntType>
IntType SampleFromSeedBits(absl::zipf_distribution<IntType> dist,
                           uint64_t first_bits) {
  absl::random_internal::sequence_urbg urbg(
      {first_bits, 0x8000000000000000ull});
  return dist(urbg);
}

template <typename IntType>
std::vector<IntType> DrawSamples(absl::zipf_distribution<IntType> dist,
                                 size_t count) {
  absl::random_internal::pcg64_2018_engine rng(0x2B7E151628AED2A6ull);
  std::vector<IntType> samples(count);
  std::generate(samples.begin(), samples.end(), [&] { return dist(rng); });
  return samples;
}

double SampleMean(absl::zipf_distribution<int> dist, size_t count) {
  absl::random_internal::pcg64_2018_engine rng(0x2B7E151628AED2A6ull);
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const int s = dist(rng);
    EXPECT_THAT(s, AllOf(Ge(0), Le(dist.max())));
    sum += static_cast<double>(s);
  }
  return sum / static_cast<double>(count);
}

TEST(ZipfDistributionTest, Int64AndUint64UpperBoundsAvoidFloatCastOverflow) {
  // Default int64_t has k = INT64_MAX (2^63 - 1), which rounds to 2^63 in
  // double; sampling at uniform_double == 0.0 must stay <= INT64_MAX without
  // [conv.fpint] UB.
  EXPECT_EQ(
      ZipfDistributionTestPeer::max_kd(absl::zipf_distribution<int64_t>{}),
      9223372036854774784.0);
  EXPECT_EQ(SampleFromSeedBits(absl::zipf_distribution<int64_t>{}, 0ull),
            int64_t{9223372036854774784});
  EXPECT_THAT(SampleFromSeedBits(absl::zipf_distribution<int64_t>{}, 1ull),
              AllOf(Ge(int64_t{0}), Le(std::numeric_limits<int64_t>::max())));

  // Non-max 64-bit k values where static_cast<double>(k) rounds strictly above
  // k (including on x87 80-bit registers): max_kd_ must mask below the top 53
  // significant bits so max_kd_ <= k unconditionally.
  constexpr int64_t kRoundsUpInDouble = (int64_t{1} << 62) - 1;
  EXPECT_EQ(ZipfDistributionTestPeer::max_kd(
                absl::zipf_distribution<int64_t>(kRoundsUpInDouble)),
            4611686018427387392.0);
  EXPECT_EQ(SampleFromSeedBits(
                absl::zipf_distribution<int64_t>(kRoundsUpInDouble), 0ull),
            int64_t{4611686018427387392});

  constexpr int64_t kNearMaxRoundsUp =
      std::numeric_limits<int64_t>::max() - 1025;
  EXPECT_EQ(ZipfDistributionTestPeer::max_kd(
                absl::zipf_distribution<int64_t>(kNearMaxRoundsUp)),
            9223372036854773760.0);
  EXPECT_EQ(SampleFromSeedBits(
                absl::zipf_distribution<int64_t>(kNearMaxRoundsUp), 0ull),
            int64_t{9223372036854773760});

  constexpr uint64_t kAbove53Bits = (uint64_t{1} << 53) + 1;
  EXPECT_EQ(ZipfDistributionTestPeer::max_kd(
                absl::zipf_distribution<uint64_t>(kAbove53Bits)),
            9007199254740992.0);
  EXPECT_EQ(
      SampleFromSeedBits(absl::zipf_distribution<uint64_t>(kAbove53Bits), 0ull),
      uint64_t{9007199254740992ull});

  constexpr uint64_t kExact53Bits = (uint64_t{1} << 53) - 1;
  EXPECT_EQ(ZipfDistributionTestPeer::max_kd(
                absl::zipf_distribution<uint64_t>(kExact53Bits)),
            9007199254740991.0);
  EXPECT_THAT(
      SampleFromSeedBits(absl::zipf_distribution<uint64_t>(kExact53Bits), 0ull),
      AllOf(Ge(uint64_t{0}), Le(kExact53Bits)));

  EXPECT_EQ(SampleFromSeedBits(absl::zipf_distribution<uint64_t>{}, 0ull),
            uint64_t{18446744073709549568ull});
  EXPECT_EQ(SampleFromSeedBits(absl::zipf_distribution<int64_t>(0), 0ull), 0);
}

TEST(ZipfDistributionTest, ExtremeParametersRemainInBoundsAndTerminate) {
  // Near q -> 1+: avoids catastrophic cancellation and negative outputs.
  EXPECT_THAT(
      DrawSamples(absl::zipf_distribution<int>(100, 1.0 + 1e-15, 10.0), 2000),
      Each(AllOf(Ge(0), Le(100))));
  EXPECT_THAT(DrawSamples(absl::zipf_distribution<int>(
                              100, std::nextafter(1.0, 2.0), 0.5),
                          2000),
              Each(AllOf(Ge(0), Le(100))));

  // Steep q with v < 1: previously overflowed pow_negative_q(v) to +inf and
  // returned -1; mathematically P(X = 0) == 1 in IEEE-754 double.
  EXPECT_THAT(DrawSamples(absl::zipf_distribution<int>(100, 1400.0, 0.6), 1000),
              Each(0));

  // Steep q with v > 1: previously underflowed hx0_minus_hxm_ to 0 and hung in
  // an infinite loop; mathematically P(X >= 1) ~ 1e-100, so samples are 0.
  EXPECT_THAT(DrawSamples(absl::zipf_distribution<int>(100, 800.0, 3.0), 1000),
              Each(0));

  // Large v = 1e16: previously cancelled exp(...) - v; weights (1 + k/v)^(-q)
  // are nearly uniform over [0, 100] with mean ~50.0.
  EXPECT_THAT(SampleMean(absl::zipf_distribution<int>(100, 2.0, 1e16), 10000),
              DoubleNear(50.0, 1.5));
}

double SqueezeThreshold(uint64_t k, double q, double v) {
  return ZipfDistributionTestPeer::s(
      absl::zipf_distribution<uint64_t>(k, q, v));
}

double SampleTailFraction(absl::zipf_distribution<uint64_t> dist,
                          uint64_t threshold, size_t count) {
  absl::random_internal::pcg64_2018_engine rng(0x2B7E151628AED2A6ull);
  size_t hits = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t x = dist(rng);
    EXPECT_LE(x, dist.max());
    if (x >= threshold) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(count);
}

double TailIntegralRoundTrip(const absl::zipf_distribution<uint64_t>& dist,
                             double x) {
  return ZipfDistributionTestPeer::tail_integral_inv(
      dist, ZipfDistributionTestPeer::tail_integral(dist, x));
}

struct ZeroBoundarySummary {
  int zeros_below_d0x5;
  double zero_probability;
  uint64_t first_nonzero_sample;
};

void PrintTo(const ZeroBoundarySummary& s, std::ostream* os) {
  *os << "{zeros_below_d0x5=" << s.zeros_below_d0x5
      << ", zero_probability=" << s.zero_probability
      << ", first_nonzero_sample=" << s.first_nonzero_sample << "}";
}

ZeroBoundarySummary ScanTopUniformGrid(absl::zipf_distribution<uint64_t> dist,
                                       int grid_steps) {
  const double d0x5 = ZipfDistributionTestPeer::d0x5(dist);
  const double total_hat = ZipfDistributionTestPeer::total_hat(dist);
  constexpr double kStep = 1.0 / static_cast<double>(uint64_t{1} << 53);
  int total_zeros = 0;
  int zeros_below = 0;
  uint64_t first_nonzero = 0;
  for (int m = 1; m <= grid_steps; ++m) {
    // Top 52-bit mantissas in [0.5, 1.0): w = 1.0 - m * 2^-53.
    const uint64_t bits = 0xfffffffffffff800ull -
                          static_cast<uint64_t>(m - 1) * (uint64_t{1} << 11);
    absl::random_internal::sequence_urbg urbg({bits, 0x0000000000000001ull});
    const uint64_t sample = dist(urbg);
    const bool accepted_first = (urbg.invocations() == 1);
    const double w = 1.0 - static_cast<double>(m) * kStep;
    const bool below_d0x5 = (w * total_hat < d0x5);
    if (accepted_first && sample == 0) {
      ++total_zeros;
      if (below_d0x5) {
        ++zeros_below;
      }
    } else if (accepted_first && first_nonzero == 0) {
      first_nonzero = sample;
    }
  }
  return {zeros_below, static_cast<double>(total_zeros) * kStep, first_nonzero};
}

MATCHER_P2(MatchesZeroMassWithoutDoubleCounting, expected_prob, tol, "") {
  return arg.zeros_below_d0x5 == 0 &&
         std::abs(arg.zero_probability - expected_prob) <= tol &&
         arg.first_nonzero_sample == 1;
}

TEST(ZipfDistributionTest, SqueezeThresholdIsSoundAndBounded) {
  constexpr uint64_t kLargeK = 1000000000000000000ull;  // 10^18

  // Previously cancelled against xm_, producing s_ = +1.0 (100% false accept).
  EXPECT_THAT(SqueezeThreshold(kLargeK, 2.0, 1e16),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.5, 1e-12)));
  // Previously disabled the squeeze (s_ = -1.0), triggering tail bias.
  EXPECT_THAT(SqueezeThreshold(kLargeK, 1.05, 1e10),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.5, 1e-12)));
  EXPECT_THAT(SqueezeThreshold(1000, 1.5, 1e-300),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.423020306144, 1e-11)));

  // Extreme q -> 1+, steep q, huge v, and degenerate k = 0.
  EXPECT_THAT(SqueezeThreshold(1000, 1.0 + 1e-12, 1e-300),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.448180838243, 1e-11)));
  EXPECT_THAT(SqueezeThreshold(1000, 1000.0, 1.0),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.012401048681, 1e-11)));
  EXPECT_THAT(SqueezeThreshold(1000, 1e10, 1e100),
              AllOf(Ge(0.0), Lt(0.5), DoubleNear(0.5, 1e-12)));
  EXPECT_THAT(SqueezeThreshold(0, 2.0, 1.0), AllOf(Ge(0.0), Lt(0.5)));

  // End-to-end sampling on previously broken parameterizations.
  EXPECT_THAT(SampleTailFraction(zipf_u64(kLargeK, 2.0, 1e16),
                                 uint64_t{10000000000000000ull}, 50000),
              DoubleNear(0.495, 0.008));
  EXPECT_THAT(DrawSamples(zipf_u64(1000, 1.5, 1e-300), 1000), Each(0));
}

TEST(ZipfDistributionTest, TailProbabilityAccurateAboveTwoToTheFiftyThree) {
  constexpr uint64_t kLargeK = 1000000000000000000ull;  // 10^18
  // With s_ = -1.0, k + 0.5 == k for k >= 2^53 caused ~50% tail rejection
  // (P(X >= 2^58) ~ 0.03098 vs exact 0.04245).
  EXPECT_THAT(SampleTailFraction(zipf_u64(kLargeK, 1.05, 1e10),
                                 uint64_t{1} << 58, 200000),
              DoubleNear(0.04245, 0.0025));
}

TEST(ZipfDistributionTest, TailIntegralInversionPreservesPrecisionNearZero) {
  constexpr uint64_t kLargeK = 1000000000000000000ull;  // 10^18

  // zm_ < -0.5 with x << v << xm_: previously inverted x = 0.5 to ~0.3701.
  const zipf_u64 dist_zm_mid(kLargeK, 2.5, 1e15);
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_mid, 0.5), DoubleNear(0.5, 1e-9));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_mid, 1.0), DoubleNear(1.0, 1e-9));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_mid, 1.5), DoubleNear(1.5, 1e-9));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_mid, 2.0), DoubleNear(2.0, 1e-9));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_mid, 10.0), DoubleNear(10.0, 1e-9));

  // zm_ >= -0.5 with d close to D(0).
  const zipf_u64 dist_zm_high(kLargeK, 1.05, 1e15);
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_high, 10.0),
              DoubleNear(10.0, 1e-9));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_high, 100.0),
              DoubleNear(100.0, 1e-9));

  // zm_ < -708 with d close to D(0).
  const zipf_u64 dist_zm_low(kLargeK, 100.0, 1e10);
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_low, 0.5), DoubleNear(0.5, 1e-6));
  EXPECT_THAT(TailIntegralRoundTrip(dist_zm_low, 1.5), DoubleNear(1.5, 1e-6));
}

TEST(ZipfDistributionTest, ZeroMassIsNotDoubleCountedBelowD0x5) {
  constexpr uint64_t kLargeK = 1000000000000000000ull;  // 10^18
  // Exact P(X == 0) ~ 1.500047e-15 (~13.51 ULPs of w below 1.0). Previously,
  // rounding x < 0.5 on the d < d0x5_ branch double-counted k = 0 (+12.5%).
  EXPECT_THAT(ScanTopUniformGrid(zipf_u64(kLargeK, 2.5, 1e15), 64),
              MatchesZeroMassWithoutDoubleCounting(1.500047364598e-15, 7e-17));
  // For q = 2.0, d0_ - d0x5_ rounds to 0.375, so d = d0x5_ - 0.125 inverts to
  // x = 0.5 (rint(0.5) == 0); clamping k < 1 prevents double-counting k = 0.
  EXPECT_THAT(ScanTopUniformGrid(zipf_u64(kLargeK, 2.0, 1e15), 64),
              MatchesZeroMassWithoutDoubleCounting(1.000999999999e-15, 7e-17));
}

TEST(ZipfDistributionTest, SubnormalZmUsesLinearLimit) {
  const double q_min = std::nextafter(1.0, 2.0);
  const double v_max = (std::numeric_limits<double>::max)();

  // (1, nextafter(1, 2), DBL_MAX): zm underflows to -0.0 without the linear
  // limit, collapsing d0x5_ to 0.0 and returning 0 unconditionally instead of
  // uniform {0, 1} with mean 0.5.
  const absl::zipf_distribution<int> dist_k1(1, q_min, v_max);
  EXPECT_EQ(ZipfDistributionTestPeer::d0x5(dist_k1), 1.0);
  EXPECT_EQ(ZipfDistributionTestPeer::total_hat(dist_k1), 2.0);
  EXPECT_THAT(ZipfDistributionTestPeer::tail_integral_inv(dist_k1, 0.5),
              DoubleNear(1.0, 1e-12));
  EXPECT_THAT(SampleMean(dist_k1, 10000), DoubleNear(0.5, 0.03));

  // (100, nextafter(1, 2), DBL_MAX): zm underflows to a 6-bit subnormal without
  // the linear limit, shifting tail_integral_inv(0.5 * d0x5_) from 50.5 to 48.
  const absl::zipf_distribution<int> dist_k100(100, q_min, v_max);
  const double d0x5_k100 = ZipfDistributionTestPeer::d0x5(dist_k100);
  EXPECT_EQ(d0x5_k100, 100.0);
  EXPECT_THAT(
      ZipfDistributionTestPeer::tail_integral_inv(dist_k100, 0.5 * d0x5_k100),
      DoubleNear(50.5, 1e-12));
  EXPECT_THAT(SampleMean(dist_k100, 10000), DoubleNear(50.0, 1.5));

  // Degenerate k = 0 in the linear limit remains 0.
  EXPECT_THAT(DrawSamples(absl::zipf_distribution<int>(0, q_min, v_max), 100),
              Each(0));

  // Large k where |zm| > DBL_MIN yet (q - 1) / v * delta would still underflow
  // to a few-bit subnormal in tail_integral_inv without the q * log_xm guard.
  const zipf_u64 dist_large_k(1000000000000000ull, 1.0 + 5e-15, v_max);
  EXPECT_THAT(TailIntegralRoundTrip(dist_large_k, 0.5), DoubleNear(0.5, 1e-12));
  EXPECT_THAT(TailIntegralRoundTrip(dist_large_k, 100.0),
              DoubleNear(100.0, 1e-12));
}

}  // namespace
