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

#ifndef ABSL_RANDOM_BETA_DISTRIBUTION_H_
#define ABSL_RANDOM_BETA_DISTRIBUTION_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <type_traits>

#include "absl/base/config.h"
#include "absl/meta/type_traits.h"
#include "absl/random/bernoulli_distribution.h"
#include "absl/random/internal/fast_uniform_bits.h"
#include "absl/random/internal/generate_real.h"
#include "absl/random/internal/iostream_state_saver.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace random_internal {

// Computes (expm1(x) - x) / x for |x| <= 0.25 without cancellation.
double BetaExpm1MinusXOverX(double x);
long double BetaExpm1MinusXOverX(long double x);

// Computes (log1p(u) - u) / u for |u| <= 0.25 without cancellation.
double BetaLog1pMinusXOverX(double u);
long double BetaLog1pMinusXOverX(long double u);

}  // namespace random_internal

// absl::beta_distribution:
// Generate a floating-point variate conforming to a Beta distribution:
//   pdf(x) \propto x^(alpha-1) * (1-x)^(beta-1),
// where the params alpha and beta are both strictly positive real values.
//
// The support is the open interval (0, 1), but the return value might be equal
// to 0 or 1, due to numerical errors when alpha and beta are very different.
//
// Usage note: One usage is that alpha and beta are counts of number of
// successes and failures. When the total number of trials are large, consider
// approximating a beta distribution with a Gaussian distribution with the same
// mean and variance. One could use the skewness, which depends only on the
// smaller of alpha and beta when the number of trials are sufficiently large,
// to quantify how far a beta distribution is from the normal distribution.
template <typename RealType = double>
class beta_distribution {
 public:
  using result_type = RealType;

  class param_type {
   public:
    using distribution_type = beta_distribution;

    explicit param_type(result_type alpha, result_type beta)
        : alpha_(alpha), beta_(beta) {
      assert(alpha >= 0);
      assert(beta >= 0);
      assert(alpha <= (std::numeric_limits<result_type>::max)());
      assert(beta <= (std::numeric_limits<result_type>::max)());
      if (alpha == 0 || beta == 0) {
        method_ = DEGENERATE_SMALL;
        if (alpha == beta) {
          inverted_ = false;
          x_ = calc_type(0.5);
        } else {
          inverted_ = (beta < alpha);
          x_ = calc_type(0);
        }
        return;
      }
      // a_ = min(beta, alpha), b_ = max(beta, alpha).
      if (beta < alpha) {
        inverted_ = true;
        a_ = static_cast<calc_type>(beta);
        b_ = static_cast<calc_type>(alpha);
      } else {
        inverted_ = false;
        a_ = static_cast<calc_type>(alpha);
        b_ = static_cast<calc_type>(beta);
      }
      // For threshold values, see also:
      // Evaluation of Beta Generation Algorithms, Ying-Chao Hung, et. al.
      // February, 2009.
      if ((b_ < calc_type(1) && a_ + b_ <= calc_type(1.2)) ||
          a_ <= static_cast<calc_type>(ThresholdForSmallA())) {
        // Choose Joehnk over Cheng when it's faster or when Cheng encounters
        // numerical issues.
        int exp_b = 0;
        std::frexp(b_, &exp_b);
        const calc_type inv_alpha =
            calc_type(1) / static_cast<calc_type>(alpha_);
        const calc_type inv_beta = calc_type(1) / static_cast<calc_type>(beta_);
        if (exp_b <= std::numeric_limits<calc_type>::min_exponent / 2 ||
            std::isinf(inv_alpha) || std::isinf(inv_beta)) {
          // Both parameters are so small that log(u)/alpha and log(v)/beta
          // could both overflow to -inf (or 1/alpha or 1/beta overflows).
          // In this limit Beta(alpha, beta) collapses to
          // Bernoulli(alpha / (alpha + beta)); rescale by 2^-exp_b to avoid
          // overflow and subnormal underflow.
          method_ = DEGENERATE_SMALL;
          const calc_type a_s = std::scalbn(a_, -exp_b);
          const calc_type b_s = std::scalbn(b_, -exp_b);
          x_ = a_s / (a_s + b_s);
          return;
        }
        method_ = JOEHNK;
        a_ = inv_alpha;
        b_ = inv_beta;
        return;
      }
      // If both parameters are at or above ThresholdForLargeA() (since
      // a_ <= b_, checking a_ suffices), the relative standard deviation is
      // <= 1 / sqrt(ThresholdForLargeA()) << ULP, so the variate collapses to
      // the mean a / (a + b). Rescale by b_ so a_ + b_ never overflows.
      if (a_ >= static_cast<calc_type>(ThresholdForLargeA())) {
        method_ = DEGENERATE_LARGE;
        // Note: on PPC for long double, evaluating
        // `std::numeric_limits::max() / ThresholdForLargeA` results in NaN.
        const calc_type r = a_ / b_;
        x_ = (inverted_ ? calc_type(1) : r) / (calc_type(1) + r);
        return;
      }
      if (a_ <= calc_type(1)) {
        method_ = CHENG_BA;
        y_ = calc_type(1) / a_;
        return;
      }
      method_ = CHENG_BB;
      // Compute y = sqrt((a + b - 2) / (2ab - a - b)) without overflowing
      // a * b or underflowing (a - 1) / (b - 1).
      const calc_type fa = (a_ - calc_type(1)) / a_;
      const calc_type fb = (b_ - calc_type(1)) / b_;
      y_ = std::sqrt((fa / b_ + fb / a_) / (fa + fb));
    }

    result_type alpha() const { return alpha_; }
    result_type beta() const { return beta_; }

    friend bool operator==(const param_type& a, const param_type& b) {
      return a.alpha_ == b.alpha_ && a.beta_ == b.beta_;
    }

    friend bool operator!=(const param_type& a, const param_type& b) {
      return !(a == b);
    }

   private:
    friend class beta_distribution;

    using calc_type =
        std::conditional_t<std::is_same_v<RealType, float>, double, RealType>;

#ifdef _MSC_VER
    // MSVC does not have constexpr implementations for std::log and std::exp
    // so they are computed at runtime.
#define ABSL_RANDOM_INTERNAL_LOG_EXP_CONSTEXPR
#else
#define ABSL_RANDOM_INTERNAL_LOG_EXP_CONSTEXPR constexpr
#endif

    // The threshold for whether std::exp(1/a) is finite.
    // Note that this value is quite large, and a smaller a_ is NOT abnormal.
    static ABSL_RANDOM_INTERNAL_LOG_EXP_CONSTEXPR result_type
    ThresholdForSmallA() {
      return result_type(1) /
             std::log((std::numeric_limits<result_type>::max)());
    }

    // The threshold for whether a * std::log(a) is finite.
    static ABSL_RANDOM_INTERNAL_LOG_EXP_CONSTEXPR result_type
    ThresholdForLargeA() {
      return std::exp(
          std::log((std::numeric_limits<result_type>::max)()) -
          std::log(std::log((std::numeric_limits<result_type>::max)())) -
          ThresholdPadding());
    }

#undef ABSL_RANDOM_INTERNAL_LOG_EXP_CONSTEXPR

    // Pad the threshold for large A for long double on PPC. This is done via a
    // template specialization below.
    static constexpr result_type ThresholdPadding() { return 0; }

    enum Method {
      JOEHNK,    // Uses algorithm Joehnk
      CHENG_BA,  // Uses algorithm BA in Cheng
      CHENG_BB,  // Uses algorithm BB in Cheng

      // Note: See also:
      //   Hung et al. Evaluation of beta generation algorithms. Communications
      //   in Statistics-Simulation and Computation 38.4 (2009): 750-770.
      // especially:
      //   Zechner, Heinz, and Ernst Stadlober. Generating beta variates via
      //   patchwork rejection. Computing 50.1 (1993): 1-18.

      DEGENERATE_SMALL,  // a_ is abnormally small.
      DEGENERATE_LARGE,  // a_ is abnormally large.
    };

    result_type alpha_;
    result_type beta_;

    calc_type a_{};  // min(alpha, beta) in Cheng, or 1/alpha_ in JOEHNK
    calc_type b_{};  // max(alpha, beta) in Cheng, or 1/beta_ in JOEHNK
    calc_type x_{};  // Bernoulli p in DEGENERATE_SMALL, or mean in LARGE
    calc_type y_{};  // "beta" in Cheng

    Method method_{};

    // Placing this last for optimal alignment.
    // Whether alpha_ != a_, i.e. true iff alpha_ > beta_.
    bool inverted_{};

    static_assert(std::is_floating_point_v<RealType>,
                  "Class-template absl::beta_distribution<> must be "
                  "parameterized using a floating-point type.");
  };

  beta_distribution() : beta_distribution(1) {}

  explicit beta_distribution(result_type alpha, result_type beta = 1)
      : param_(alpha, beta) {}

  explicit beta_distribution(const param_type& p) : param_(p) {}

  void reset() {}

  // Generating functions
  template <typename URBG>
  result_type operator()(URBG& g) {  // NOLINT(runtime/references)
    return (*this)(g, param_);
  }

  template <typename URBG>
  result_type operator()(URBG& g,  // NOLINT(runtime/references)
                         const param_type& p);

  param_type param() const { return param_; }
  void param(const param_type& p) { param_ = p; }

  result_type(min)() const { return 0; }
  result_type(max)() const { return 1; }

  result_type alpha() const { return param_.alpha(); }
  result_type beta() const { return param_.beta(); }

  friend bool operator==(const beta_distribution& a,
                         const beta_distribution& b) {
    return a.param_ == b.param_;
  }
  friend bool operator!=(const beta_distribution& a,
                         const beta_distribution& b) {
    return a.param_ != b.param_;
  }

 private:
  using calc_type = typename param_type::calc_type;

  template <typename URBG>
  result_type AlgorithmJoehnk(URBG& g,  // NOLINT(runtime/references)
                              const param_type& p);

  template <typename URBG>
  result_type AlgorithmCheng(URBG& g,  // NOLINT(runtime/references)
                             const param_type& p);

  template <typename URBG>
  result_type DegenerateCase(URBG& g,  // NOLINT(runtime/references)
                             const param_type& p) {
    if (p.method_ == param_type::DEGENERATE_SMALL) {
      if (p.x_ == calc_type(0)) {
        return p.inverted_ ? result_type(1) : result_type(0);
      }
      const bool hit =
          absl::bernoulli_distribution(static_cast<double>(p.x_))(g);
      return (hit != p.inverted_) ? result_type(1) : result_type(0);
    }
    return static_cast<result_type>(p.x_);
  }

  param_type param_;
  random_internal::FastUniformBits<uint64_t> fast_u64_;
};

#if defined(__powerpc64__) || defined(__PPC64__) || defined(__powerpc__) || \
    defined(__ppc__) || defined(__PPC__)
// PPC needs a more stringent boundary for long double.
template <>
constexpr long double
beta_distribution<long double>::param_type::ThresholdPadding() {
  return 10;
}
#endif

template <typename RealType>
template <typename URBG>
typename beta_distribution<RealType>::result_type
beta_distribution<RealType>::AlgorithmJoehnk(
    URBG& g,  // NOLINT(runtime/references)
    const param_type& p) {
  using random_internal::GeneratePositiveTag;
  using random_internal::GenerateRealFromBits;

  // Based on Joehnk, M. D. Erzeugung von betaverteilten und gammaverteilten
  // Zufallszahlen. Metrika 8.1 (1964): 5-15.
  // This method is described in Knuth, Vol 2 (Third Edition), pp 134.
  static constexpr calc_type kLn2 =
      calc_type(0.693147180559945309417232121458176568L);  // log(2)
  // Direct method. std::pow is slow for float; when 1/alpha < 1 or 1/beta < 1,
  // pow(u, p.a_) can round to 1.0 and mask z > 1; and when 1/alpha or 1/beta
  // is large, pow frequently underflows below min_normal and wastes work before
  // falling through to log space.
  static constexpr calc_type kMaxDirectPowExponent = calc_type(500);
  const bool try_direct_pow = !std::is_same_v<float, result_type> &&
                              p.a_ >= calc_type(1) && p.b_ >= calc_type(1) &&
                              p.a_ <= kMaxDirectPowExponent &&
                              p.b_ <= kMaxDirectPowExponent;

  for (;;) {
    const calc_type u = static_cast<calc_type>(
        GenerateRealFromBits<double, GeneratePositiveTag, false>(fast_u64_(g)));
    const calc_type v = static_cast<calc_type>(
        GenerateRealFromBits<double, GeneratePositiveTag, false>(fast_u64_(g)));

    if (try_direct_pow) {
      const calc_type x = std::pow(u, p.a_);
      const calc_type y = std::pow(v, p.b_);
      const calc_type z = x + y;
      if (z > calc_type(1)) {
        // Reject if and only if `x + y > 1.0`.
        continue;
      }
      if (z < calc_type(1) &&
          (std::min)(x, y) >= (std::numeric_limits<calc_type>::min)()) {
        // Only return x / z when both x and y are normal; otherwise fall
        // through to the log-transform path to avoid subnormal flush.
        return static_cast<result_type>(x / z);
      }
    }

    // Log transform.
    // x = log(pow(u, 1 / alpha)), y = log(pow(v, 1 / beta)).
    // Since u, v in (0, 1), x and y are negative (possibly -inf for the
    // smaller parameter, but never both -inf because param_type routes
    // max(alpha, beta) <= sqrt(min_normal) to DEGENERATE_SMALL).
    const calc_type x = std::log(u) / static_cast<calc_type>(p.alpha_);
    const calc_type y = std::log(v) / static_cast<calc_type>(p.beta_);
    const calc_type m = (std::max)(x, y);
    // Reject iff log(e^x + e^y) > 0. Since log(e^x + e^y) <= m + log(2),
    // m <= -log(2) guarantees acceptance without computing exp/log1p.
    if (m > -kLn2 &&
        m + std::log1p(std::exp(-std::abs(x - y))) > calc_type(0)) {
      continue;
    }
    if (x >= y) {
      return static_cast<result_type>(calc_type(1) /
                                      (calc_type(1) + std::exp(y - x)));
    }
    const calc_type e = std::exp(x - y);
    return static_cast<result_type>(e / (calc_type(1) + e));
  }
}

template <typename RealType>
template <typename URBG>
typename beta_distribution<RealType>::result_type
beta_distribution<RealType>::AlgorithmCheng(
    URBG& g,  // NOLINT(runtime/references)
    const param_type& p) {
  using random_internal::GeneratePositiveTag;
  using random_internal::GenerateRealFromBits;

  // Based on Cheng, Russell CH. Generating beta variates with nonintegral
  // shape parameters. Communications of the ACM 21.4 (1978): 317-322.
  // (https://dl.acm.org/citation.cfm?id=359482).
  static constexpr calc_type kLogFour =
      calc_type(1.3862943611198906188344642429163531361L);  // log(4)
  static constexpr calc_type kS =
      calc_type(2.6094379124341003746007593332261876L);  // 1 + log(5)
  static constexpr calc_type kMinNormal =
      (std::numeric_limits<calc_type>::min)();
  static constexpr calc_type kMaxQuarter =
      (std::numeric_limits<calc_type>::max)() / calc_type(4);

  const bool use_algorithm_ba = (p.method_ == param_type::CHENG_BA);
  const calc_type a = p.a_;
  const calc_type b = p.b_;
  const calc_type r = a / b;
  const calc_type inv_1pr = calc_type(1) / (calc_type(1) + r);
  const calc_type a_frac = r * inv_1pr;
  const calc_type b_frac = calc_type(1) - a_frac;
  const bool subnormal_a_frac = (a_frac < kMinNormal);

  calc_type v, expm1_v, u;
  for (;;) {
    const double u1 =
        GenerateRealFromBits<double, GeneratePositiveTag, false>(fast_u64_(g));
    const double u2 =
        GenerateRealFromBits<double, GeneratePositiveTag, false>(fast_u64_(g));
    const calc_type logit = static_cast<calc_type>(std::log(u1 / (1.0 - u1)));
    v = p.y_ * logit;
    expm1_v = std::expm1(v);
    if (!std::isfinite(expm1_v)) {
      continue;
    }
    if (subnormal_a_frac) {
      // a < 4 and b > max / 4; avoid subnormal rounding in a_frac.
      if (std::abs(expm1_v) > kMaxQuarter) {
        u = (a * (expm1_v / b)) * inv_1pr;
      } else {
        u = ((a * expm1_v) / b) * inv_1pr;
      }
    } else {
      u = a_frac * expm1_v;
    }
    if (!std::isfinite(u)) {
      continue;
    }

    // Compute a * (expm1(v) - v) without cancellation or v^2 underflow.
    const calc_type a_expm1_minus_v =
        (std::abs(v) <= calc_type(0.25))
            ? (a * v) * random_internal::BetaExpm1MinusXOverX(v)
            : a * (expm1_v - v);

    // Cheng's squeeze: s = a + (gamma * v - log(4)) - a * exp(v)
    //                    = logit - a * (expm1(v) - v) - log(4),
    // since gamma = a + 1 / y and v / y = logit.
    const calc_type s = logit - a_expm1_minus_v - kLogFour;
    // u1, u2 >= 2^-65, so z >= 2^-195 >> DBL_MIN cannot underflow in double.
    const double z = u1 * u1 * u2;
    if (!use_algorithm_ba &&
        s + kS >= calc_type(5) * static_cast<calc_type>(z)) {
      break;
    }
    const calc_type t = static_cast<calc_type>(std::log(z));
    if (!use_algorithm_ba && s >= t) {
      break;
    }

    // Cheng's exact acceptance:
    //   lhs = -(a + b) * log(1 + u) + a * v + logit - log(4),
    // where u = (a / (a + b)) * expm1(v). For |u| <= 0.25, rewrite using
    // (a + b) * u == a * expm1(v) to eliminate O(a * v) cancellation:
    //   -(a + b) * log(1 + u) + a * v
    //     = -(a * expm1(v)) * ((log1p(u) - u) / u) - a * (expm1(v) - v).
    // Note |u| <= 0.25 implies |a * expm1(v)| <= 0.25 * (a + b) <= 0.5 * b.
    calc_type lhs;
    if (std::abs(u) <= calc_type(0.25)) {
      lhs = -(a * expm1_v) * random_internal::BetaLog1pMinusXOverX(u) -
            a_expm1_minus_v + logit - kLogFour;
    } else {
      const calc_type log1p_u = std::log1p(u);
      if (log1p_u > kMaxQuarter / b) {
        continue;
      }
      lhs = -(b * log1p_u) * (calc_type(1) + r) + a * v + logit - kLogFour;
    }
    if (lhs >= t) {
      break;
    }
  }

  calc_type res;
  if (std::abs(v) <= calc_type(0.5)) {
    if (subnormal_a_frac) {
      res =
          p.inverted_ ? (b_frac / (calc_type(1) + u)) : ((a * std::exp(v)) / b);
    } else {
      const calc_type delta = (b_frac * u) / (calc_type(1) + u);
      res = p.inverted_ ? (b_frac - delta) : (a_frac + delta);
    }
  } else if (p.inverted_) {
    res = b_frac / (calc_type(1) + u);
  } else if (u > calc_type(1)) {
    res = calc_type(1) - b_frac / (calc_type(1) + u);
  } else if (subnormal_a_frac) {
    const calc_type exp_v = std::exp(v);
    if (exp_v > kMaxQuarter) {
      res = (a * (exp_v / b)) * inv_1pr / (calc_type(1) + u);
    } else {
      res = ((a * exp_v) / b) * inv_1pr / (calc_type(1) + u);
    }
  } else {
    res = (a_frac * std::exp(v)) / (calc_type(1) + u);
  }
  return (std::min)(static_cast<result_type>(res), result_type(1));
}

template <typename RealType>
template <typename URBG>
typename beta_distribution<RealType>::result_type
beta_distribution<RealType>::operator()(URBG& g,  // NOLINT(runtime/references)
                                        const param_type& p) {
  switch (p.method_) {
    case param_type::JOEHNK:
      return AlgorithmJoehnk(g, p);
    case param_type::CHENG_BA:
    case param_type::CHENG_BB:
      return AlgorithmCheng(g, p);
    default:
      return DegenerateCase(g, p);
  }
}

template <typename CharT, typename Traits, typename RealType>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,  // NOLINT(runtime/references)
    const beta_distribution<RealType>& x) {
  auto saver = random_internal::make_ostream_state_saver(os);
  os.precision(random_internal::stream_precision_helper<RealType>::kPrecision);
  os << x.alpha() << os.fill() << x.beta();
  return os;
}

template <typename CharT, typename Traits, typename RealType>
std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is,  // NOLINT(runtime/references)
    beta_distribution<RealType>& x) {       // NOLINT(runtime/references)
  using result_type = typename beta_distribution<RealType>::result_type;
  using param_type = typename beta_distribution<RealType>::param_type;
  result_type alpha, beta;

  auto saver = random_internal::make_istream_state_saver(is);
  alpha = random_internal::read_floating_point<result_type>(is);
  if (is.fail()) return is;
  beta = random_internal::read_floating_point<result_type>(is);
  if (!is.fail()) {
    x.param(param_type(alpha, beta));
  }
  return is;
}

ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_RANDOM_BETA_DISTRIBUTION_H_
