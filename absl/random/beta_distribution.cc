// Copyright 2026 The Abseil Authors.
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

#include "absl/random/beta_distribution.h"

#include <cmath>  // IWYU pragma: keep

#include "absl/base/config.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace random_internal {
namespace {

// Use std::fma only when hardware FMA is available; software libm fma() adds
// significant call overhead with <= 2 ULP difference on these Horner series.
#if defined(FP_FAST_FMA) || defined(__FP_FAST_FMA) || defined(__FMA__) || \
    defined(__aarch64__) || defined(_M_ARM64)
inline double Fma(double a, double b, double c) { return std::fma(a, b, c); }
#else
inline double Fma(double a, double b, double c) { return a * b + c; }
#endif

#if defined(FP_FAST_FMAL) || defined(__FP_FAST_FMAL)
inline long double Fma(long double a, long double b, long double c) {
  return std::fma(a, b, c);
}
#else
inline long double Fma(long double a, long double b, long double c) {
  return a * b + c;
}
#endif

// Horner evaluation of sum_{k=2}^{14} x^(k-1) / k! = (expm1(x) - x) / x for
// |x| <= 0.25. Truncation error is <= 0.25^14 / 15! < 3e-21.
template <typename T>
T Expm1MinusXOverXImpl(T x) {
  T p = T(1.14707455977297247138516979786821057e-11L);           // 1 / 14!
  p = Fma(p, x, T(1.60590438368216145993923771701549479e-10L));  // 1 / 13!
  p = Fma(p, x, T(2.08767569878680989792100903212014323e-9L));   // 1 / 12!
  p = Fma(p, x, T(2.50521083854417187750521083854417188e-8L));   // 1 / 11!
  p = Fma(p, x, T(2.75573192239858906525573192239858907e-7L));   // 1 / 10!
  p = Fma(p, x, T(2.75573192239858906525573192239858907e-6L));   // 1 / 9!
  p = Fma(p, x, T(2.48015873015873015873015873015873016e-5L));   // 1 / 8!
  p = Fma(p, x, T(1.98412698412698412698412698412698413e-4L));   // 1 / 7!
  p = Fma(p, x, T(1.38888888888888888888888888888888889e-3L));   // 1 / 6!
  p = Fma(p, x, T(8.33333333333333333333333333333333333e-3L));   // 1 / 5!
  p = Fma(p, x, T(4.16666666666666666666666666666666667e-2L));   // 1 / 4!
  p = Fma(p, x, T(1.66666666666666666666666666666666667e-1L));   // 1 / 3!
  p = Fma(p, x, T(0.5));                                         // 1 / 2!
  return x * p;
}

// Using w = u / (2 + u), log(1 + u) = 2 * atanh(w), so
//   (log1p(u) - u) / u
//     = w * (-1 + (2 * w / (2 + u)) * sum_{k=0}^{10} w^(2k) / (2k + 3)).
// For |u| <= 0.25, |w| <= 1/7, so truncation error <= (1/7)^22 / 25 < 2e-20.
template <typename T>
T Log1pMinusXOverXImpl(T u) {
  const T denom = T(2) + u;
  const T w = u / denom;
  const T w2 = w * w;
  T p = T(4.34782608695652173913043478260869565e-2L);            // 1 / 23
  p = Fma(p, w2, T(4.76190476190476190476190476190476190e-2L));  // 1 / 21
  p = Fma(p, w2, T(5.26315789473684210526315789473684211e-2L));  // 1 / 19
  p = Fma(p, w2, T(5.88235294117647058823529411764705882e-2L));  // 1 / 17
  p = Fma(p, w2, T(6.66666666666666666666666666666666667e-2L));  // 1 / 15
  p = Fma(p, w2, T(7.69230769230769230769230769230769231e-2L));  // 1 / 13
  p = Fma(p, w2, T(9.09090909090909090909090909090909091e-2L));  // 1 / 11
  p = Fma(p, w2, T(1.11111111111111111111111111111111111e-1L));  // 1 / 9
  p = Fma(p, w2, T(1.42857142857142857142857142857142857e-1L));  // 1 / 7
  p = Fma(p, w2, T(2.00000000000000000000000000000000000e-1L));  // 1 / 5
  p = Fma(p, w2, T(3.33333333333333333333333333333333333e-1L));  // 1 / 3
  return w * Fma(T(2) * w / denom, p, T(-1));
}

}  // namespace

double BetaExpm1MinusXOverX(double x) { return Expm1MinusXOverXImpl(x); }

long double BetaExpm1MinusXOverX(long double x) {
  return Expm1MinusXOverXImpl(x);
}

double BetaLog1pMinusXOverX(double u) { return Log1pMinusXOverXImpl(u); }

long double BetaLog1pMinusXOverX(long double u) {
  return Log1pMinusXOverXImpl(u);
}

}  // namespace random_internal
ABSL_NAMESPACE_END
}  // namespace absl
