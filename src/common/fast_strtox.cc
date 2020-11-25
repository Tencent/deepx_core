// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/fast_strtox.h>
#include <cmath>        // INFINITY
#include <type_traits>  // std::is_signed, ...

namespace deepx_core {

double fast_strtod(const char* s, char** end) noexcept {
  static constexpr double POW10[] = {
      1E-323, 1E-322, 1E-321, 1E-320, 1E-319, 1E-318, 1E-317, 1E-316, 1E-315,
      1E-314, 1E-313, 1E-312, 1E-311, 1E-310, 1E-309, 1E-308, 1E-307, 1E-306,
      1E-305, 1E-304, 1E-303, 1E-302, 1E-301, 1E-300, 1E-299, 1E-298, 1E-297,
      1E-296, 1E-295, 1E-294, 1E-293, 1E-292, 1E-291, 1E-290, 1E-289, 1E-288,
      1E-287, 1E-286, 1E-285, 1E-284, 1E-283, 1E-282, 1E-281, 1E-280, 1E-279,
      1E-278, 1E-277, 1E-276, 1E-275, 1E-274, 1E-273, 1E-272, 1E-271, 1E-270,
      1E-269, 1E-268, 1E-267, 1E-266, 1E-265, 1E-264, 1E-263, 1E-262, 1E-261,
      1E-260, 1E-259, 1E-258, 1E-257, 1E-256, 1E-255, 1E-254, 1E-253, 1E-252,
      1E-251, 1E-250, 1E-249, 1E-248, 1E-247, 1E-246, 1E-245, 1E-244, 1E-243,
      1E-242, 1E-241, 1E-240, 1E-239, 1E-238, 1E-237, 1E-236, 1E-235, 1E-234,
      1E-233, 1E-232, 1E-231, 1E-230, 1E-229, 1E-228, 1E-227, 1E-226, 1E-225,
      1E-224, 1E-223, 1E-222, 1E-221, 1E-220, 1E-219, 1E-218, 1E-217, 1E-216,
      1E-215, 1E-214, 1E-213, 1E-212, 1E-211, 1E-210, 1E-209, 1E-208, 1E-207,
      1E-206, 1E-205, 1E-204, 1E-203, 1E-202, 1E-201, 1E-200, 1E-199, 1E-198,
      1E-197, 1E-196, 1E-195, 1E-194, 1E-193, 1E-192, 1E-191, 1E-190, 1E-189,
      1E-188, 1E-187, 1E-186, 1E-185, 1E-184, 1E-183, 1E-182, 1E-181, 1E-180,
      1E-179, 1E-178, 1E-177, 1E-176, 1E-175, 1E-174, 1E-173, 1E-172, 1E-171,
      1E-170, 1E-169, 1E-168, 1E-167, 1E-166, 1E-165, 1E-164, 1E-163, 1E-162,
      1E-161, 1E-160, 1E-159, 1E-158, 1E-157, 1E-156, 1E-155, 1E-154, 1E-153,
      1E-152, 1E-151, 1E-150, 1E-149, 1E-148, 1E-147, 1E-146, 1E-145, 1E-144,
      1E-143, 1E-142, 1E-141, 1E-140, 1E-139, 1E-138, 1E-137, 1E-136, 1E-135,
      1E-134, 1E-133, 1E-132, 1E-131, 1E-130, 1E-129, 1E-128, 1E-127, 1E-126,
      1E-125, 1E-124, 1E-123, 1E-122, 1E-121, 1E-120, 1E-119, 1E-118, 1E-117,
      1E-116, 1E-115, 1E-114, 1E-113, 1E-112, 1E-111, 1E-110, 1E-109, 1E-108,
      1E-107, 1E-106, 1E-105, 1E-104, 1E-103, 1E-102, 1E-101, 1E-100, 1E-099,
      1E-098, 1E-097, 1E-096, 1E-095, 1E-094, 1E-093, 1E-092, 1E-091, 1E-090,
      1E-089, 1E-088, 1E-087, 1E-086, 1E-085, 1E-084, 1E-083, 1E-082, 1E-081,
      1E-080, 1E-079, 1E-078, 1E-077, 1E-076, 1E-075, 1E-074, 1E-073, 1E-072,
      1E-071, 1E-070, 1E-069, 1E-068, 1E-067, 1E-066, 1E-065, 1E-064, 1E-063,
      1E-062, 1E-061, 1E-060, 1E-059, 1E-058, 1E-057, 1E-056, 1E-055, 1E-054,
      1E-053, 1E-052, 1E-051, 1E-050, 1E-049, 1E-048, 1E-047, 1E-046, 1E-045,
      1E-044, 1E-043, 1E-042, 1E-041, 1E-040, 1E-039, 1E-038, 1E-037, 1E-036,
      1E-035, 1E-034, 1E-033, 1E-032, 1E-031, 1E-030, 1E-029, 1E-028, 1E-027,
      1E-026, 1E-025, 1E-024, 1E-023, 1E-022, 1E-021, 1E-020, 1E-019, 1E-018,
      1E-017, 1E-016, 1E-015, 1E-014, 1E-013, 1E-012, 1E-011, 1E-010, 1E-009,
      1E-008, 1E-007, 1E-006, 1E-005, 1E-004, 1E-003, 1E-002, 1E-001, 1E+000,
      1E+001, 1E+002, 1E+003, 1E+004, 1E+005, 1E+006, 1E+007, 1E+008, 1E+009,
      1E+010, 1E+011, 1E+012, 1E+013, 1E+014, 1E+015, 1E+016, 1E+017, 1E+018,
      1E+019, 1E+020, 1E+021, 1E+022, 1E+023, 1E+024, 1E+025, 1E+026, 1E+027,
      1E+028, 1E+029, 1E+030, 1E+031, 1E+032, 1E+033, 1E+034, 1E+035, 1E+036,
      1E+037, 1E+038, 1E+039, 1E+040, 1E+041, 1E+042, 1E+043, 1E+044, 1E+045,
      1E+046, 1E+047, 1E+048, 1E+049, 1E+050, 1E+051, 1E+052, 1E+053, 1E+054,
      1E+055, 1E+056, 1E+057, 1E+058, 1E+059, 1E+060, 1E+061, 1E+062, 1E+063,
      1E+064, 1E+065, 1E+066, 1E+067, 1E+068, 1E+069, 1E+070, 1E+071, 1E+072,
      1E+073, 1E+074, 1E+075, 1E+076, 1E+077, 1E+078, 1E+079, 1E+080, 1E+081,
      1E+082, 1E+083, 1E+084, 1E+085, 1E+086, 1E+087, 1E+088, 1E+089, 1E+090,
      1E+091, 1E+092, 1E+093, 1E+094, 1E+095, 1E+096, 1E+097, 1E+098, 1E+099,
      1E+100, 1E+101, 1E+102, 1E+103, 1E+104, 1E+105, 1E+106, 1E+107, 1E+108,
      1E+109, 1E+110, 1E+111, 1E+112, 1E+113, 1E+114, 1E+115, 1E+116, 1E+117,
      1E+118, 1E+119, 1E+120, 1E+121, 1E+122, 1E+123, 1E+124, 1E+125, 1E+126,
      1E+127, 1E+128, 1E+129, 1E+130, 1E+131, 1E+132, 1E+133, 1E+134, 1E+135,
      1E+136, 1E+137, 1E+138, 1E+139, 1E+140, 1E+141, 1E+142, 1E+143, 1E+144,
      1E+145, 1E+146, 1E+147, 1E+148, 1E+149, 1E+150, 1E+151, 1E+152, 1E+153,
      1E+154, 1E+155, 1E+156, 1E+157, 1E+158, 1E+159, 1E+160, 1E+161, 1E+162,
      1E+163, 1E+164, 1E+165, 1E+166, 1E+167, 1E+168, 1E+169, 1E+170, 1E+171,
      1E+172, 1E+173, 1E+174, 1E+175, 1E+176, 1E+177, 1E+178, 1E+179, 1E+180,
      1E+181, 1E+182, 1E+183, 1E+184, 1E+185, 1E+186, 1E+187, 1E+188, 1E+189,
      1E+190, 1E+191, 1E+192, 1E+193, 1E+194, 1E+195, 1E+196, 1E+197, 1E+198,
      1E+199, 1E+200, 1E+201, 1E+202, 1E+203, 1E+204, 1E+205, 1E+206, 1E+207,
      1E+208, 1E+209, 1E+210, 1E+211, 1E+212, 1E+213, 1E+214, 1E+215, 1E+216,
      1E+217, 1E+218, 1E+219, 1E+220, 1E+221, 1E+222, 1E+223, 1E+224, 1E+225,
      1E+226, 1E+227, 1E+228, 1E+229, 1E+230, 1E+231, 1E+232, 1E+233, 1E+234,
      1E+235, 1E+236, 1E+237, 1E+238, 1E+239, 1E+240, 1E+241, 1E+242, 1E+243,
      1E+244, 1E+245, 1E+246, 1E+247, 1E+248, 1E+249, 1E+250, 1E+251, 1E+252,
      1E+253, 1E+254, 1E+255, 1E+256, 1E+257, 1E+258, 1E+259, 1E+260, 1E+261,
      1E+262, 1E+263, 1E+264, 1E+265, 1E+266, 1E+267, 1E+268, 1E+269, 1E+270,
      1E+271, 1E+272, 1E+273, 1E+274, 1E+275, 1E+276, 1E+277, 1E+278, 1E+279,
      1E+280, 1E+281, 1E+282, 1E+283, 1E+284, 1E+285, 1E+286, 1E+287, 1E+288,
      1E+289, 1E+290, 1E+291, 1E+292, 1E+293, 1E+294, 1E+295, 1E+296, 1E+297,
      1E+298, 1E+299, 1E+300, 1E+301, 1E+302, 1E+303, 1E+304, 1E+305, 1E+306,
      1E+307, 1E+308,
  };

  double b = 0;
  int64_t e1 = 0, e2 = 0, new_e2;
  int negate, decimal;

  // base
  negate = *s == '-';
  decimal = 0;
  if (*s == '-' || *s == '+') {
    ++s;
  }
  while (*s == '.' || ('0' <= *s && *s <= '9')) {
    if (*s != '.') {
      // no overflow check
      b = b * 10 + *s - '0';
      e1 -= decimal;
    } else {
      decimal = 1;
    }
    ++s;
  }
  if (negate) {
    b = -b;
  }

  // exp
  if ((*s | ('e' ^ 'E')) == 'e') {
    ++s;
    negate = *s == '-';
    decimal = 0;
    if (*s == '-' || *s == '+') {
      ++s;
    }
    while (*s == '.' || ('0' <= *s && *s <= '9')) {
      if (*s != '.') {
        new_e2 = e2 * 10 + *s - '0';
        if (e2 < new_e2) {
          e2 = new_e2;
          e1 -= decimal;
        } else if (e2 > new_e2) {
          // exp overflow
          *end = (char*)s;
          return negate ? -INFINITY : INFINITY;
        }
      } else {
        decimal = 1;
      }
      ++s;
    }
    if (negate) {
      e2 = -e2;
    }
  }

  *end = (char*)s;
  return b * POW10[323 + e1 + e2];
}

template <typename Int>
static Int _fast_strtoi(const char* s, char** end,
                        std::true_type /*is_signed*/) noexcept {
  Int result = 0, value;
  int negative = 0;
  while (*s == ' ' || *s == '\t') {
    ++s;
  }
  if (*s == '-') {
    negative = 1;
    ++s;
  }
  for (;;) {
    value = *s;
    if ('0' <= value && value <= '9') {
      value -= '0';
      // no overflow check
      result = result * 10 + value;
    } else {
      break;
    }
    ++s;
  }
  *end = (char*)s;
  return negative ? -result : result;
}

template <typename Int>
static Int _fast_strtoi(const char* s, char** end,
                        std::false_type /*is_signed*/) noexcept {
  Int result = 0, value;
  while (*s == ' ' || *s == '\t') {
    ++s;
  }
  for (;;) {
    value = *s;
    if ('0' <= value && value <= '9') {
      value -= '0';
      // no overflow check
      result = result * 10 + value;
    } else {
      break;
    }
    ++s;
  }
  *end = (char*)s;
  return result;
}

template <typename Int>
static Int _fast_strtoi(const char* s, char** end) noexcept {
  return _fast_strtoi<Int>(s, end, typename std::is_signed<Int>::type());
}

uint8_t fast_strtou8(const char* s, char** end) noexcept {
  return _fast_strtoi<uint8_t>(s, end);
}

int8_t fast_strtoi8(const char* s, char** end) noexcept {
  return _fast_strtoi<int8_t>(s, end);
}

uint16_t fast_strtou16(const char* s, char** end) noexcept {
  return _fast_strtoi<uint16_t>(s, end);
}

int16_t fast_strtoi16(const char* s, char** end) noexcept {
  return _fast_strtoi<int16_t>(s, end);
}

uint32_t fast_strtou32(const char* s, char** end) noexcept {
  return _fast_strtoi<uint32_t>(s, end);
}

int32_t fast_strtoi32(const char* s, char** end) noexcept {
  return _fast_strtoi<int32_t>(s, end);
}

uint64_t fast_strtou64(const char* s, char** end) noexcept {
  return _fast_strtoi<uint64_t>(s, end);
}

int64_t fast_strtoi64(const char* s, char** end) noexcept {
  return _fast_strtoi<int64_t>(s, end);
}

}  // namespace deepx_core
