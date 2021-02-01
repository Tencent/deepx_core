// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/profile_util.h>
#include <deepx_core/dx_log.h>
#include <algorithm>  // std::sort
#include <cstdio>
#include <sstream>

namespace deepx_core {

/************************************************************************/
/* ProfileItem */
/************************************************************************/
void DumpProfileItems(std::vector<ProfileItem>* items) {
  std::sort(items->begin(), items->end(),
            [](const ProfileItem& a, const ProfileItem& b) {
              return a.nanosecond > b.nanosecond;
            });

  double total_nanosecond = 0;
  for (const ProfileItem& item : *items) {
    total_nanosecond += item.nanosecond;
  }

  const char* unit;
  double unit_scale;
  if (total_nanosecond > 60 * 1e+9) {
    // more than 1 minute
    unit = "minutes";
    unit_scale = 60 * 1e+9;
  } else if (total_nanosecond > 1e+9) {
    // more than 1 second
    unit = "seconds";
    unit_scale = 1e+9;
  } else if (total_nanosecond > 1e+6) {
    // more than 1 millisecond
    unit = "milliseconds";
    unit_scale = 1e+6;
  } else if (total_nanosecond > 1e+3) {
    // more than 1 microsecond
    unit = "microseconds";
    unit_scale = 1e+3;
  } else {
    unit = "nanoseconds";
    unit_scale = 1;
  }

  char buf[256];
  std::ostringstream os;
  os << std::endl;
  snprintf(buf, sizeof(buf), "%-64s %12s %12s", "phase", unit, "percentage");
  os << buf << std::endl;
  os << std::string(64 + 12 + 12 + 2, '-') << std::endl;
  for (const ProfileItem& item : *items) {
    snprintf(buf, sizeof(buf), "%-64s %12.3f %12.3f", item.phase.c_str(),
             item.nanosecond / unit_scale,
             item.nanosecond / total_nanosecond * 100);
    os << buf << std::endl;
  }
  DXINFO("%s", os.str().c_str());
}

}  // namespace deepx_core
