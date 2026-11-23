#pragma once

#include <string>
#include <vector>

#include "gnss_online_residual_trace_types.h"

class GnssOnlineResidualTraceWriter
{
  public:
    bool configure(const std::string &path, bool append, std::string *error);
    bool append(const std::vector<GnssOnlineResidualTraceRow> &rows,
        const GnssOnlineResidualSolverSummary &summary, uint64_t optimization_call_id,
        std::string *error) const;
    const std::string &path() const { return path_; }
    bool configured() const { return configured_; }

  private:
    static const char *header();
    std::string path_;
    bool configured_ = false;
};
