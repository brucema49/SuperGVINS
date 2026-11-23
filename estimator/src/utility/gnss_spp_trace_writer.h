#pragma once

#include <cstdint>
#include <string>

#include <gnss_comm/gnss_spp.hpp>

class GnssSppTraceWriter
{
  public:
    bool configure(const std::string &path, bool append, std::string *error);
    bool append(const gnss_comm::SppTrace &trace, uint64_t spp_call_id, std::string *error) const;
    const std::string &path() const { return path_; }
    bool configured() const { return configured_; }

  private:
    static const char *header();
    std::string path_;
    bool configured_ = false;
};
