#include "gnss_spp_trace_writer.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

#include <gnss_comm/gnss_utility.hpp>

namespace
{

const double RAD_TO_DEG = 180.0 / M_PI;

const char *constellationName(uint32_t system)
{
    switch (system)
    {
        case SYS_GPS: return "GPS";
        case SYS_GLO: return "GLO";
        case SYS_BDS: return "BDS";
        case SYS_GAL: return "GAL";
        default: return "UNKNOWN";
    }
}

bool fileHasContent(const std::string &path)
{
    struct stat file_info;
    return stat(path.c_str(), &file_info) == 0 && file_info.st_size > 0;
}

void setError(std::string *error, const std::string &message)
{
    if (error)
        *error = message;
}

bool ensureDirectory(const std::string &directory, std::string *error)
{
    if (directory.empty() || directory == "/")
        return true;

    size_t position = directory[0] == '/' ? 1 : 0;
    while (position <= directory.size())
    {
        const size_t separator = directory.find('/', position);
        const std::string current = directory.substr(0, separator);
        if (!current.empty())
        {
            struct stat directory_info;
            if (stat(current.c_str(), &directory_info) != 0)
            {
                if (errno != ENOENT || mkdir(current.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
                {
                    setError(error, "Unable to create SPP trace directory " + current + ": " +
                        std::strerror(errno));
                    return false;
                }
            }
            else if (!S_ISDIR(directory_info.st_mode))
            {
                setError(error, "SPP trace parent is not a directory: " + current);
                return false;
            }
        }
        if (separator == std::string::npos)
            break;
        position = separator + 1;
    }
    return true;
}

}  // namespace

const char *GnssSppTraceWriter::header()
{
    return "residual_phase,is_post_initialization,spp_call_id,iteration,obs_gpst_sec,sat_id,constellation,prn,l1_index,"
           "used_in_solve,exclude_reason,psr_m,psr_std_m,cn0_dbhz,freq_hz,"
           "pre_residual_m,post_residual_m,pre_az_deg,pre_el_deg,post_az_deg,post_el_deg,"
           "weight,iono_delay_m,trop_delay_m,sv_x_m,sv_y_m,sv_z_m,sv_clock_bias_s,"
           "rcv_pre_x_m,rcv_pre_y_m,rcv_pre_z_m,rcv_pre_dt_gps_m,rcv_pre_dt_glo_m,"
           "rcv_pre_dt_gal_m,rcv_pre_dt_bds_m,rcv_post_x_m,rcv_post_y_m,rcv_post_z_m,"
           "rcv_post_dt_gps_m,rcv_post_dt_glo_m,rcv_post_dt_gal_m,rcv_post_dt_bds_m,"
           "dx_norm,solver_status\n";
}

bool GnssSppTraceWriter::configure(const std::string &path, bool append, std::string *error)
{
    if (configured_ && path_ == path)
        return true;
    if (path.empty())
    {
        setError(error, "SPP trace file path is empty");
        return false;
    }

    const size_t final_separator = path.find_last_of('/');
    if (final_separator != std::string::npos &&
        !ensureDirectory(path.substr(0, final_separator), error))
        return false;

    const bool write_header = !append || !fileHasContent(path);
    const std::ios_base::openmode mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
    std::ofstream output(path.c_str(), mode);
    if (!output.is_open())
    {
        setError(error, "Unable to open SPP trace file: " + path);
        return false;
    }
    if (write_header)
        output << header();
    output.close();
    if (!output)
    {
        setError(error, "Unable to initialize SPP trace file: " + path);
        return false;
    }

    path_ = path;
    configured_ = true;
    if (error)
        error->clear();
    return true;
}

bool GnssSppTraceWriter::append(const gnss_comm::SppTrace &trace, uint64_t spp_call_id,
    std::string *error) const
{
    if (!configured_)
    {
        setError(error, "SPP trace writer is not configured");
        return false;
    }

    std::ofstream output(path_.c_str(), std::ios::out | std::ios::app);
    if (!output.is_open())
    {
        setError(error, "Unable to append SPP trace file: " + path_);
        return false;
    }
    output << std::setprecision(17);
    for (const gnss_comm::SppTraceRow &row : trace.rows)
    {
        output << "initialization_spp,0," << spp_call_id << ',' << row.iteration << ',' << row.obs_gpst_sec << ','
               << gnss_comm::sat2str(row.sat) << ',' << constellationName(row.system) << ','
               << row.prn << ',' << row.l1_index << ',' << (row.used_in_solve ? 1 : 0) << ','
               << row.exclude_reason << ',' << row.psr_m << ',' << row.psr_std_m << ','
               << row.cn0_dbhz << ',' << row.freq_hz << ',' << row.pre_residual_m << ','
               << row.post_residual_m << ',' << row.pre_az_rad * RAD_TO_DEG << ','
               << row.pre_el_rad * RAD_TO_DEG << ',' << row.post_az_rad * RAD_TO_DEG << ','
               << row.post_el_rad * RAD_TO_DEG << ',' << row.weight << ',' << row.iono_delay_m << ','
               << row.trop_delay_m << ',' << row.sv_pos.x() << ',' << row.sv_pos.y() << ','
               << row.sv_pos.z() << ',' << row.sv_clock_bias_s;
        for (int i = 0; i < 7; ++i)
            output << ',' << row.rcv_state_pre(i);
        for (int i = 0; i < 7; ++i)
            output << ',' << row.rcv_state_post(i);
        output << ',' << row.dx_norm << ',' << trace.solver_status << '\n';
    }
    output.close();
    if (!output)
    {
        setError(error, "Failed while writing SPP trace file: " + path_);
        return false;
    }
    if (error)
        error->clear();
    return true;
}
