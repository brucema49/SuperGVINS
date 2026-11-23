#include "gnss_online_residual_trace_writer.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

#include <gnss_comm/gnss_utility.hpp>

namespace
{

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
                if (errno != ENOENT ||
                    mkdir(current.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
                {
                    setError(error, "Unable to create online residual trace directory " +
                        current + ": " + std::strerror(errno));
                    return false;
                }
            }
            else if (!S_ISDIR(directory_info.st_mode))
            {
                setError(error, "Online residual trace parent is not a directory: " + current);
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

const char *GnssOnlineResidualTraceWriter::header()
{
    return "residual_phase,is_post_initialization,optimization_call_id,window_index,"
           "lower_index,upper_index,interpolation_ratio,obs_gpst_sec,sat_id,constellation,"
           "prn,l1_index,psr_m,psr_std_m,cn0_dbhz,freq_hz,"
           "prior_raw_psr_residual_m,posterior_raw_psr_residual_m,"
           "prior_weighted_psr_residual,posterior_weighted_psr_residual,"
           "prior_psr_weight,posterior_psr_weight,"
           "prior_rcv_ecef_x_m,prior_rcv_ecef_y_m,prior_rcv_ecef_z_m,prior_rcv_dt_m,"
           "posterior_rcv_ecef_x_m,posterior_rcv_ecef_y_m,posterior_rcv_ecef_z_m,"
           "posterior_rcv_dt_m,solver_termination_type,solver_iterations,"
           "solver_successful_steps,has_accepted_update,"
           "solver_initial_cost,solver_final_cost\n";
}

bool GnssOnlineResidualTraceWriter::configure(const std::string &path, bool append,
    std::string *error)
{
    if (configured_ && path_ == path)
        return true;
    if (path.empty())
    {
        setError(error, "Online residual trace file path is empty");
        return false;
    }

    const size_t final_separator = path.find_last_of('/');
    if (final_separator != std::string::npos &&
        !ensureDirectory(path.substr(0, final_separator), error))
        return false;

    const bool write_header = !append || !fileHasContent(path);
    const std::ios_base::openmode mode =
        std::ios::out | (append ? std::ios::app : std::ios::trunc);
    std::ofstream output(path.c_str(), mode);
    if (!output.is_open())
    {
        setError(error, "Unable to open online residual trace file: " + path);
        return false;
    }
    if (write_header)
        output << header();
    output.close();
    if (!output)
    {
        setError(error, "Unable to initialize online residual trace file: " + path);
        return false;
    }

    path_ = path;
    configured_ = true;
    if (error)
        error->clear();
    return true;
}

bool GnssOnlineResidualTraceWriter::append(
    const std::vector<GnssOnlineResidualTraceRow> &rows,
    const GnssOnlineResidualSolverSummary &summary, uint64_t optimization_call_id,
    std::string *error) const
{
    if (!configured_)
    {
        setError(error, "Online residual trace writer is not configured");
        return false;
    }

    std::ofstream output(path_.c_str(), std::ios::out | std::ios::app);
    if (!output.is_open())
    {
        setError(error, "Unable to append online residual trace file: " + path_);
        return false;
    }
    output << std::setprecision(17);
    for (const GnssOnlineResidualTraceRow &row : rows)
    {
        output << "post_initialization,1," << optimization_call_id << ','
               << row.window_index << ',' << row.lower_index << ',' << row.upper_index << ','
               << row.interpolation_ratio << ',' << row.obs_gpst_sec << ','
               << gnss_comm::sat2str(row.sat) << ',' << constellationName(row.system) << ','
               << row.prn << ',' << row.l1_index << ',' << row.psr_m << ','
               << row.psr_std_m << ',' << row.cn0_dbhz << ',' << row.freq_hz << ','
               << row.prior.raw_psr_residual_m << ','
               << row.posterior.raw_psr_residual_m << ','
               << row.prior.weighted_psr_residual << ','
               << row.posterior.weighted_psr_residual << ','
               << row.prior.psr_weight << ',' << row.posterior.psr_weight << ','
               << row.prior.rcv_ecef.x() << ',' << row.prior.rcv_ecef.y() << ','
               << row.prior.rcv_ecef.z() << ',' << row.prior.rcv_dt_m << ','
               << row.posterior.rcv_ecef.x() << ',' << row.posterior.rcv_ecef.y() << ','
               << row.posterior.rcv_ecef.z() << ',' << row.posterior.rcv_dt_m << ','
               << summary.termination_type << ',' << summary.iterations << ','
               << summary.successful_steps << ',' << (summary.successful_steps > 0 ? 1 : 0) << ','
               << summary.initial_cost << ',' << summary.final_cost << '\n';
    }
    output.close();
    if (!output)
    {
        setError(error, "Failed while writing online residual trace file: " + path_);
        return false;
    }
    if (error)
        error->clear();
    return true;
}
