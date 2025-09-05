#pragma once
#include <string>
#include <fstream>
#include <filesystem>

namespace hpc {

inline void csv_write_header_if_new(const std::string& path, const std::string& header)

{
    namespace fs = std::filesystem;
    const bool need_header = !fs::exists(path) || fs::file_size(path) == 0;

    std::ofstream f(path, std::ios::app);
    if (!f) {
        throw std::runtime_error("csv_write_header_if_new: cannot open file " + path);
    }

    if (need_header) {
        f << header << "\n";
    }
}

/// Append a single row (string already formatted as CSV).
inline void csv_append_line(const std::string& path,
                            const std::string& line)
{
    std::ofstream f(path, std::ios::app);
    if (!f) {
        throw std::runtime_error("csv_append_line: cannot open file " + path);
    }

    f << line << "\n";
}

}