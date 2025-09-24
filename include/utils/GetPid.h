#ifndef GTEPID_H
#define GTEPID_H

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <time.h>
#include <sys/time.h>
#include <inttypes.h>


namespace NeuTracer
{
    std::vector<uint32_t> bfs_get_procs(uint32_t root_pid);

} // namespace name

#endif // GTEPID_H