#pragma once

#include "machine_result.hpp"
#include "perspective.hpp"

// Recognition only saves targets. UART is opened exclusively by send_saved_targets.
bool save_machine_targets(const piece_data& origin, const MachineResult& result,
                          const std::string& task_name, const std::string& path = "machine_targets.json");
bool send_saved_targets(const std::string& path, const std::string& device = "/dev/ttyAMA0");
