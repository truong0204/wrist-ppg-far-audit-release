# Source this file (". 06_firmware/idf_env.sh") to use the ESP-IDF v5.5.6 installed inside H32.
H32_FW="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
export IDF_TOOLS_PATH="$H32_FW/.espressif"
export PATH="$H32_FW/.pyshim:$PATH"
. "$H32_FW/esp-idf/export.sh" > /dev/null
