// SPDX-License-Identifier: MIT

#include <cuda.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <cupti_version.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};
uint32_t g_cuptiRuntimeVersion = CUPTI_API_VERSION;

struct Options {
    std::vector<int> devices;
    bool allDevices = true;
    std::string metric = "gr__cycles_active.avg";
    std::string denominator = "gr__cycles_elapsed.max";
    bool hasDenominator = true;
    bool labels = true;
    bool color = false;
    bool flush = true;
    bool triggerTime = true;
    int width = 32;
    uint64_t sampleUs = 1000;
    uint64_t frameUs = 10000;
    uint64_t sysclkCycles = 1000000;
    size_t hardwareBufferMb = 64;
    uint32_t bufferSamples = 4096;
    double fullScale = 0.0;
    double durationSec = 0.0;
    bool showVersion = false;
};

std::string cuptiError(CUptiResult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuptiGetResultString(result, &name);
    cuptiGetErrorMessage(result, &message);
    std::ostringstream out;
    out << (name ? name : "CUPTI_ERROR") << ": " << (message ? message : "no message");
    return out.str();
}

std::string cudaError(CUresult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << (name ? name : "CUDA_ERROR") << ": " << (message ? message : "no message");
    return out.str();
}

void checkCupti(CUptiResult result, const std::string& what) {
    if (result != CUPTI_SUCCESS) {
        throw std::runtime_error(what + " failed: " + cuptiError(result));
    }
}

void checkCuda(CUresult result, const std::string& what) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(what + " failed: " + cudaError(result));
    }
}

void onSignal(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

std::vector<std::string> split(const std::string& value, char delim) {
    std::vector<std::string> parts;
    std::string current;
    std::istringstream in(value);
    while (std::getline(in, current, delim)) {
        if (!current.empty()) {
            parts.push_back(current);
        }
    }
    return parts;
}

int parseInt(const std::string& value, const std::string& name) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

uint64_t parseU64(const std::string& value, const std::string& name) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (!end || *end != '\0') {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

double parseDouble(const std::string& value, const std::string& name) {
    char* end = nullptr;
    double parsed = std::strtod(value.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return parsed;
}

void usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " [options]\n"
        << "\n"
        << "metric modes:\n"
        << "  Raw ratio mode, default: --metric A --denominator B renders clamp(A / B, 0..1).\n"
        << "  Percent mode:           --metric X --no-denominator --full-scale 100.\n"
        << "  Metric names are CUPTI/PerfWorks names and vary by GPU/CUPTI release.\n"
        << "  Unsupported metric names fail during startup before sampling begins.\n"
        << "\n"
        << "options:\n"
        << "  --devices all|0,1       GPUs to sample (default: all)\n"
        << "  --metric NAME           Ratio numerator or single activity metric\n"
        << "                          Default: gr__cycles_active.avg\n"
        << "  --denominator NAME      Ratio denominator metric\n"
        << "                          Default: gr__cycles_elapsed.max\n"
        << "  --no-denominator        Treat --metric as a single value\n"
        << "  --full-scale VALUE      Value that means 100%; auto if 0 (default)\n"
        << "  --sample-us USEC        PM sampling period for time trigger (default: 1000)\n"
        << "  --frame-us USEC         Output aggregation period (default: 10000)\n"
        << "  --buffer-samples N      Counter-data samples per decode frame (default: 4096)\n"
        << "  --hw-buffer-mb N        CUPTI hardware buffer MB per GPU (default: 64)\n"
        << "  --trigger time|sysclk   PM sampling trigger mode (default: time)\n"
        << "  --sysclk-cycles N       SYSCLK trigger interval cycles (default: 1000000)\n"
        << "  --width CHARS           Braille chars per GPU band (default: 32)\n"
        << "  --duration-sec SEC      Stop after duration; 0 means until Ctrl-C\n"
        << "  --no-labels             Print only bands, no gN labels\n"
        << "  --color                 Color Braille cells by per-cell utilization\n"
        << "  --no-flush              Do not flush stdout after every frame\n"
        << "  --version               Print CUDA/CUPTI versions and exit\n"
        << "  --help                  Show this help\n"
        << "\n"
        << "useful profiles:\n"
        << "  General GPU busy, 1 kHz samples, 100 lines/s:\n"
        << "    " << argv0 << " --sample-us 1000 --frame-us 10000\n"
        << "\n"
        << "  LLM server overview across all GPUs, compact dashboard output:\n"
        << "    " << argv0 << " --devices all --sample-us 1000 --frame-us 10000 --width 32\n"
        << "\n"
        << "  Colorized all-GPU overview with per-line percent gauges:\n"
        << "    " << argv0 << " --devices all --color --sample-us 1000 --frame-us 10000 --width 32\n"
        << "\n"
        << "  LLM SM active view, useful for prefill/decode occupancy changes:\n"
        << "    " << argv0 << " --metric sm__cycles_active.avg --denominator gr__cycles_elapsed.max\n"
        << "\n"
        << "  LLM Tensor Core activity, if available on this CUPTI/GPU:\n"
        << "    " << argv0 << " --metric sm__pipe_tensor_cycles_active.avg --denominator gr__cycles_elapsed.max\n"
        << "\n"
        << "  LLM memory bandwidth pressure, if percent throughput metric is available:\n"
        << "    " << argv0 << " --metric dram__throughput.avg.pct_of_peak_sustained_elapsed --no-denominator --full-scale 100\n"
        << "\n"
        << "  Short high-rate debugging window, 4 kHz PM samples, 200 lines/s:\n"
        << "    " << argv0 << " --sample-us 250 --frame-us 5000 --duration-sec 10 --buffer-samples 8192 --hw-buffer-mb 256\n"
        << "\n"
        << "  Lower-overhead long watch, 500 Hz PM samples, 50 lines/s:\n"
        << "    " << argv0 << " --sample-us 2000 --frame-us 20000\n"
        << "\n"
        << "communication/sync examples:\n"
        << "  PM sampling does not trace CUDA/NCCL sync APIs directly. Infer waits from\n"
        << "  SM/Tensor idle gaps, peer GPUs staying busy, and NVLink/PCIe bursts.\n"
        << "\n"
        << "  Tensor-parallel or data-parallel TX traffic, often all-reduce/reduce-scatter:\n"
        << "    " << argv0 << " --metric nvltx__throughput.avg.pct_of_peak_sustained_elapsed --no-denominator --full-scale 100\n"
        << "\n"
        << "  Tensor-parallel all-gather/RX traffic or expert-parallel all-to-all receive:\n"
        << "    " << argv0 << " --metric nvlrx__throughput.avg.pct_of_peak_sustained_elapsed --no-denominator --full-scale 100\n"
        << "\n"
        << "  PCIe/off-node staging pressure, useful when IB/NIC traffic lands through PCIe:\n"
        << "    " << argv0 << " --metric pcie__throughput.avg.pct_of_peak_sustained_elapsed --no-denominator --full-scale 100\n"
        << "\n"
        << "  Pipeline-parallel bubbles, scheduler stalls, or explicit sync waits:\n"
        << "    " << argv0 << " --devices all --metric sm__cycles_active.avg --denominator gr__cycles_elapsed.max --width 16\n"
        << "\n"
        << "  Decode/KV-cache memory pressure, common when Tensor Cores are not saturated:\n"
        << "    " << argv0 << " --metric dram__throughput.avg.pct_of_peak_sustained_elapsed --no-denominator --full-scale 100\n"
        << "\n"
        << "top 30 LLM metrics to try:\n"
        << "  Use .pct_of_peak... metrics with --no-denominator --full-scale 100.\n"
        << "  Use active-cycle .avg metrics with --denominator gr__cycles_elapsed.max.\n"
        << "   1. gr__cycles_active.avg                              whole GPU busy / PP bubbles\n"
        << "   2. gr__cycles_elapsed.max                             denominator for ratio views\n"
        << "   3. sm__cycles_active.avg                              SM residency / CUDA work present\n"
        << "   4. sm__throughput.avg.pct_of_peak_sustained_elapsed   compute SOL for prefill/train\n"
        << "   5. sm__warps_active.avg.pct_of_peak_sustained_active  achieved occupancy / latency hiding\n"
        << "   6. smsp__warps_active.avg.pct_of_peak_sustained_active per-scheduler occupancy skew\n"
        << "   7. smsp__warps_eligible.sum.per_cycle_active          scheduler has runnable warps\n"
        << "   8. smsp__issue_active.avg.pct_of_peak_sustained_active issue slot utilization\n"
        << "   9. sm__pipe_tensor_cycles_active.avg                  Tensor Core active cycles\n"
        << "  10. sm__inst_executed_pipe_tensor.sum                  Tensor pipe instruction volume\n"
        << "  11. sm__pipe_tensor_op_hmma_cycles_active.avg          FP16/BF16 HMMA Tensor Core activity\n"
        << "  12. sm__inst_executed_pipe_tensor_op_hmma.sum          FP16/BF16 HMMA instruction volume\n"
        << "  13. sm__inst_executed_pipe_tensor_op_imma.sum          INT8/INT4 IMMA instruction volume\n"
        << "  14. sm__pipe_fma_cycles_active.avg                     CUDA-core FP activity fallback\n"
        << "  15. sm__inst_executed_pipe_fma.sum                     FP32/FP64 FMA instruction volume\n"
        << "  16. sm__inst_executed_pipe_alu.sum                     integer/control overhead\n"
        << "  17. dram__throughput.avg.pct_of_peak_sustained_elapsed HBM pressure / decode bottlenecks\n"
        << "  18. dram__bytes_read.sum                               HBM read volume, KV/cache reads\n"
        << "  19. dram__bytes_write.sum                              HBM write volume, optimizer/KV writes\n"
        << "  20. dram__sectors_read.sum                             HBM read transactions\n"
        << "  21. dram__sectors_write.sum                            HBM write transactions\n"
        << "  22. lts__throughput.avg.pct_of_peak_sustained_elapsed  L2 pressure\n"
        << "  23. lts__t_bytes.sum                                   L2 byte traffic\n"
        << "  24. lts__t_sectors_op_read.sum                         L2 read sectors\n"
        << "  25. lts__t_sectors_op_write.sum                        L2 write sectors\n"
        << "  26. l1tex__throughput.avg.pct_of_peak_sustained_elapsed L1/TEX/shared-memory pressure\n"
        << "  27. nvlrx__bytes.sum                                   NVLink receive bytes, TP/EP/DP comm\n"
        << "  28. nvltx__bytes.sum                                   NVLink transmit bytes, TP/EP/DP comm\n"
        << "  29. nvlrx__throughput.avg.pct_of_peak_sustained_elapsed NVLink RX saturation\n"
        << "  30. nvltx__throughput.avg.pct_of_peak_sustained_elapsed NVLink TX saturation\n";
}

Options parseOptions(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--version") {
            opts.showVersion = true;
        } else if (arg == "--devices") {
            const std::string value = needValue(arg);
            opts.devices.clear();
            opts.allDevices = (value == "all");
            if (!opts.allDevices) {
                for (const std::string& part : split(value, ',')) {
                    opts.devices.push_back(parseInt(part, arg));
                }
            }
        } else if (arg == "--metric") {
            opts.metric = needValue(arg);
        } else if (arg == "--denominator") {
            opts.denominator = needValue(arg);
            opts.hasDenominator = true;
        } else if (arg == "--no-denominator") {
            opts.hasDenominator = false;
        } else if (arg == "--full-scale") {
            opts.fullScale = parseDouble(needValue(arg), arg);
        } else if (arg == "--sample-us") {
            opts.sampleUs = parseU64(needValue(arg), arg);
        } else if (arg == "--frame-us") {
            opts.frameUs = parseU64(needValue(arg), arg);
        } else if (arg == "--buffer-samples") {
            opts.bufferSamples = static_cast<uint32_t>(parseU64(needValue(arg), arg));
        } else if (arg == "--hw-buffer-mb") {
            opts.hardwareBufferMb = static_cast<size_t>(parseU64(needValue(arg), arg));
        } else if (arg == "--trigger") {
            const std::string value = needValue(arg);
            if (value == "time") {
                opts.triggerTime = true;
            } else if (value == "sysclk") {
                opts.triggerTime = false;
            } else {
                throw std::runtime_error("invalid --trigger: " + value);
            }
        } else if (arg == "--sysclk-cycles") {
            opts.sysclkCycles = parseU64(needValue(arg), arg);
        } else if (arg == "--width") {
            opts.width = parseInt(needValue(arg), arg);
        } else if (arg == "--duration-sec") {
            opts.durationSec = parseDouble(needValue(arg), arg);
        } else if (arg == "--no-labels") {
            opts.labels = false;
        } else if (arg == "--color") {
            opts.color = true;
        } else if (arg == "--no-flush") {
            opts.flush = false;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opts.width <= 0) {
        throw std::runtime_error("--width must be positive");
    }
    if (opts.sampleUs == 0 || opts.frameUs == 0) {
        throw std::runtime_error("--sample-us and --frame-us must be positive");
    }
    if (opts.bufferSamples == 0) {
        throw std::runtime_error("--buffer-samples must be positive");
    }
    if (opts.hardwareBufferMb == 0) {
        throw std::runtime_error("--hw-buffer-mb must be positive");
    }
    return opts;
}

void printVersions() {
    uint32_t cuptiRuntime = 0;
    const CUptiResult cuptiResult = cuptiGetVersion(&cuptiRuntime);
    int cudaDriver = 0;
    const CUresult cudaResult = cuDriverGetVersion(&cudaDriver);

    std::cerr << "compiled CUPTI API " << CUPTI_API_VERSION << '\n';
    if (cuptiResult == CUPTI_SUCCESS) {
        std::cerr << "runtime CUPTI API  " << cuptiRuntime << '\n';
    } else {
        std::cerr << "runtime CUPTI API  unavailable: " << cuptiError(cuptiResult) << '\n';
    }
    if (cudaResult == CUDA_SUCCESS) {
        std::cerr << "CUDA driver        " << cudaDriver << '\n';
    } else {
        std::cerr << "CUDA driver        unavailable: " << cudaError(cudaResult) << '\n';
    }
}

size_t cuptiStructSizeHostInitialize() {
    if (g_cuptiRuntimeVersion < 130200) {
        return CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Initialize_Params, pHostObject);
    }
    return CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;
}

size_t cuptiStructSizeDeviceSupported() {
    if (g_cuptiRuntimeVersion < 130200) {
        return CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_DeviceSupported_Params, api);
    }
    return CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE;
}

std::string supportLevelName(CUpti_Profiler_Support_Level level) {
    switch (level) {
    case CUPTI_PROFILER_CONFIGURATION_UNKNOWN:
        return "unknown";
    case CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED:
        return "unsupported";
    case CUPTI_PROFILER_CONFIGURATION_DISABLED:
        return "disabled";
    case CUPTI_PROFILER_CONFIGURATION_SUPPORTED:
        return "supported";
    default:
        return "?";
    }
}

void appendUtf8(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

int brailleBitForBottomToTopScan(int rowFromBottom, int col) {
    static const int bits[4][2] = {
        {6, 7}, // dots 7,8
        {2, 5}, // dots 3,6
        {1, 4}, // dots 2,5
        {0, 3}, // dots 1,4
    };
    return bits[rowFromBottom][col];
}

int bitCount(uint8_t value) {
    int count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1;
    }
    return count;
}

int colorForBrailleDots(int activeDots) {
    static const int colors[9] = {
        240, // unused for blank cells unless callers explicitly color them
        27, 33, 39, 45, 82, 118, 190, 196
    };
    activeDots = std::max(0, std::min(8, activeDots));
    return colors[activeDots];
}

void appendFgColor(std::string& out, int color) {
    out += "\033[38;5;";
    out += std::to_string(color);
    out += 'm';
}

int activityPercent(double activity) {
    if (!std::isfinite(activity) || activity < 0.0) {
        activity = 0.0;
    }
    if (activity > 1.0) {
        activity = 1.0;
    }
    return static_cast<int>(std::llround(activity * 100.0));
}

std::string formatGpuLabel(int index, double activity) {
    std::ostringstream out;
    out << 'g' << index << '(' << std::setw(3) << activityPercent(activity) << "%) ";
    return out.str();
}

std::string formatTimestamp(std::chrono::system_clock::time_point now) {
    const auto sinceEpoch = now.time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(sinceEpoch) % std::chrono::seconds(1);
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    if (!localtime_r(&time, &local)) {
        return "00:00:00.000000";
    }

    std::ostringstream out;
    out << std::setfill('0')
        << std::setw(2) << local.tm_hour << ':'
        << std::setw(2) << local.tm_min << ':'
        << std::setw(2) << local.tm_sec << '.'
        << std::setw(6) << micros.count();
    return out.str();
}

std::string renderBraille(double activity, int width, bool color) {
    if (!std::isfinite(activity) || activity < 0.0) {
        activity = 0.0;
    }
    if (activity > 1.0) {
        activity = 1.0;
    }

    const int totalDots = width * 8;
    const int activeDots = static_cast<int>(std::llround(activity * totalDots));
    std::vector<uint8_t> cells(static_cast<size_t>(width), 0);

    int acc = 0;
    for (int dot = 0; dot < totalDots; ++dot) {
        acc += activeDots;
        if (acc < totalDots) {
            continue;
        }
        acc -= totalDots;

        const int subcols = width * 2;
        const int row = dot / subcols;
        const int x = dot % subcols;
        const int cell = x / 2;
        const int col = x % 2;
        cells[static_cast<size_t>(cell)] |= static_cast<uint8_t>(1u << brailleBitForBottomToTopScan(row, col));
    }

    std::string out;
    out.reserve(static_cast<size_t>(width) * (color ? 14 : 3));
    int currentColor = -1;
    for (uint8_t mask : cells) {
        if (color) {
            const int dots = bitCount(mask);
            if (dots != 0) {
                const int desiredColor = colorForBrailleDots(dots);
                if (desiredColor != currentColor) {
                    appendFgColor(out, desiredColor);
                    currentColor = desiredColor;
                }
            }
        }
        appendUtf8(out, 0x2800u + mask);
    }
    if (currentColor != -1) {
        out += "\033[39m";
    }
    return out;
}

double normalizeActivity(double value, const Options& opts, bool ratioMode) {
    if (!std::isfinite(value) || value < 0.0) {
        return 0.0;
    }
    if (ratioMode) {
        return std::max(0.0, std::min(1.0, value));
    }
    if (opts.fullScale > 0.0) {
        return std::max(0.0, std::min(1.0, value / opts.fullScale));
    }
    if (value > 1.0) {
        value /= 100.0;
    }
    return std::max(0.0, std::min(1.0, value));
}

struct GpuSampler {
    int index = 0;
    CUdevice device = 0;
    std::string deviceName;
    std::string chipName;
    std::vector<std::string> metricStorage;
    std::vector<const char*> metricNames;
    CUpti_PmSampling_Object* pm = nullptr;
    CUpti_Profiler_Host_Object* host = nullptr;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    bool started = false;
    bool ratioMode = true;
    bool overflowed = false;
    bool overflowWarned = false;

    GpuSampler() = default;
    ~GpuSampler() {
        cleanup();
    }

    GpuSampler(const GpuSampler&) = delete;
    GpuSampler& operator=(const GpuSampler&) = delete;

    GpuSampler(GpuSampler&& other) noexcept {
        moveFrom(other);
    }

    GpuSampler& operator=(GpuSampler&& other) noexcept {
        if (this != &other) {
            cleanup();
            moveFrom(other);
        }
        return *this;
    }

    void moveFrom(GpuSampler& other) noexcept {
        index = other.index;
        device = other.device;
        deviceName = std::move(other.deviceName);
        chipName = std::move(other.chipName);
        metricStorage = std::move(other.metricStorage);
        metricNames.clear();
        for (const std::string& metric : metricStorage) {
            metricNames.push_back(metric.c_str());
        }
        pm = other.pm;
        host = other.host;
        configImage = std::move(other.configImage);
        counterDataImage = std::move(other.counterDataImage);
        started = other.started;
        ratioMode = other.ratioMode;
        overflowed = other.overflowed;
        overflowWarned = other.overflowWarned;

        other.pm = nullptr;
        other.host = nullptr;
        other.started = false;
        other.overflowWarned = false;
    }

    void init(const Options& opts) {
        checkCuda(cuDeviceGet(&device, index), "cuDeviceGet");
        char name[256] = {};
        checkCuda(cuDeviceGetName(name, static_cast<int>(sizeof(name)), device), "cuDeviceGetName");
        deviceName = name;

        CUpti_Profiler_DeviceSupported_Params support{};
        support.structSize = cuptiStructSizeDeviceSupported();
        support.cuDevice = device;
        support.api = CUPTI_PROFILER_PM_SAMPLING;
        const CUptiResult supportResult = cuptiProfilerDeviceSupported(&support);
        if (supportResult == CUPTI_SUCCESS && support.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
            std::ostringstream out;
            out << "GPU " << index << " PM sampling unsupported: overall=" << supportLevelName(support.isSupported)
                << " arch=" << supportLevelName(support.architecture)
                << " sku=" << supportLevelName(support.sku)
                << " vgpu=" << supportLevelName(support.vGpu)
                << " confidential=" << supportLevelName(support.confidentialCompute)
                << " wsl=" << supportLevelName(support.wsl);
            throw std::runtime_error(out.str());
        }
        if (supportResult != CUPTI_SUCCESS) {
            std::cerr << "warning: gpu " << index << " cuptiProfilerDeviceSupported failed: "
                      << cuptiError(supportResult) << "; continuing to PM sampling setup\n";
        }

        CUpti_Device_GetChipName_Params chip{};
        chip.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
        chip.deviceIndex = static_cast<size_t>(index);
        checkCupti(cuptiDeviceGetChipName(&chip), "cuptiDeviceGetChipName");
        chipName = chip.pChipName ? chip.pChipName : "";

        metricStorage.clear();
        metricStorage.push_back(opts.metric);
        ratioMode = opts.hasDenominator;
        if (ratioMode) {
            metricStorage.push_back(opts.denominator);
        }
        metricNames.clear();
        for (const std::string& metric : metricStorage) {
            metricNames.push_back(metric.c_str());
        }

        CUpti_PmSampling_Enable_Params enable{};
        enable.structSize = CUpti_PmSampling_Enable_Params_STRUCT_SIZE;
        enable.deviceIndex = static_cast<size_t>(index);
        checkCupti(cuptiPmSamplingEnable(&enable), "cuptiPmSamplingEnable");
        pm = enable.pPmSamplingObject;

        CUpti_Profiler_Host_Initialize_Params hostInit{};
        hostInit.structSize = cuptiStructSizeHostInitialize();
        hostInit.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
        hostInit.pChipName = chipName.c_str();
        hostInit.pCounterAvailabilityImage = nullptr;
        hostInit.pSinglePassMetricSetName = nullptr;
        checkCupti(cuptiProfilerHostInitialize(&hostInit), "cuptiProfilerHostInitialize");
        host = hostInit.pHostObject;

        createConfigImage();
        setConfig(opts);
        createCounterDataImage(opts.bufferSamples);
    }

    void createConfigImage() {
        CUpti_Profiler_Host_ConfigAddMetrics_Params add{};
        add.structSize = CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE;
        add.pHostObject = host;
        add.ppMetricNames = metricNames.data();
        add.numMetrics = metricNames.size();
        checkCupti(cuptiProfilerHostConfigAddMetrics(&add), "cuptiProfilerHostConfigAddMetrics");

        CUpti_Profiler_Host_GetConfigImageSize_Params size{};
        size.structSize = CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE;
        size.pHostObject = host;
        checkCupti(cuptiProfilerHostGetConfigImageSize(&size), "cuptiProfilerHostGetConfigImageSize");
        configImage.assign(size.configImageSize, 0);

        CUpti_Profiler_Host_GetConfigImage_Params image{};
        image.structSize = CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE;
        image.pHostObject = host;
        image.configImageSize = configImage.size();
        image.pConfigImage = configImage.data();
        checkCupti(cuptiProfilerHostGetConfigImage(&image), "cuptiProfilerHostGetConfigImage");

        CUpti_Profiler_Host_GetNumOfPasses_Params passes{};
        passes.structSize = CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE;
        passes.configImageSize = configImage.size();
        passes.pConfigImage = configImage.data();
        checkCupti(cuptiProfilerHostGetNumOfPasses(&passes), "cuptiProfilerHostGetNumOfPasses");
        if (passes.numOfPasses != 1) {
            std::ostringstream out;
            out << "metric set requires " << passes.numOfPasses << " passes; PM sampling needs one pass";
            throw std::runtime_error(out.str());
        }
    }

    void setConfig(const Options& opts) {
        CUpti_PmSampling_SetConfig_Params config{};
        config.structSize = CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE;
        config.pPmSamplingObject = pm;
        config.configSize = configImage.size();
        config.pConfig = configImage.data();
        config.hardwareBufferSize = opts.hardwareBufferMb * 1024ull * 1024ull;
        config.samplingInterval = opts.triggerTime ? opts.sampleUs * 1000ull : opts.sysclkCycles;
        config.triggerMode = opts.triggerTime
            ? CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL
            : CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
        config.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST;
        checkCupti(cuptiPmSamplingSetConfig(&config), "cuptiPmSamplingSetConfig");
    }

    void createCounterDataImage(uint32_t maxSamples) {
        CUpti_PmSampling_GetCounterDataSize_Params size{};
        size.structSize = CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE;
        size.pPmSamplingObject = pm;
        size.pMetricNames = metricNames.data();
        size.numMetrics = metricNames.size();
        size.maxSamples = maxSamples;
        checkCupti(cuptiPmSamplingGetCounterDataSize(&size), "cuptiPmSamplingGetCounterDataSize");
        counterDataImage.assign(size.counterDataSize, 0);
        resetCounterDataImage();
    }

    void resetCounterDataImage() {
        CUpti_PmSampling_CounterDataImage_Initialize_Params init{};
        init.structSize = CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE;
        init.pPmSamplingObject = pm;
        init.counterDataSize = counterDataImage.size();
        init.pCounterData = counterDataImage.data();
        checkCupti(cuptiPmSamplingCounterDataImageInitialize(&init), "cuptiPmSamplingCounterDataImageInitialize");
    }

    void start() {
        CUpti_PmSampling_Start_Params startParams{};
        startParams.structSize = CUpti_PmSampling_Start_Params_STRUCT_SIZE;
        startParams.pPmSamplingObject = pm;
        checkCupti(cuptiPmSamplingStart(&startParams), "cuptiPmSamplingStart");
        started = true;
    }

    double decodeFrame(const Options& opts, size_t* completedOut) {
        CUpti_PmSampling_DecodeData_Params decode{};
        decode.structSize = CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE;
        decode.pPmSamplingObject = pm;
        decode.pCounterDataImage = counterDataImage.data();
        decode.counterDataImageSize = counterDataImage.size();
        const CUptiResult decodeResult = cuptiPmSamplingDecodeData(&decode);
        if (decodeResult != CUPTI_SUCCESS && decodeResult != CUPTI_ERROR_OUT_OF_MEMORY) {
            checkCupti(decodeResult, "cuptiPmSamplingDecodeData");
        }
        overflowed = overflowed || decode.overflow || decodeResult == CUPTI_ERROR_OUT_OF_MEMORY;

        CUpti_PmSampling_GetCounterDataInfo_Params info{};
        info.structSize = CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE;
        info.pCounterDataImage = counterDataImage.data();
        info.counterDataImageSize = counterDataImage.size();
        checkCupti(cuptiPmSamplingGetCounterDataInfo(&info), "cuptiPmSamplingGetCounterDataInfo");

        double sum = 0.0;
        size_t count = 0;
        std::vector<double> values(metricNames.size(), 0.0);
        for (size_t sample = 0; sample < info.numCompletedSamples; ++sample) {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params eval{};
            eval.structSize = CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE;
            eval.pHostObject = host;
            eval.pCounterDataImage = counterDataImage.data();
            eval.counterDataImageSize = counterDataImage.size();
            eval.rangeIndex = sample;
            eval.ppMetricNames = metricNames.data();
            eval.numMetrics = metricNames.size();
            eval.pMetricValues = values.data();
            const CUptiResult evalResult = cuptiProfilerHostEvaluateToGpuValues(&eval);
            if (evalResult != CUPTI_SUCCESS) {
                continue;
            }

            double raw = values[0];
            if (ratioMode) {
                const double denom = values.size() > 1 ? values[1] : 0.0;
                raw = denom > 0.0 ? raw / denom : 0.0;
            }
            sum += normalizeActivity(raw, opts, ratioMode);
            ++count;
        }

        resetCounterDataImage();
        if (completedOut) {
            *completedOut = count;
        }
        return count ? (sum / static_cast<double>(count)) : 0.0;
    }

    void stop() noexcept {
        if (!started || !pm) {
            return;
        }
        CUpti_PmSampling_Stop_Params stopParams{};
        stopParams.structSize = CUpti_PmSampling_Stop_Params_STRUCT_SIZE;
        stopParams.pPmSamplingObject = pm;
        cuptiPmSamplingStop(&stopParams);
        started = false;
    }

    void cleanup() noexcept {
        stop();
        if (pm) {
            CUpti_PmSampling_Disable_Params disable{};
            disable.structSize = CUpti_PmSampling_Disable_Params_STRUCT_SIZE;
            disable.pPmSamplingObject = pm;
            cuptiPmSamplingDisable(&disable);
            pm = nullptr;
        }
        if (host) {
            CUpti_Profiler_Host_Deinitialize_Params deinit{};
            deinit.structSize = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE;
            deinit.pHostObject = host;
            cuptiProfilerHostDeinitialize(&deinit);
            host = nullptr;
        }
    }
};

std::vector<int> selectedDevices(const Options& opts) {
    int count = 0;
    checkCuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
    std::vector<int> devices;
    if (opts.allDevices) {
        for (int i = 0; i < count; ++i) {
            devices.push_back(i);
        }
        return devices;
    }
    for (int dev : opts.devices) {
        if (dev < 0 || dev >= count) {
            std::ostringstream out;
            out << "GPU index " << dev << " is out of range 0.." << (count - 1);
            throw std::runtime_error(out.str());
        }
        devices.push_back(dev);
    }
    return devices;
}

struct ProfilerRuntime {
    bool initialized = false;

    void init() {
        CUpti_Profiler_Initialize_Params profilerInit{};
        profilerInit.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
        checkCupti(cuptiProfilerInitialize(&profilerInit), "cuptiProfilerInitialize");
        initialized = true;
    }

    ~ProfilerRuntime() {
        if (!initialized) {
            return;
        }
        CUpti_Profiler_DeInitialize_Params profilerDeinit{};
        profilerDeinit.structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE;
        cuptiProfilerDeInitialize(&profilerDeinit);
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parseOptions(argc, argv);
        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);

        checkCuda(cuInit(0), "cuInit");
        if (opts.showVersion) {
            printVersions();
            return 0;
        }

        uint32_t cuptiRuntime = 0;
        if (cuptiGetVersion(&cuptiRuntime) == CUPTI_SUCCESS) {
            g_cuptiRuntimeVersion = cuptiRuntime;
        }
        if (cuptiRuntime != 0 && cuptiRuntime != CUPTI_API_VERSION) {
            std::cerr << "warning: compiled CUPTI API " << CUPTI_API_VERSION
                      << " but runtime CUPTI API is " << cuptiRuntime << '\n';
        }

        ProfilerRuntime profiler;
        profiler.init();

        std::vector<GpuSampler> samplers;
        for (int dev : selectedDevices(opts)) {
            try {
                GpuSampler sampler;
                sampler.index = dev;
                sampler.init(opts);
                std::cerr << "gpu " << dev << ": " << sampler.deviceName << ", chip " << sampler.chipName
                          << ", metric " << opts.metric;
                if (opts.hasDenominator) {
                    std::cerr << " / " << opts.denominator;
                }
                std::cerr << '\n';
                samplers.push_back(std::move(sampler));
            } catch (const std::exception& e) {
                std::cerr << "skip gpu " << dev << ": " << e.what() << '\n';
            }
        }

        if (samplers.empty()) {
            throw std::runtime_error("no GPUs available for CUPTI PM sampling");
        }

        for (GpuSampler& sampler : samplers) {
            sampler.start();
        }

        const auto started = std::chrono::steady_clock::now();
        auto next = started + std::chrono::microseconds(opts.frameUs);
        auto nextLagWarning = started;
        uint64_t lagEventsSinceWarning = 0;
        uint64_t missedFrameSlotsSinceWarning = 0;
        uint64_t peakLagUsSinceWarning = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_until(next);
            next += std::chrono::microseconds(opts.frameUs);

            if (opts.durationSec > 0.0) {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - started).count();
                if (elapsed >= opts.durationSec) {
                    break;
                }
            }

            std::string line = formatTimestamp(std::chrono::system_clock::now());
            line.push_back(' ');
            for (size_t i = 0; i < samplers.size(); ++i) {
                size_t completed = 0;
                const double activity = samplers[i].decodeFrame(opts, &completed);
                if (samplers[i].overflowed && !samplers[i].overflowWarned) {
                    std::cerr << "warning: gpu " << samplers[i].index
                              << " PM sampling buffer overflowed; increase --hw-buffer-mb, --buffer-samples, or --frame-us\n";
                    samplers[i].overflowWarned = true;
                }
                if (i != 0) {
                    line.push_back(' ');
                }
                if (opts.labels) {
                    line += formatGpuLabel(samplers[i].index, activity);
                }
                line += renderBraille(activity, opts.width, opts.color);
            }
            std::cout << line << '\n';
            if (opts.flush) {
                std::cout.flush();
            }

            const auto now = std::chrono::steady_clock::now();
            if (next < now) {
                const uint64_t lagUs = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(now - next).count());
                const uint64_t missedFrameSlots = (lagUs / opts.frameUs) + 1;
                ++lagEventsSinceWarning;
                missedFrameSlotsSinceWarning += missedFrameSlots;
                peakLagUsSinceWarning = std::max(peakLagUsSinceWarning, lagUs);
                if (now >= nextLagWarning) {
                    std::cerr << "warning: sampler loop is behind by " << lagUs << " us"
                              << " (peak " << peakLagUsSinceWarning << " us, skipped about "
                              << missedFrameSlotsSinceWarning << " frame slots across "
                              << lagEventsSinceWarning << " lag events); try increasing --frame-us, "
                              << "reducing --devices/--width, using --no-flush, or increasing "
                              << "--buffer-samples/--hw-buffer-mb\n";
                    lagEventsSinceWarning = 0;
                    missedFrameSlotsSinceWarning = 0;
                    peakLagUsSinceWarning = 0;
                    nextLagWarning = now + std::chrono::seconds(1);
                }
                next = now + std::chrono::microseconds(opts.frameUs);
            }
        }

        for (GpuSampler& sampler : samplers) {
            if (sampler.overflowed && !sampler.overflowWarned) {
                std::cerr << "warning: gpu " << sampler.index << " PM sampling buffer overflowed; increase --hw-buffer-mb, --buffer-samples, or --frame-us\n";
            }
            sampler.cleanup();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
