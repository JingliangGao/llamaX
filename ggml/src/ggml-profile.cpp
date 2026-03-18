#include "ggml-profile.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <chrono>

static std::string global_profile_buffer;
static bool global_profile_started = false;
static std::mutex global_profile_mutex;
static int profile_ref_count = 0;
static std::unordered_map<std::string, FILE*> global_profile_streams;

class ProfileWriter {
public:
    ~ProfileWriter() {
        if (global_profile_started) {
            std::lock_guard<std::mutex> lock(global_profile_mutex);
            // remove trailing comma
            if (!global_profile_buffer.empty() && global_profile_buffer.back() == ',') {
                global_profile_buffer.pop_back();
            }
            global_profile_buffer += "]";
            for (auto& p : global_profile_streams) {
                if (p.second != stderr) {
                    fwrite(global_profile_buffer.c_str(), 1, global_profile_buffer.size(), p.second);
                }
            }
            global_profile_buffer = "";
            global_profile_started = false;
        }
    }
};

static ProfileWriter profile_writer;

// std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
#define _GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
#define _GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
#define _GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

using _trimonths = std::chrono::duration<_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
#undef _GLIBCXX_CHRONO_INT64_T

template <class ClockT>
static inline int64_t timeSinceEpoch(const std::chrono::time_point<ClockT>& t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

class ChromeTraceBaseTime {
 public:
  ChromeTraceBaseTime() = default;
  static ChromeTraceBaseTime& singleton();
  void init() {
    get();
  }
  int64_t get() {
    // Make all timestamps relative to 3 month intervals.
    static int64_t base_time = timeSinceEpoch(std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
    return base_time;
  }
};


#ifdef GGML_GRAPH_PROFILER

struct ggml_profile_output {
    const char * prefix;
    FILE *       stream;
};

// global mutex lock, used to protect file writing
static std::mutex profile_file_mutex;

// graph index
static int64_t graph_index = 0;

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}


inline uint64_t transToRelativeTime(int64_t time) {
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  return res > 0 ? (uint64_t)res : 0;
}

extern "C" void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads)
{
    // TODO: make this a param
    const char *env = getenv("GGML_GRAPH_PROFILE");

    if (!env) { return; }

    // The number of threads may change between passes (pp vs tg).
    // Allocate for max_n_threads for simplicity for now.
    // TODO: use aligned allocator

    size_t node_size = sizeof(struct ggml_profile_timing) * GGML_MAX_N_THREADS;
    size_t pvec_size = sizeof(std::intptr_t) * cg->n_nodes;
    size_t time_size = node_size * cg->n_nodes;
    size_t t_size    = pvec_size + time_size + sizeof(ggml_profile_output) + sizeof(ggml_profile_data);

    uint8_t * ptr = (uint8_t *) malloc(t_size);
    if (!ptr) {
        fprintf(stderr, "ggml-profile: failed to allocate profiling data : n_threads %d n_nodes %d\n", n_threads, cg->n_nodes);
        return;
    }
    memset(ptr, 0, t_size);

    // init all pointers
    cg->prof         = (ggml_profile_data *)    ptr; ptr += sizeof(ggml_profile_data);
    cg->prof->output = (ggml_profile_output *)  ptr; ptr += sizeof(ggml_profile_output);
    cg->prof->timing = (ggml_profile_timing **) ptr; ptr += pvec_size;
    for (int i=0; i < cg->n_nodes; i++) {
        cg->prof->timing[i] = (struct ggml_profile_timing *) ptr; ptr += node_size;
    }

    // init the output
    ggml_profile_output *out = cg->prof->output;
    if (!strcmp("stderr", env) || !strcmp("1", env)) {
        out->prefix = "ggml-profile:";
        out->stream = stderr;
    } else {
        out->prefix = "";
        // check if file is open
        auto it = global_profile_streams.find(env);
        if (it != global_profile_streams.end()) {
            out->stream = it->second;
        } else {
            out->stream = fopen(env, "w");
            if (out->stream) {
                global_profile_streams[env] = out->stream;
            }
        }
    }

    profile_ref_count++;
    if (!global_profile_started) {
        global_profile_buffer = "[";
        global_profile_started = true;
    }

}


extern "C" void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { ggml_graph_profile_init(cg, n_threads); }
    if (!cg->prof) { return; }
}

static inline int ggml_profile_format_tensor_dims(char *str, struct ggml_tensor *t)
{
    return sprintf(str, "[%ld,%ld,%ld,%ld]",
        (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], (long) t->ne[3]);
}

static inline void ggml_profile_format_op_dims(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += ggml_profile_format_tensor_dims(p, t->src[0]);

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += ggml_profile_format_tensor_dims(p, t->src[i]);
       }

       p += sprintf(p, " -> ");
    }

    // format self dims separately for better visual alignment
    char self[64];
    ggml_profile_format_tensor_dims(self, t);

    p += sprintf(p, "%12s", self);
}

static inline void ggml_profile_format_op_types(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += sprintf(p, "%s", ggml_type_name(t->src[0]->type));

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += sprintf(p, "%s", ggml_type_name(t->src[i]->type));
       }

       p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%3s", ggml_type_name(t->type));
}

static inline void ggml_profile_format_op_names(char *str, const struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += sprintf(p, "%s", t->src[0]->name);

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += sprintf(p, "%s", t->src[i]->name);
       }

       p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%s", t->name);
}

extern "C" void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { return; }

    ggml_profile_output *out = cg->prof->output;

    uint64_t pid = GetPid();
    // uint64_t tid_base = GetTid();

    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char names[128 * GGML_MAX_SRC];

    // use a buffer to collect all outputs, avoiding multiple file writes
    std::string output_buffer;

    for (int i = 0; i < cg->n_nodes; i++) {
        uint64_t p_nsec = 0;
        uint64_t s_nsec = 0;
        uint64_t t_nsec = 0;
        uint64_t time_stamp = 0;

        // add up per thread counters and reset them
        for (int t=0; t < n_threads; t++) {
            ggml_profile_timing &timing = cg->prof->timing[i][t];

            // accquire time start stamp            JingliangGao 2026/03/16
            if (t == 0) {
                ggml_profile_timing &timing_start = cg->prof->timing[i][0];
                time_stamp = transToRelativeTime(timing_start.nsec[GGML_PROF_OP_START]);
            }

            p_nsec += timing.nsec[GGML_PROF_OP_SYNC] - timing.nsec[GGML_PROF_OP_START];
            s_nsec += timing.nsec[GGML_PROF_OP_END]  - timing.nsec[GGML_PROF_OP_SYNC];
            t_nsec += timing.nsec[GGML_PROF_OP_END]  - timing.nsec[GGML_PROF_OP_START];

            timing.nsec[GGML_PROF_OP_START] = 0;
            timing.nsec[GGML_PROF_OP_SYNC]  = 0;
            timing.nsec[GGML_PROF_OP_END]   = 0;
        }

        // skip ops with no time recorded (e.g. optimized out)
        if (t_nsec == 0) { continue; }

        ggml_profile_format_op_dims(dims, cg->nodes[i]);
        ggml_profile_format_op_types(types, cg->nodes[i]);
        ggml_profile_format_op_names(names, cg->nodes[i]);

        // create JSON string
        char buf[512];

        int len = snprintf(
            buf, sizeof(buf),
            "\n{\n"
            "  \"ph\": \"X\", \"cat\": \"cpu_op\", \"name\": \"%s\", \"pid\": %llu, \"tid\": %llu,\n"
            "  \"ts\": %llu.%03llu, \"dur\": %llu.%03llu,\n"
            "  \"args\": {\n"
            "   \"op dims\": \"%s\", \"op types\": \"%s\", \"tensor names\": \"%s\"\n"
            "  }\n"
            "},",
            ggml_op_name(cg->nodes[i]->op),
            (unsigned long long)pid,
            (unsigned long long)graph_index,
            (unsigned long long)(time_stamp / 1000),
            (unsigned long long)(time_stamp % 1000),
            (unsigned long long)(t_nsec / 1000),
            (unsigned long long)(t_nsec % 1000),
            dims, types, names
        );

        // add into buffer
        output_buffer.append(buf, len);
    }

    {
        std::lock_guard<std::mutex> lock(global_profile_mutex);
        global_profile_buffer += output_buffer;
    }
}

extern "C" void ggml_graph_profile_free(struct ggml_cgraph *cg)
{
    if (!cg->prof) { return; }

    ggml_profile_output *out = cg->prof->output;
    if (out->stream != stderr) {
        fclose(out->stream);
    }

    profile_ref_count--;
    if (profile_ref_count == 0) {
        global_profile_buffer = "";
        global_profile_started = false;
    }

    free(cg->prof); cg->prof = nullptr;
}

extern "C" void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    if (!cg->prof) { return; }

    using clock = std::chrono::high_resolution_clock;

    ggml_profile_timing &timing = cg->prof->timing[node_n][ith];
    timing.nsec[e] = std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

#endif // GGML_GRAPH_PROFILER
