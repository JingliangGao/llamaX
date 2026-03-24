#include "ggml-profile.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include <string>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <chrono>

#define CACHE_SIZE 81920   /* Size : 8192 : 128k, 81920 : 1 M */

static std::string global_profile_buffer;
static bool global_profile_started = false;
static std::mutex global_profile_mutex;
static int profile_ref_count = 0;
static std::unordered_map<std::string, FILE*> global_profile_streams;

__attribute__((no_instrument_function))
static inline uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}


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
    FILE * stream;
};

// global mutex lock, used to protect file writing
static std::mutex profile_file_mutex;

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}


inline uint64_t transToRelativeTime(int64_t time) {
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  return res > 0 ? (uint64_t)res : 0;
}

extern "C" void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads, const char * graph_name)
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
        out->prefix = "unknown";
        out->stream = stderr;
    } else {
        out->prefix = graph_name;
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


extern "C" void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads, const char * graph_name)
{
    if (!cg->prof) { ggml_graph_profile_init(cg, n_threads, graph_name); }
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
        uint64_t pid = GetPid();
        uint64_t tid = GetTid();

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
            "  \"ph\": \"X\", \"cat\": \"%s\", \"name\": \"%s\", \"pid\": %llu, \"tid\": \"%s\",\n"
            "  \"ts\": %llu.%03llu, \"dur\": %llu.%03llu,\n"
            "  \"args\": {\n"
            "   \"pid\": %llu, \"tid\": %llu,\n"
            "   \"op dims\": \"%s\", \"op types\": \"%s\", \"tensor names\": \"%s\"\n"
            "  }\n"
            "},",
            cg->prof->output->prefix,
            ggml_op_name(cg->nodes[i]->op),
            (unsigned long long)pid,
            cg->prof->output->prefix,
            (unsigned long long)(time_stamp / 1000),
            (unsigned long long)(time_stamp % 1000),
            (unsigned long long)(t_nsec / 1000),
            (unsigned long long)(t_nsec % 1000),
            (unsigned long long)pid, (unsigned long long)tid,
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

    ggml_profile_timing &timing = cg->prof->timing[node_n][ith];
    timing.nsec[e] = get_time_ns();
}


/* func: cache all functions {'pointer' : name} */
typedef struct {
    void *addr;
    const char *name;
} cache_entry;

/* func: cache ggml-/llama- functions {'pointer' : start_time} */
typedef struct {
    void *fn;
    uint64_t start_time;
} stack_entry;

static cache_entry g_cache[CACHE_SIZE];                 /* SIZE = 16 Byte * CACHE_SIZE */
static stack_entry g_stack[int(CACHE_SIZE * 0.25)];     /* SIZE = 16 Byte * (CACHE_SIZE * 0.25) */
static int g_stack_top = 0;
static int g_cache_count = 0;

/* linear search */
__attribute__((no_instrument_function))
static const char *lookup_symbol(void *fn) {
    for (int i = 0; i < g_cache_count; i++) {
        if (g_cache[i].addr == fn) {
            return g_cache[i].name;
        }
    }
    return NULL;
}

__attribute__((no_instrument_function))
static const char *resolve_symbol(void *fn) {
    const char *name = lookup_symbol(fn);
    if (name) {
        return name;
    }

    Dl_info info;
    if (dladdr(fn, &info) && info.dli_sname) {

        if (g_cache_count < CACHE_SIZE) {
            g_cache[g_cache_count].addr = fn;
            g_cache[g_cache_count].name = info.dli_sname;
            g_cache_count++;
        }

        if (strstr(info.dli_sname, "ggml") ||
            strstr(info.dli_sname, "llama")) {
            return info.dli_sname;
        }
    }

    return NULL;
}

extern "C" __attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    (void)call_site;

    const char *name = resolve_symbol(this_fn);
    if (!name) return;

    /* push {'pointer' : start_time} into stack */
    if (g_stack_top < int(CACHE_SIZE * 0.25)) {
        g_stack[g_stack_top].fn = this_fn;
        g_stack[g_stack_top].start_time = get_time_ns();
        g_stack_top++;
    }

}

extern "C" __attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)call_site;

    // parse func name from pointer
    const char *name = resolve_symbol(this_fn);
    if (!name) return;

    if (g_stack_top <= 0) return;
    g_stack_top--;
    stack_entry *entry = &g_stack[g_stack_top];

    uint64_t end = get_time_ns();
    uint64_t duration = end - entry->start_time;
    uint64_t pid = GetPid();
    uint64_t tid = GetTid();
    uint64_t time_stamp = transToRelativeTime(entry->start_time);

    // create JSON string
    std::string output_buffer;
    char buf[512];

    int len = snprintf(
        buf, sizeof(buf),
        "\n{\n"
        "  \"ph\": \"X\", \"cat\": \"%s\", \"name\": \"%s\", \"pid\": %llu, \"tid\": %llu,\n"
        "  \"ts\": %llu.%03llu, \"dur\": %llu.%03llu , \n"
        "  \"args\": {\n"
        "   \"pid\": %llu, \"tid\": %llu \n"
        "  }\n"
        "},",
        "call_stack",
        name,
        (unsigned long long)(0),
        (unsigned long long)(tid),
        (unsigned long long)(time_stamp / 1000),
        (unsigned long long)(time_stamp % 1000),
        (unsigned long long)(duration / 1000),
        (unsigned long long)(duration % 1000),
        (unsigned long long)(pid), (unsigned long long)(tid)
    );

    // add into global buffer
    global_profile_buffer.append(buf, len);
}

#endif // GGML_GRAPH_PROFILER
