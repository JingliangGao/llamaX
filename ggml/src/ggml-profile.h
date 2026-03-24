#pragma once

#include "ggml-impl.h"
#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>
// GGML internal header

#ifdef __cplusplus
#define GGML_THREAD_LOCAL thread_local
#else
#define GGML_THREAD_LOCAL _Thread_local
#endif


static inline uint64_t GetTid(void)
{
    static GGML_THREAD_LOCAL uint64_t tid = 0;

    if (tid == 0) {
        tid = (uint64_t)syscall(SYS_gettid);
    }

    return tid;
}

static inline uint64_t GetPid(void)
{
    static GGML_THREAD_LOCAL uint64_t pid = 0;

    if (pid == 0) {
        pid = (uint64_t)getpid();
    }

    return pid;
}


static inline uint64_t hash_fnv1a(const char *str) {
    uint64_t h = 1469598103934665603ULL;
    while (*str) {
        h ^= (unsigned char)(*str++);
        h *= 1099511628211ULL;
    }
    return h;
}



#ifdef __cplusplus
extern "C" {
#endif

// op profile events & timing (per op / per thread)
enum ggml_profile_event {
    GGML_PROF_OP_START,
    GGML_PROF_OP_SYNC,
    GGML_PROF_OP_END
};

struct ggml_profile_timing {
    uint64_t nsec[GGML_PROF_OP_END + 1]; // event times in nsec
};

struct ggml_profile_output;

struct ggml_profile_data {
    struct ggml_profile_output *output;
    struct ggml_profile_timing ** timing; // per op / per thread timing
};

// check if profiling is enabled for this graph
static inline bool ggml_graph_profile_enabled(const struct ggml_cgraph *cg)
{
    return cg->prof != NULL;
}

// get pointer to the timing data for specific node / thread
// can be used by the backends to populate data collected internally
static inline struct ggml_profile_timing * ggml_graph_profile_timing(const struct ggml_cgraph *cg, int node_n, int ith)
{
    if (!cg->prof) { return NULL; }
    return &cg->prof->timing[node_n][ith];
}

#ifndef GGML_GRAPH_PROFILER

// Stub out all profiler functions


static inline void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads, const char * graph_name)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
    GGML_UNUSED(graph_name);
}

static inline void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads, const char * graph_name)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
    GGML_UNUSED(graph_name);
}

static inline void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_graph_profile_free(struct ggml_cgraph *cg)
{
    GGML_UNUSED(cg);
}

static inline void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(e);
    GGML_UNUSED(node_n);
    GGML_UNUSED(ith);
}

static inline void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
    GGML_UNUSED(this_fn);
    GGML_UNUSED(call_site);
}

static inline void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
    GGML_UNUSED(this_fn);
    GGML_UNUSED(call_site);
}

#else

void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads, const char * graph_name);
void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads, const char * graph_name);
void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads);
void ggml_graph_profile_free(struct ggml_cgraph *cg);
void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith);
void __cyg_profile_func_enter(void *this_fn, void *call_site);
void __cyg_profile_func_exit(void *this_fn, void *call_site);

#endif // GGML_GRAPH_PROFILER

#ifdef __cplusplus
}
#endif
