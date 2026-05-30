/* K.2-spike — sp_npu_shim.c
 *
 * Investigation-scope C shim wrapping the QNN HTP backend lifecycle. Exposes
 * 4 clean C entrypoints to Rust:
 *
 *   int sp_qnn_init(const char* htp_so_path);
 *   int sp_qnn_run_add_smoke(const int8_t* a, const int8_t* b, int8_t* c,
 *                            uint32_t n, uint64_t* out_wall_ns);
 *   void sp_qnn_shutdown(void);
 *   const char* sp_qnn_last_error(void);
 *
 * Per K.2-SPIKE-DESIGN.md, this shim does the multi-level QnnInterface
 * dispatch + the Qnn_Tensor_t / Qnn_OpConfig_t struct construction here,
 * where the header definitions naturally live. Rust binary only calls the
 * 4 entrypoints above.
 *
 * Anti-contamination: this file is REFERENCE-derived from QnnSampleApp.cpp
 * lifecycle (main.cpp:457-528) but re-written from scratch for the lattice.
 * No SDK source is linked in; libQnnHtp.so is dlopen'd at runtime.
 *
 * Smoke graph: ONE ElementWiseAdd node. Inputs a[N], b[N] (INT8), output
 * c[N] (INT8). N=64. Expected c[i] = a[i] + b[i] (clamped per int8 sat).
 *
 * Test inputs: a[i] = (i % 32), b[i] = (i % 16). Both safely in int8 range.
 * Expected c[i] = (i % 32) + (i % 16).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnLog.h"
#include "QnnTypes.h"
#include "QnnOpDef.h"

/* ─── Diagnostics ──────────────────────────────────────────────────────── */

static char g_err[512];

const char* sp_qnn_last_error(void) {
    return g_err;
}

static void set_err(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[sp-npu-spike] ERR: %s\n", g_err);
}

/* ─── State ────────────────────────────────────────────────────────────── */

typedef Qnn_ErrorHandle_t (*QnnInterface_getProvidersFn)(const QnnInterface_t***, uint32_t*);

static struct {
    void* htp_so;
    const QnnInterface_t* iface;             /* not owned; lives inside htp_so */
    Qnn_LogHandle_t log;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    int initialized;
} g = {0};

/* Logger callback. QNN_LOG_LEVEL_INFO etc passed in level. */
static void qnn_log_cb(const char* fmt, QnnLog_Level_t level,
                       uint64_t timestamp, va_list argp) {
    const char* prefix;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:   prefix = "QNN-ERROR";   break;
        case QNN_LOG_LEVEL_WARN:    prefix = "QNN-WARN";    break;
        case QNN_LOG_LEVEL_INFO:    prefix = "QNN-INFO";    break;
        case QNN_LOG_LEVEL_VERBOSE: prefix = "QNN-VERBOSE"; break;
        case QNN_LOG_LEVEL_DEBUG:   prefix = "QNN-DEBUG";   break;
        default:                    prefix = "QNN-?";       break;
    }
    fprintf(stderr, "[%s] ", prefix);
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
}

/* ─── Init ─────────────────────────────────────────────────────────────── */

/* sp_qnn_init — dlopen libQnnHtp.so, fetch dispatch table, create log/backend/
 * context. Returns 0 on success, !=0 on failure (call sp_qnn_last_error). */
int sp_qnn_init(const char* htp_so_path) {
    if (g.initialized) {
        set_err("already initialized");
        return -1;
    }
    g.htp_so = dlopen(htp_so_path, RTLD_NOW | RTLD_LOCAL);
    if (!g.htp_so) {
        set_err("dlopen(%s) failed: %s", htp_so_path, dlerror());
        return -2;
    }
    QnnInterface_getProvidersFn get_providers =
        (QnnInterface_getProvidersFn)dlsym(g.htp_so, "QnnInterface_getProviders");
    if (!get_providers) {
        set_err("dlsym(QnnInterface_getProviders) failed: %s", dlerror());
        dlclose(g.htp_so); g.htp_so = NULL;
        return -3;
    }
    const QnnInterface_t** providers = NULL;
    uint32_t n_providers = 0;
    Qnn_ErrorHandle_t e = get_providers(&providers, &n_providers);
    if (e != QNN_SUCCESS || !providers || n_providers == 0) {
        set_err("QnnInterface_getProviders failed: rc=%lld n=%u",
                (long long)e, n_providers);
        dlclose(g.htp_so); g.htp_so = NULL;
        return -4;
    }
    g.iface = providers[0];
    fprintf(stderr, "[sp-npu-spike] provider name=%s backendId=%u api=%u.%u.%u\n",
            g.iface->providerName ? g.iface->providerName : "<null>",
            g.iface->backendId,
            g.iface->apiVersion.coreApiVersion.major,
            g.iface->apiVersion.coreApiVersion.minor,
            g.iface->apiVersion.coreApiVersion.patch);

    /* Step 1: create log handle. Mandatory first call. */
    e = g.iface->QNN_INTERFACE_VER_NAME.logCreate(qnn_log_cb,
                                                  QNN_LOG_LEVEL_WARN,
                                                  &g.log);
    if (e != QNN_SUCCESS) {
        set_err("logCreate failed: rc=%lld", (long long)e);
        dlclose(g.htp_so); g.htp_so = NULL;
        return -5;
    }

    /* Step 2: create backend handle. */
    e = g.iface->QNN_INTERFACE_VER_NAME.backendCreate(g.log, NULL, &g.backend);
    if (e != QNN_SUCCESS) {
        set_err("backendCreate failed: rc=%lld -- this is the most likely "
                "Signed-PD blocker; check ADSP_LIBRARY_PATH + vendor skel",
                (long long)e);
        g.iface->QNN_INTERFACE_VER_NAME.logFree(g.log);
        dlclose(g.htp_so); g.htp_so = NULL;
        return -6;
    }
    fprintf(stderr, "[sp-npu-spike] backendCreate OK; handle=%p\n",
            (void*)g.backend);

    /* Step 3: create context (skip deviceCreate — many backends accept NULL
     * device for simple workloads; HTP may require it. If contextCreate fails
     * with INVALID_DEVICE, retry path: call deviceCreate first.) */
    e = g.iface->QNN_INTERFACE_VER_NAME.contextCreate(g.backend, NULL, NULL, &g.context);
    if (e != QNN_SUCCESS) {
        set_err("contextCreate (device=NULL) failed: rc=%lld; "
                "K.2 full sprint will need deviceCreate path", (long long)e);
        g.iface->QNN_INTERFACE_VER_NAME.backendFree(g.backend);
        g.iface->QNN_INTERFACE_VER_NAME.logFree(g.log);
        dlclose(g.htp_so); g.htp_so = NULL;
        return -7;
    }
    fprintf(stderr, "[sp-npu-spike] contextCreate OK; handle=%p\n",
            (void*)g.context);

    g.initialized = 1;
    g_err[0] = '\0';
    return 0;
}

/* ─── Smoke run (build graph + execute) ────────────────────────────────── */

/* Build a graph with a single ElementWiseAdd node, finalize, and execute.
 * a, b are INT8 input buffers of N elements; c is INT8 output buffer of N.
 * Returns 0 on success; *out_wall_ns set to execute wall-clock nanoseconds. */
int sp_qnn_run_add_smoke(const int8_t* a, const int8_t* b, int8_t* c,
                          uint32_t n, uint64_t* out_wall_ns) {
    if (!g.initialized) {
        set_err("not initialized");
        return -1;
    }
    if (!a || !b || !c || n == 0) {
        set_err("invalid args");
        return -2;
    }
    Qnn_ErrorHandle_t e;

    /* Step 4: create graph. */
    e = g.iface->QNN_INTERFACE_VER_NAME.graphCreate(g.context,
                                                    "sp_npu_spike_add",
                                                    NULL,
                                                    &g.graph);
    if (e != QNN_SUCCESS) {
        set_err("graphCreate failed: rc=%lld", (long long)e);
        return -3;
    }

    /* Step 5: register 3 tensors (in_a, in_b, out_c) in the graph.
     *
     * IMPORTANT: QNN_TENSOR_TYPE_APP_WRITE = inputs supplied by host.
     *            QNN_TENSOR_TYPE_APP_READ  = outputs read by host.
     *            We use QNN_DATATYPE_SFIXED_POINT_8 (signed int8 with
     *            quantization params) for HTP compatibility. Set
     *            scale=1.0, offset=0 so the integer values pass through
     *            without re-scaling — that lets the bit-identity check
     *            be a pure modular add (with int8 saturation at the
     *            silicon level if any).
     */
    uint32_t dims[1] = { n };

    /* Per QNN error 7004 ("tensor buffer parameters not supported"): at tensor
     * CREATION time, clientBuf must be NULL. Buffers are bound only at execute
     * time. The tensor object is "registered" with the graph (assigned an id +
     * shape + dtype + quant) at create; the data pointer comes later. */
    Qnn_Tensor_t t_a = QNN_TENSOR_INIT;
    t_a.v1.id              = 0;   /* QNN auto-assigns; client provides 0 */
    t_a.v1.name            = "in_a";
    t_a.v1.type            = QNN_TENSOR_TYPE_APP_WRITE;
    t_a.v1.dataFormat      = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t_a.v1.dataType        = QNN_DATATYPE_SFIXED_POINT_8;
    t_a.v1.quantizeParams.encodingDefinition    = QNN_DEFINITION_DEFINED;
    t_a.v1.quantizeParams.quantizationEncoding  = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t_a.v1.quantizeParams.scaleOffsetEncoding.scale  = 1.0f;
    t_a.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t_a.v1.rank            = 1;
    t_a.v1.dimensions      = dims;
    t_a.v1.memType         = QNN_TENSORMEMTYPE_RAW;
    t_a.v1.clientBuf.data     = NULL;
    t_a.v1.clientBuf.dataSize = 0;
    e = g.iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(g.graph, &t_a);
    if (e != QNN_SUCCESS) {
        set_err("tensorCreateGraphTensor(in_a) failed: rc=%lld", (long long)e);
        return -4;
    }
    /* QNN populates t_a.v1.id with the auto-assigned id; capture it. */
    fprintf(stderr, "[sp-npu-spike] tensor in_a registered id=%u\n", t_a.v1.id);

    Qnn_Tensor_t t_b = QNN_TENSOR_INIT;
    t_b.v1.id              = 0;
    t_b.v1.name            = "in_b";
    t_b.v1.type            = QNN_TENSOR_TYPE_APP_WRITE;
    t_b.v1.dataFormat      = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t_b.v1.dataType        = QNN_DATATYPE_SFIXED_POINT_8;
    t_b.v1.quantizeParams.encodingDefinition    = QNN_DEFINITION_DEFINED;
    t_b.v1.quantizeParams.quantizationEncoding  = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t_b.v1.quantizeParams.scaleOffsetEncoding.scale  = 1.0f;
    t_b.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t_b.v1.rank            = 1;
    t_b.v1.dimensions      = dims;
    t_b.v1.memType         = QNN_TENSORMEMTYPE_RAW;
    t_b.v1.clientBuf.data     = NULL;
    t_b.v1.clientBuf.dataSize = 0;
    e = g.iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(g.graph, &t_b);
    if (e != QNN_SUCCESS) {
        set_err("tensorCreateGraphTensor(in_b) failed: rc=%lld", (long long)e);
        return -5;
    }
    fprintf(stderr, "[sp-npu-spike] tensor in_b registered id=%u\n", t_b.v1.id);

    Qnn_Tensor_t t_c = QNN_TENSOR_INIT;
    t_c.v1.id              = 0;
    t_c.v1.name            = "out_c";
    t_c.v1.type            = QNN_TENSOR_TYPE_APP_READ;
    t_c.v1.dataFormat      = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t_c.v1.dataType        = QNN_DATATYPE_SFIXED_POINT_8;
    t_c.v1.quantizeParams.encodingDefinition    = QNN_DEFINITION_DEFINED;
    t_c.v1.quantizeParams.quantizationEncoding  = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t_c.v1.quantizeParams.scaleOffsetEncoding.scale  = 1.0f;
    t_c.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t_c.v1.rank            = 1;
    t_c.v1.dimensions      = dims;
    t_c.v1.memType         = QNN_TENSORMEMTYPE_RAW;
    t_c.v1.clientBuf.data     = NULL;
    t_c.v1.clientBuf.dataSize = 0;
    e = g.iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(g.graph, &t_c);
    if (e != QNN_SUCCESS) {
        set_err("tensorCreateGraphTensor(out_c) failed: rc=%lld", (long long)e);
        return -6;
    }
    fprintf(stderr, "[sp-npu-spike] tensor out_c registered id=%u\n", t_c.v1.id);

    /* Step 6: add the ElementWiseAdd op. The registered tensors (without
     * buffers) are passed in op config so the runtime knows the op connects
     * to the registered ids by name. */
    Qnn_Tensor_t inputs[2]  = { t_a, t_b };
    Qnn_Tensor_t outputs[1] = { t_c };
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name           = "add_op";
    op.v1.packageName    = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName       = QNN_OP_ELEMENT_WISE_ADD;
    op.v1.numOfParams    = 0;
    op.v1.params         = NULL;
    op.v1.numOfInputs    = 2;
    op.v1.inputTensors   = inputs;
    op.v1.numOfOutputs   = 1;
    op.v1.outputTensors  = outputs;
    e = g.iface->QNN_INTERFACE_VER_NAME.graphAddNode(g.graph, op);
    if (e != QNN_SUCCESS) {
        set_err("graphAddNode(ElementWiseAdd) failed: rc=%lld", (long long)e);
        return -7;
    }

    /* Step 7: finalize the graph. NPU op-fusion + kernel selection here. */
    e = g.iface->QNN_INTERFACE_VER_NAME.graphFinalize(g.graph, NULL, NULL);
    if (e != QNN_SUCCESS) {
        set_err("graphFinalize failed: rc=%lld", (long long)e);
        return -8;
    }
    fprintf(stderr, "[sp-npu-spike] graphFinalize OK\n");

    /* Step 8: execute the graph with the actual buffers.
     * Per QNN convention the execute tensors are SHALLOW COPIES of the
     * registered tensor descriptors (same id, same name, same dtype) but
     * with clientBuf populated for the actual run. The runtime binds by id. */
    Qnn_Tensor_t exec_a = t_a;
    exec_a.v1.clientBuf.data     = (void*)a;
    exec_a.v1.clientBuf.dataSize = n * sizeof(int8_t);
    Qnn_Tensor_t exec_b = t_b;
    exec_b.v1.clientBuf.data     = (void*)b;
    exec_b.v1.clientBuf.dataSize = n * sizeof(int8_t);
    Qnn_Tensor_t exec_c = t_c;
    exec_c.v1.clientBuf.data     = (void*)c;
    exec_c.v1.clientBuf.dataSize = n * sizeof(int8_t);
    Qnn_Tensor_t exec_in[2]  = { exec_a, exec_b };
    Qnn_Tensor_t exec_out[1] = { exec_c };
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    e = g.iface->QNN_INTERFACE_VER_NAME.graphExecute(g.graph,
                                                     exec_in, 2,
                                                     exec_out, 1,
                                                     NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (e != QNN_SUCCESS) {
        set_err("graphExecute failed: rc=%lld", (long long)e);
        return -9;
    }
    if (out_wall_ns) {
        *out_wall_ns = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ULL
                     + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    }
    g_err[0] = '\0';
    return 0;
}

/* ─── Shutdown ─────────────────────────────────────────────────────────── */

void sp_qnn_shutdown(void) {
    if (!g.initialized) return;
    if (g.context && g.iface->QNN_INTERFACE_VER_NAME.contextFree) {
        g.iface->QNN_INTERFACE_VER_NAME.contextFree(g.context, NULL);
        g.context = NULL;
    }
    if (g.backend && g.iface->QNN_INTERFACE_VER_NAME.backendFree) {
        g.iface->QNN_INTERFACE_VER_NAME.backendFree(g.backend);
        g.backend = NULL;
    }
    if (g.log && g.iface->QNN_INTERFACE_VER_NAME.logFree) {
        g.iface->QNN_INTERFACE_VER_NAME.logFree(g.log);
        g.log = NULL;
    }
    if (g.htp_so) {
        dlclose(g.htp_so);
        g.htp_so = NULL;
    }
    g.iface = NULL;
    g.initialized = 0;
}
