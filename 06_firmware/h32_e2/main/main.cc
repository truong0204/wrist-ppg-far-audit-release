// H32 T6 firmware (ESP-IDF v5.5.6, ESP-WROOM-32): replay of 2-s chunks of PPG (64 Hz) and ACC (32 Hz) over UART0;
// per 8-s window (stride 2 s): spectra (float32) -> SpecFCN INT8 (TFLite Micro + ESP-NN) -> softmax, SQI, causal HMM
// (float32) -> ACC state, state-time blocks, debounce, rules K and R (float32). Sends back HR, SQI, state, block
// records and per-stage times. PLAN_T6 §1-§4. Protocol: 05_experiments/t6_mcu/replay.py.
#include <cstdio>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_idf_version.h"
#include "esp_private/esp_clk.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
extern "C" {
#include "h32_core.h"
extern const uint8_t h32_model[];
extern const unsigned int h32_model_len;
}

#define UARTN UART_NUM_0
#define BAUD 921600
#define ARENA_BYTES (96 * 1024)

static uint8_t s_arena[ARENA_BYTES] __attribute__((aligned(16)));
static float s_ppg[512], s_acc[256 * 3];            // last 8 s
static float s_spec[256 * 4];
static h32_hmm_t s_hmm;
static h32_rules_t s_rules;
static uint8_t s_rx[4 + 128 * 4 + 192 * 4 + 2];   // seq, PPG, ACC, checksum (tag read before)
static uint32_t s_chunks = 0, s_win = 0;
static int64_t s_last_seq = -1;                     // stall guard: a repeated chunk gets the cached reply
static uint8_t s_last[400]; static int s_last_len = 0;

static void rd(uint8_t *b, int n) { int got = 0; while (got < n) { int r = uart_read_bytes(UARTN, b + got, n - got, portMAX_DELAY); if (r > 0) got += r; } }
static void wr(const void *b, int n) { uart_write_bytes(UARTN, (const char *)b, n); }
static uint16_t ck(const uint8_t *b, int n) { uint32_t s = 0; for (int i = 0; i < n; i++) s += b[i]; return (uint16_t)s; }

static void send_text(const char tag[4], const char *txt) {
    uint16_t n = (uint16_t)strlen(txt); wr(tag, 4); wr(&n, 2); wr(txt, n);
    uint16_t c = ck((const uint8_t *)txt, n); wr(&c, 2);
}

static void reset_stream(void) {
    memset(s_ppg, 0, sizeof s_ppg); memset(s_acc, 0, sizeof s_acc);
    h32_hmm_init(&s_hmm); h32_rules_init(&s_rules); s_chunks = 0; s_win = 0; s_last_seq = -1; s_last_len = 0;
}

#pragma pack(push, 1)
struct Res {
    char tag[4]; uint32_t win; float hr, sqi; uint8_t state, mon;
    uint32_t t_spec, t_inf, t_post, t_rules, t_total;
    uint8_t nblk; h32_block_t blk;
    uint32_t seq; uint32_t in_hash;          // amendment 1: FNV-1a of the 1,024 int8 model inputs
    int8_t logits[256];                      // amendment 1 (b): raw int8 model output
};
#pragma pack(pop)

extern "C" void app_main(void) {
    uart_config_t cfg = {};
    cfg.baud_rate = BAUD; cfg.data_bits = UART_DATA_8_BITS; cfg.parity = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1; cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE; cfg.source_clk = UART_SCLK_DEFAULT;
    uart_driver_install(UARTN, 8192, 4096, 0, nullptr, 0); uart_param_config(UARTN, &cfg);
#if CONFIG_IDF_TARGET_ESP32
    uart_set_pin(UARTN, 1, 3, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);   // ESP32: UART0 TX GPIO1, RX GPIO3 (console is off)
#elif CONFIG_IDF_TARGET_ESP32S3
    uart_set_pin(UARTN, 43, 44, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE); // ESP32-S3: UART0 TX GPIO43, RX GPIO44
#endif

    const tflite::Model *model = tflite::GetModel(h32_model);
    static tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddConv2D(); resolver.AddAdd(); resolver.AddReshape();
    static tflite::MicroInterpreter interp(model, resolver, s_arena, ARENA_BYTES);
    bool ok = interp.AllocateTensors() == kTfLiteOk;
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);
    reset_stream();

    char info[512];
#ifdef CONFIG_NN_OPTIMIZED
    const char *nn = "optimized";
#elif defined(CONFIG_NN_ANSI_C)
    const char *nn = "ansi_c";
#else
    const char *nn = "unknown";
#endif
#ifdef CONFIG_COMPILER_OPTIMIZATION_PERF
    const char *opt = "perf(-O2)";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_SIZE)
    const char *opt = "size(-Os)";
#else
    const char *opt = "other";
#endif
    snprintf(info, sizeof info,
             "{\"ok\":%d,\"idf\":\"%s\",\"cpu_mhz\":%d,\"esp_nn\":\"%s\",\"opt\":\"%s\",\"arena_bytes\":%d,\"arena_used\":%u,"
             "\"model_bytes\":%u,\"in_type\":%d,\"out_type\":%d,\"psram\":%d,\"free_heap\":%u}",
             ok, esp_get_idf_version(), (int)(esp_clk_cpu_freq() / 1000000), nn, opt, ARENA_BYTES,
             (unsigned)interp.arena_used_bytes(), h32_model_len, (int)in->type, (int)out->type,
#ifdef CONFIG_SPIRAM
             1,
#else
             0,
#endif
             (unsigned)esp_get_free_heap_size());

    uint8_t tag[4];
    for (;;) {
        rd(tag, 4);
        if (!memcmp(tag, "H32S", 4)) { reset_stream(); send_text("H32I", info); continue; }
        if (!memcmp(tag, "H32E", 4)) {
            char st[200];
            snprintf(st, sizeof st, "{\"min_free_heap\":%u,\"stack_hwm_bytes\":%u,\"arena_used\":%u,\"windows\":%u}",
                     (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
                     (unsigned)(uxTaskGetStackHighWaterMark(nullptr) * sizeof(StackType_t)), (unsigned)interp.arena_used_bytes(), (unsigned)s_win);
            send_text("H32X", st); continue;
        }
        if (memcmp(tag, "H32W", 4)) continue;                 // resynchronise on the next tag
        rd(s_rx, sizeof s_rx);
        uint32_t seq; memcpy(&seq, s_rx, 4);
        const float *pp = (const float *)(s_rx + 4), *aa = (const float *)(s_rx + 4 + 512);
        uint16_t c; memcpy(&c, s_rx + 4 + 512 + 768, 2);
        if (c != ck(s_rx, 4 + 512 + 768)) { wr("H32N", 4); wr(&seq, 4); continue; }
        if ((int64_t)seq == s_last_seq) { wr(s_last, s_last_len); continue; }   // resend, do not reprocess
        memmove(s_ppg, s_ppg + 128, sizeof(float) * 384); memcpy(s_ppg + 384, pp, sizeof(float) * 128);
        memmove(s_acc, s_acc + 192, sizeof(float) * 576); memcpy(s_acc + 576, aa, sizeof(float) * 192);
        s_chunks++;
        s_last_seq = seq;
        if (s_chunks < 4) { memcpy(s_last, "H32A", 4); memcpy(s_last + 4, &seq, 4); s_last_len = 8; wr(s_last, 8); continue; }
        static Res r; memset(&r, 0, sizeof r); memcpy(r.tag, "H32R", 4); r.win = s_win; r.seq = seq;
        int64_t t0 = esp_timer_get_time();
        h32_spectra(s_ppg, s_acc, s_spec); h32_quant_input(s_spec, in->data.int8); r.state = (uint8_t)h32_acc_state(s_acc);
        int64_t t1 = esp_timer_get_time();
        uint32_t hsh = 2166136261u; for (int i = 0; i < 1024; i++) { hsh ^= (uint8_t)in->data.int8[i]; hsh *= 16777619u; }
        r.in_hash = hsh;                     // before Invoke (TFLM may reuse the input buffer); not timed
        int64_t t1b = esp_timer_get_time();
        interp.Invoke();
        int64_t t2 = esp_timer_get_time();
        h32_post(&s_hmm, out->data.int8, &r.hr, &r.sqi); r.mon = (double)r.sqi >= 0.02896360121667385;
        int64_t t3 = esp_timer_get_time();
        r.nblk = (uint8_t)h32_rules_step(&s_rules, r.state, r.mon, r.hr, &r.blk);
        int64_t t4 = esp_timer_get_time();
        r.t_spec = (uint32_t)(t1 - t0); r.t_inf = (uint32_t)(t2 - t1b); r.t_post = (uint32_t)(t3 - t2);
        r.t_rules = (uint32_t)(t4 - t3); r.t_total = (uint32_t)((t4 - t0) - (t1b - t1));
        memcpy(r.logits, out->data.int8, 256);
        uint16_t rc = ck((const uint8_t *)&r + 4, sizeof r - 4);
        memcpy(s_last, &r, sizeof r); memcpy(s_last + sizeof r, &rc, 2); s_last_len = sizeof r + 2;
        wr(s_last, s_last_len); s_win++;
    }
}
