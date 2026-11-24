/* H32 T6 portable core (C99, float32): pre-processing, post-processing (softmax, SQI, HMM), ACC state, and the
 * streaming block machinery with rules K and R. Compiled into the ESP-IDF firmware and, unchanged, into a host
 * shared library for checks against the Python pipeline (05_experiments/t6_mcu/host_check.py).
 * The SpecFCN itself (INT8) runs in TFLite Micro on the board; the core only quantises its input and reads its
 * output. Mirrors 04_code/h32/features.py, e2cnn.py, method.py and confirm.py (rules K, R). */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H32_NF 256          /* frequency bins of the model input */
#define H32_NCH 4           /* PPG, ACC x, y, z */
#define H32_NSTATE 3

/* features.spectra for one window: ppg[512] at 64 Hz, acc[256*3] at 32 Hz interleaved (x, y, z).
 * out: channels-last [256][4] (Keras layout), normalised as in features.py. */
void h32_spectra(const float *ppg, const float *acc, float *out);
/* model input quantisation: q = clip(round(x / s) + zp) */
void h32_quant_input(const float *spec, int8_t *q);
/* method.states for one window */
int h32_acc_state(const float *acc);

typedef struct { float belief[256]; int n; } h32_hmm_t;
void h32_hmm_init(h32_hmm_t *h);
/* logits: int8 model output [256]; returns HR (bpm) and SQI = max softmax over the 194 masked bins */
void h32_post(h32_hmm_t *h, const int8_t *logits, float *hr, float *sqi);
/* same, from float logits over the 194 masked bins (host checks) */
void h32_post_float(h32_hmm_t *h, const float *logit194, float *hr, float *sqi);

typedef struct {
    uint8_t state; uint16_t j; float S; float tauK; float tauR;
    uint8_t hasK, decK, hasR, decR;
} h32_block_t;

typedef struct {
    int32_t clock[H32_NSTATE];          /* monitored windows per state */
    float cur[H32_NSTATE];              /* running block max of the open block */
    float first[H32_NSTATE][8];         /* S of blocks 0..5 (K calibration) */
    float ring[H32_NSTATE][8];          /* S of the last 8 blocks (R window j-7..j-2) */
    float tauK[H32_NSTATE]; uint8_t hasTauK[H32_NSTATE];
    /* last two windows, for the debounce D_i = min(h_i, h_{i-1}, h_{i-2}) */
    int n; int8_t pst[2]; int8_t pmon[2]; int32_t pblk[2]; float phr[2];
} h32_rules_t;

void h32_rules_init(h32_rules_t *r);
/* one window (in time order). Returns 1 and fills *out if a block of some state was completed by this window. */
int h32_rules_step(h32_rules_t *r, int state, int monitored, float hr, h32_block_t *out);

#ifdef __cplusplus
}
#endif
