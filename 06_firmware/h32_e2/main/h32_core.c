/* H32 T6 portable core, see h32_core.h. All arithmetic float32 (PLAN_T6 §1); tables from h32_consts.h. */
#include "h32_core.h"
#include "h32_consts.h"
#include <math.h>
#include <string.h>

#define NEG_INF (-INFINITY)

/* ------------------------------------------------------------------ FFT (radix-2, in place, float32) */
static float s_cos[1024], s_sin[1024];   /* e^{-2 pi i k / 2048}, k < 1024 */
static int s_tw_ready = 0;
static float s_re[2048], s_im[2048];
static float s_ext[512 + 2 * 27], s_tmp[512 + 2 * 27];
static float s_mag[129];

static void tw_init(void) {
    if (s_tw_ready) return;
    for (int k = 0; k < 1024; k++) {
        double a = -2.0 * 3.14159265358979323846 * (double)k / 2048.0;
        s_cos[k] = (float)cos(a); s_sin[k] = (float)sin(a);
    }
    s_tw_ready = 1;
}

static void fft(float *re, float *im, int n) {        /* n = 1024 or 2048 */
    int stride0 = 2048 / n;
    for (int i = 1, j = 0; i < n; i++) {               /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = (n / len) * stride0;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = s_sin[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + k + half] = re[i + k] - tr; im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr; im[i + k] += ti;
            }
        }
    }
}

/* ------------------------------------------------------------------ scipy detrend / sosfilt / sosfiltfilt */
static void detrend_linear(float *x, int n) {
    float tm = 0.5f * (float)(n - 1), xm = 0.f, sxy = 0.f, sxx = 0.f;
    for (int i = 0; i < n; i++) xm += x[i];
    xm /= (float)n;
    for (int i = 0; i < n; i++) { float t = (float)i - tm; sxy += t * (x[i] - xm); sxx += t * t; }
    float b = sxy / sxx;
    for (int i = 0; i < n; i++) x[i] -= xm + b * ((float)i - tm);
}

/* direct form II transposed, as scipy.signal.sosfilt; zi scaled by x0 */
static void sosfilt(const double *sos, const double *zi, int nsec, float *x, int n, float x0) {
    for (int s = 0; s < nsec; s++) {
        const double *c = sos + 6 * s;
        float b0 = (float)c[0], b1 = (float)c[1], b2 = (float)c[2], a1 = (float)c[4], a2 = (float)c[5];
        float z0 = (float)zi[2 * s] * x0, z1 = (float)zi[2 * s + 1] * x0;
        for (int i = 0; i < n; i++) {
            float xi = x[i], y = b0 * xi + z0;
            z0 = b1 * xi - a1 * y + z1; z1 = b2 * xi - a2 * y; x[i] = y;
        }
    }
}

static void sosfiltfilt(const double *sos, const double *zi, int nsec, int edge, float *x, int n) {
    int m = n + 2 * edge;                                     /* odd extension (scipy odd_ext) */
    for (int i = 0; i < edge; i++) s_ext[i] = 2.f * x[0] - x[edge - i];
    memcpy(s_ext + edge, x, sizeof(float) * n);
    for (int i = 0; i < edge; i++) s_ext[edge + n + i] = 2.f * x[n - 1] - x[n - 2 - i];
    sosfilt(sos, zi, nsec, s_ext, m, s_ext[0]);
    for (int i = 0; i < m; i++) s_tmp[i] = s_ext[m - 1 - i];
    sosfilt(sos, zi, nsec, s_tmp, m, s_tmp[0]);
    for (int i = 0; i < n; i++) x[i] = s_tmp[m - 1 - (edge + i)];
}

/* one channel: detrend, band-pass (filtfilt), Hann, |rfft|, interpolate onto the 256 model frequencies */
static void spec1(float *w, int n, int nfft, const double *sos, const double *zi, int nsec, int edge,
                  const int16_t *iidx, const float *it, float *out, int out_stride) {
    detrend_linear(w, n);
    sosfiltfilt(sos, zi, nsec, edge, w, n);
    for (int i = 0; i < nfft; i++) { s_re[i] = 0.f; s_im[i] = 0.f; }
    for (int i = 0; i < n; i++) {
        float h = 0.5f - 0.5f * cosf(2.f * 3.14159265f * (float)i / (float)(n - 1));
        s_re[i] = w[i] * h;
    }
    fft(s_re, s_im, nfft);
    for (int k = 0; k < 129; k++) s_mag[k] = sqrtf(s_re[k] * s_re[k] + s_im[k] * s_im[k]);
    for (int f = 0; f < H32_NF; f++) {
        int i = iidx[f]; float t = it[f];
        out[f * out_stride] = s_mag[i - 1] * (1.f - t) + s_mag[i] * t;
    }
}

void h32_spectra(const float *ppg, const float *acc, float *out) {
    static float w[512];
    tw_init();
    memcpy(w, ppg, sizeof(float) * 512);
    spec1(w, 512, H32_PPG_NFFT, H32_PPG_SOS, H32_PPG_ZI, H32_PPG_NSEC, H32_PPG_EDGE, H32_PPG_IIDX, H32_PPG_IT, out + 0, H32_NCH);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 256; i++) w[i] = acc[3 * i + c];
        spec1(w, 256, H32_ACC_NFFT, H32_ACC_SOS, H32_ACC_ZI, H32_ACC_NSEC, H32_ACC_EDGE, H32_ACC_IIDX, H32_ACC_IT, out + 1 + c, H32_NCH);
    }
    float mp = 0.f, ma = 0.f;
    for (int f = 0; f < H32_NF; f++) {
        if (out[f * 4] > mp) mp = out[f * 4];
        for (int c = 1; c < 4; c++) if (out[f * 4 + c] > ma) ma = out[f * 4 + c];
    }
    mp += 1e-9f; ma += 1e-9f;
    for (int f = 0; f < H32_NF; f++) {
        out[f * 4] /= mp;
        for (int c = 1; c < 4; c++) out[f * 4 + c] /= ma;
    }
}

void h32_quant_input(const float *spec, int8_t *q) {
    const float inv = (float)(1.0 / H32_IN_SCALE);
    for (int i = 0; i < H32_NF * H32_NCH; i++) {
        long v = lrintf(spec[i] * inv) + H32_IN_ZP;           /* round half to even, as numpy.round */
        q[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
}

int h32_acc_state(const float *acc) {
    float m = 0.f, s2 = 0.f, nrm[256];
    for (int i = 0; i < 256; i++) {
        nrm[i] = sqrtf(acc[3 * i] * acc[3 * i] + acc[3 * i + 1] * acc[3 * i + 1] + acc[3 * i + 2] * acc[3 * i + 2]);
        m += nrm[i];
    }
    m /= 256.f;
    for (int i = 0; i < 256; i++) { float d = nrm[i] - m; s2 += d * d; }
    float sd = sqrtf(s2 / 256.f);                               /* np.std, ddof 0 */
    return sd < (float)H32_EDGE0 ? 0 : (sd < (float)H32_EDGE1 ? 1 : 2);   /* np.digitize */
}

/* ------------------------------------------------------------------ softmax, SQI, HMM (e2cnn.hmm_track_grid) */
void h32_hmm_init(h32_hmm_t *h) {
    for (int i = 0; i < H32_NB; i++) h->belief[i] = 1.f / (float)H32_NB;
    h->n = 0;
}

void h32_post_float(h32_hmm_t *h, const float *lg, float *hr, float *sqi) {
    static float p[H32_NB], prior[H32_NB];
    float mx = lg[0];
    for (int i = 1; i < H32_NB; i++) if (lg[i] > mx) mx = lg[i];
    float sum = 0.f;
    for (int i = 0; i < H32_NB; i++) { p[i] = expf(lg[i] - mx); sum += p[i]; }
    float pm = 0.f;
    for (int i = 0; i < H32_NB; i++) { p[i] /= sum; if (p[i] > pm) pm = p[i]; }
    *sqi = pm;
    for (int i = 0; i < H32_NB; i++) {                          /* np.convolve(belief, k, 'same') */
        float a = 0.f;
        int j0 = i - H32_HMM_HALF < 0 ? 0 : i - H32_HMM_HALF, j1 = i + H32_HMM_HALF > H32_NB - 1 ? H32_NB - 1 : i + H32_HMM_HALF;
        for (int j = j0; j <= j1; j++) a += h->belief[j] * H32_HMM_K[i - j + H32_HMM_HALF];
        prior[i] = a + H32_HMM_FLOOR / (float)H32_NB;
    }
    float s = 0.f;
    for (int i = 0; i < H32_NB; i++) { prior[i] *= (p[i] + H32_P_EPS); s += prior[i]; }
    int am = 0; float bm = -1.f;
    for (int i = 0; i < H32_NB; i++) { h->belief[i] = prior[i] / s; if (h->belief[i] > bm) { bm = h->belief[i]; am = i; } }
    *hr = H32_GRID[am]; h->n++;
}

void h32_post(h32_hmm_t *h, const int8_t *logits, float *hr, float *sqi) {
    static float lg[H32_NB];
    for (int i = 0; i < H32_NB; i++) lg[i] = ((float)logits[H32_MASK_IDX[i]] - (float)H32_OUT_ZP) * H32_OUT_SCALE;
    h32_post_float(h, lg, hr, sqi);
}

/* ------------------------------------------------------------------ blocks, debounce, rules K and R */
void h32_rules_init(h32_rules_t *r) {
    memset(r, 0, sizeof(*r));
    for (int k = 0; k < H32_NSTATE; k++) {
        r->cur[k] = NEG_INF;
        for (int j = 0; j < 8; j++) { r->first[k][j] = NEG_INF; r->ring[k][j] = NEG_INF; }
    }
}

int h32_rules_step(h32_rules_t *r, int st, int mon, float hr, h32_block_t *out) {
    int32_t blk = -1;
    if (mon) { blk = r->clock[st] / H32_LW; r->clock[st]++; }
    float D = NEG_INF;                                          /* method.debounce */
    if (r->n >= 2 && mon && r->pmon[0] && r->pmon[1] && r->pst[0] == st && r->pst[1] == st && r->pblk[0] == blk && r->pblk[1] == blk) {
        D = hr; if (r->phr[0] < D) D = r->phr[0]; if (r->phr[1] < D) D = r->phr[1];
    }
    r->pst[1] = r->pst[0]; r->pmon[1] = r->pmon[0]; r->pblk[1] = r->pblk[0]; r->phr[1] = r->phr[0];
    r->pst[0] = (int8_t)st; r->pmon[0] = (int8_t)mon; r->pblk[0] = blk; r->phr[0] = hr; r->n++;
    if (!mon) return 0;
    if (D > r->cur[st]) r->cur[st] = D;                         /* method.block_scores: block maximum */
    if (r->clock[st] % H32_LW != 0) return 0;
    int j = r->clock[st] / H32_LW - 1;                          /* block j of state st is complete */
    float S = r->cur[st]; r->cur[st] = NEG_INF;
    memset(out, 0, sizeof(*out));
    out->state = (uint8_t)st; out->j = (uint16_t)j; out->S = S; out->tauK = NAN; out->tauR = NAN;
    if (j < H32_NCAL) r->first[st][j] = S;
    if (j == H32_NCAL - 1) {                                    /* confirm.tau_K: max of blocks 0..5 */
        float t = NEG_INF; for (int i = 0; i < H32_NCAL; i++) if (r->first[st][i] > t) t = r->first[st][i];
        r->tauK[st] = t; r->hasTauK[st] = 1;
    }
    if (j >= H32_K_FIRST && r->hasTauK[st]) { out->hasK = 1; out->tauK = r->tauK[st]; out->decK = S > r->tauK[st]; }
    if (j >= H32_R_FIRST) {                                     /* confirm.tau_win, lags 2..7 */
        float t = NEG_INF; for (int l = 2; l <= 7; l++) { float v = r->ring[st][(j - l) % 8]; if (v > t) t = v; }
        out->hasR = 1; out->tauR = t; out->decR = S > t;
    }
    r->ring[st][j % 8] = S;
    return 1;
}
