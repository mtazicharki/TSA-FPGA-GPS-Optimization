//==============================================================================
// acq_unified.cpp
//==============================================================================
#include "acq_stsaV2.h"
#include "dds_lut_rom.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

static const int STSA_DDS_LUT_BITS   = 10;
static const int STSA_DDS_PHASE_BITS = 32;
static const int STSA_FS_HZ          = 11999000;
static const int STSA_FC_HZ          = 3563000;
static const int PRN_VARIANTS        = 3;
static const int CORR_TILE           = 4;
static const int COARSE_TOPK         = 16;

#if defined(__SYNTHESIS__) && !defined(__INTELLISENSE__)
typedef ap_uint<2> variant_t;
typedef ap_uint<1> prn_sign_t;
#else
typedef unsigned char variant_t;
typedef unsigned char prn_sign_t;
#endif

#ifdef __SYNTHESIS__
typedef ap_uint<STSA_DDS_PHASE_BITS> dds_phase_u_t;
typedef ap_int<STSA_DDS_PHASE_BITS>  dds_phase_t;
static inline ap_uint<STSA_DDS_LUT_BITS> dds_lut_index(dds_phase_u_t phase_acc) {
#pragma HLS INLINE
    return phase_acc.range(STSA_DDS_PHASE_BITS - 1, STSA_DDS_PHASE_BITS - STSA_DDS_LUT_BITS);
}
#else
typedef uint32_t dds_phase_u_t;
typedef int32_t  dds_phase_t;
static inline unsigned dds_lut_index(dds_phase_u_t phase_acc) {
    return phase_acc >> (STSA_DDS_PHASE_BITS - STSA_DDS_LUT_BITS);
}
#endif

static inline dds_phase_t hz_to_phase_inc(doppler_t fd_hz) {
#pragma HLS INLINE
    const int freq_total_hz = -(STSA_FC_HZ + (int)fd_hz);
    long long num = ((long long)freq_total_hz << STSA_DDS_PHASE_BITS);
    return (dds_phase_t)(num / STSA_FS_HZ);
}

static inline prn_sign_t to_prn_sign(sample_t x) {
#pragma HLS INLINE
    return (x >= (sample_t)0) ? (prn_sign_t)1 : (prn_sign_t)0;
}

static prn_sign_t g_prn_var[PRN_VARIANTS][2 * N];
static int g_tau_start_tbl[NB_PHASES];

static void init_prn_var(const sample_t prn[N]) {
#pragma HLS INLINE off
    INIT_BASE: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        int next_idx = (i + 1 < N) ? i + 1 : 0;
        int prev_idx = (i > 0) ? i - 1 : N - 1;
        sample_t base       = prn[i];
        sample_t half_plus  = (sample_t)0.5f * (prn[i] + prn[next_idx]);
        sample_t half_minus = (sample_t)0.5f * (prn[i] + prn[prev_idx]);
        g_prn_var[0][i]     = to_prn_sign(base);
        g_prn_var[1][i]     = to_prn_sign(half_plus);
        g_prn_var[2][i]     = to_prn_sign(half_minus);
    }
    DUP_BASE: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        g_prn_var[0][i + N] = g_prn_var[0][i];
        g_prn_var[1][i + N] = g_prn_var[1][i];
        g_prn_var[2][i + N] = g_prn_var[2][i];
    }
}

static void init_tau_start_tbl() {
#pragma HLS INLINE off
    INIT_TAU: for (int tau = 0; tau < NB_PHASES; tau++) {
#pragma HLS PIPELINE II=1
        g_tau_start_tbl[tau] = (int)(((long long)tau * (long long)N) / NB_PHASES);
    }
}

struct corr_pkt_unified_t {
    peak_power_t power;
    int tau;
    int variant;
};

static void unified_corr_one_doppler(
    const sample_t mix_i[N],
    const sample_t mix_q[N],
    int n_samples,
    const int mix_indices[N],
    int tau_begin,
    int tau_count,
    int variant,
    doppler_t fd,
    acc_t accI_buf[NB_PHASES],
    acc_t accQ_buf[NB_PHASES],
    bool accumulate,
    hls::stream<corr_pkt_unified_t> &pow_s
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_prn_var cyclic factor=16 dim=2

    UNIFIED_TAU_TILE_LOOP: for (int t = 0; t < tau_count; t += CORR_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=256 avg=128

        acc_t accI[CORR_TILE];
        acc_t accQ[CORR_TILE];
#pragma HLS ARRAY_PARTITION variable=accI complete dim=1
#pragma HLS ARRAY_PARTITION variable=accQ complete dim=1

        INIT_TILE: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS UNROLL
            int tau = t + k + tau_begin;
            if (accumulate && tau < NB_PHASES) {
                accI[k] = accI_buf[tau];
                accQ[k] = accQ_buf[tau];
            } else {
                accI[k] = 0;
                accQ[k] = 0;
            }
        }

        UNIFIED_SAMPLE_LOOP: for (int i = 0; i < n_samples; i++) {
#pragma HLS PIPELINE II=1
            sample_t bi = mix_i[i];
            sample_t bq = mix_q[i];
            int n_orig  = mix_indices[i];

            UNIFIED_TILE_LOOP: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS UNROLL
                int tau = t + k + tau_begin;
                if (tau < NB_PHASES) {
                    int prn_idx = g_tau_start_tbl[tau] + n_orig;
                    if (prn_idx >= N) prn_idx -= N;
                    prn_sign_t s = g_prn_var[variant][prn_idx];
                    sample_t si  = bi;
                    sample_t sq  = bq;
                    if (!s) { si = -si; sq = -sq; }
                    accI[k] += si;
                    accQ[k] += sq;
                }
            }
        }

        WRITE_TILE: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS PIPELINE II=1
            int tau = t + k + tau_begin;
            if (tau < NB_PHASES) {
                if (accumulate) {
                    accI_buf[tau] = accI[k];
                    accQ_buf[tau] = accQ[k];
                }
                corr_pkt_unified_t pkt;
                pkt.power   = (peak_power_t)(accI[k] * accI[k] + accQ[k] * accQ[k]);
                pkt.tau     = tau;
                pkt.variant = variant;
                pow_s.write(pkt);
            }
        }
    }
}

static void coarse_pilot(
    const sample_t signal[N],
    const sample_t prn_raw[N],
    const int stsa_indices[PRECISION][N / PRECISION + 1],
    const int stsa_counts[PRECISION],
    const int shifts[NB_PHASES],
    const doppler_t doppler_offsets[NB_DOPPLER_COARSE],
    int &best_doppler_out,
    int &best_phase_out,
    peak_power_t &best_peak_out,
    metric_t &var_out,
    bool &coarse_detected
) {
#pragma HLS INLINE off

    acc_t accI_d[NB_PHASES];
    acc_t accQ_d[NB_PHASES];
#pragma HLS BIND_STORAGE variable=accI_d type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=accQ_d type=ram_2p impl=bram

    sample_t mix_i[N / PRECISION + 1];
    sample_t mix_q[N / PRECISION + 1];
    int      mix_n[N / PRECISION + 1];
#pragma HLS BIND_STORAGE variable=mix_i type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=mix_q type=ram_1p impl=bram

    peak_power_t global_best = 0;
    int global_best_d = 0;
    int global_best_tau = 0;

    metric_t sum_var = 0;
    int n_var_samples = 0;

    hls::stream<corr_pkt_unified_t> pow_s;
#pragma HLS STREAM variable=pow_s depth=256

    COARSE_DOPPLER_PILOT: for (int d = 0; d < NB_DOPPLER_COARSE; d++) {
#pragma HLS LOOP_TRIPCOUNT min=41 max=41
        doppler_t fd = doppler_offsets[d];
        dds_phase_t phase_inc = hz_to_phase_inc(fd);

        INIT_ACC: for (int t = 0; t < NB_PHASES; t++) {
#pragma HLS PIPELINE II=1
            accI_d[t] = 0;
            accQ_d[t] = 0;
        }

        COARSE_PASS_PILOT: for (int pass = 0; pass < PRECISION; pass++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int M = stsa_counts[pass];
            dds_phase_u_t phase_acc  = (dds_phase_u_t)((long long)phase_inc * pass);
            dds_phase_u_t phase_step = (dds_phase_u_t)((long long)phase_inc * PRECISION);

            MIX_COARSE_PILOT: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
                int n = stsa_indices[pass][i];
                unsigned lut_idx = dds_lut_index(phase_acc);
                trig_t c = DDS_COS_LUT[lut_idx];
                trig_t s = DDS_SIN_LUT[lut_idx];
                sample_t x = signal[n];
                mix_i[i] = x * c;
                mix_q[i] = -(x * s);
                mix_n[i] = n;
                phase_acc = (dds_phase_u_t)(phase_acc + phase_step);
            }

            unified_corr_one_doppler(
                mix_i, mix_q, M, mix_n,
                0, NB_PHASES,
                0,
                fd,
                accI_d, accQ_d,
                true,
                pow_s
            );

            if (pass == PRECISION - 1) {
                DRAIN_COARSE: for (int t = 0; t < NB_PHASES; t++) {
#pragma HLS PIPELINE II=1
                    corr_pkt_unified_t pkt = pow_s.read();
                    if (pkt.power > global_best) {
                        global_best     = pkt.power;
                        global_best_d   = d;
                        global_best_tau = pkt.tau;
                    }
                }
            } else {
                DRAIN_INTER: for (int t = 0; t < NB_PHASES; t++) {
#pragma HLS PIPELINE II=1
                    (void)pow_s.read();
                }
            }
        }
    }

    best_doppler_out = (int)doppler_offsets[global_best_d];
    best_phase_out   = global_best_tau;
    best_peak_out    = global_best;
    var_out          = (metric_t)0;
    coarse_detected  = (global_best > (peak_power_t)0);
}

static void fine_pilot(
    const sample_t signal[N],
    int coarse_doppler,
    int coarse_phase,
    const doppler_t doppler_offsets_fine[NB_DOPPLER_FINE],
    int fine_doppler_count,
    int fine_tau_begin,
    int fine_tau_count,
    int &best_doppler_out,
    int &best_phase_out,
    peak_power_t &best_peak_out,
    bool &fine_detected,
    hls::stream<axis_t> &corr_out,
    int &max_power_out,
    int &mean_power_out
) {
#pragma HLS INLINE off

    sample_t mix_i[N];
    sample_t mix_q[N];
    int      mix_n[N];
#pragma HLS BIND_STORAGE variable=mix_i type=ram_1p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=mix_q type=ram_1p impl=bram latency=2

    acc_t accI_dummy[NB_PHASES];
    acc_t accQ_dummy[NB_PHASES];

    hls::stream<corr_pkt_unified_t> pow_s;
#pragma HLS STREAM variable=pow_s depth=256

    peak_power_t global_best = 0;
    int global_best_d = 0;
    int global_best_tau = 0;
    int global_best_var = 0;
    peak_power_t sum_power = 0;
    int n_pkts = 0;

    FINE_DOPPLER_PILOT: for (int d = 0; d < fine_doppler_count; d++) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=21
        doppler_t fd = doppler_offsets_fine[d];
        dds_phase_t phase_inc = hz_to_phase_inc(fd);
        dds_phase_u_t phase_acc = 0;

        MIX_FINE_PILOT: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
            unsigned lut_idx = dds_lut_index(phase_acc);
            trig_t c = DDS_COS_LUT[lut_idx];
            trig_t s = DDS_SIN_LUT[lut_idx];
            sample_t x = signal[n];
            mix_i[n] = x * c;
            mix_q[n] = -(x * s);
            mix_n[n] = n;
            phase_acc = (dds_phase_u_t)(phase_acc + (dds_phase_u_t)phase_inc);
        }

        FINE_VARIANT_PILOT: for (int v = 0; v < PRN_VARIANTS; v++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
            unified_corr_one_doppler(
                mix_i, mix_q, N, mix_n,
                fine_tau_begin, fine_tau_count,
                v,
                fd,
                accI_dummy, accQ_dummy,
                false,
                pow_s
            );

            DRAIN_FINE: for (int t = 0; t < fine_tau_count; t++) {
#pragma HLS PIPELINE II=1
                corr_pkt_unified_t pkt = pow_s.read();
                sum_power += pkt.power;
                n_pkts++;
                if (pkt.power > global_best) {
                    global_best     = pkt.power;
                    global_best_d   = d;
                    global_best_tau = pkt.tau;
                    global_best_var = pkt.variant;
                }
                axis_t out_pkt;
                out_pkt.data = (int32_t)((double)pkt.power);
                out_pkt.keep = -1;
                out_pkt.strb = -1;
                out_pkt.last = (d == fine_doppler_count - 1) &&
                               (v == PRN_VARIANTS - 1) &&
                               (t == fine_tau_count - 1);
                corr_out.write(out_pkt);
            }
        }
    }

    best_doppler_out = (int)doppler_offsets_fine[global_best_d];
    best_phase_out   = global_best_tau;
    best_peak_out    = global_best;
    fine_detected    = (global_best > (peak_power_t)0);
    max_power_out    = (int)(double)global_best;
    mean_power_out   = (n_pkts > 0) ? (int)((double)sum_power / n_pkts) : 0;
}

static void acquisition_unified(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    hls::stream<axis_t> &corr_out,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    metric_t threshold_coarse,
    metric_t threshold_fine,
    int &max_power_out,
    int &mean_power_out
) {
#pragma HLS INLINE off

    sample_t signal[N];
    sample_t prn_raw[N];
#pragma HLS BIND_STORAGE variable=signal  type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=prn_raw type=ram_1p impl=bram

    CAPTURE_RX: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        axis_t rx_val  = rx_stream.read();
        axis_t prn_val = prn_stream.read();
        signal[i]  = (sample_t)rx_val.data;
        prn_raw[i] = (sample_t)prn_val.data;
    }


    init_prn_var(prn_raw);
    init_tau_start_tbl();

    int stsa_indices[PRECISION][N / PRECISION + 1];
    int stsa_counts[PRECISION];
    int shifts[NB_PHASES];
    doppler_t doppler_offsets_coarse[NB_DOPPLER_COARSE];
#pragma HLS BIND_STORAGE variable=stsa_indices type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=shifts       type=ram_1p impl=bram

    BUILD_STSA: for (int pass = 0; pass < PRECISION; pass++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        int count = 0;
        for (int n = pass; n < N; n += PRECISION) {
            stsa_indices[pass][count++] = n;
        }
        stsa_counts[pass] = count;
    }
    BUILD_SHIFTS: for (int tau = 0; tau < NB_PHASES; tau++) {
#pragma HLS PIPELINE II=1
        shifts[tau] = g_tau_start_tbl[tau];
    }
    BUILD_COARSE_DOPPLER: for (int d = 0; d < NB_DOPPLER_COARSE; d++) {
#pragma HLS PIPELINE II=1
        int offset = (d - NB_DOPPLER_COARSE / 2) * DOPPLER_BIN_COARSE;
        doppler_offsets_coarse[d] = (doppler_t)offset;
    }

    int coarse_doppler = 0, coarse_phase = 0;
    peak_power_t coarse_peak = 0;
    metric_t coarse_var = 0;
    bool coarse_det = false;

    coarse_pilot(
        signal, prn_raw,
        stsa_indices, stsa_counts, shifts,
        doppler_offsets_coarse,
        coarse_doppler, coarse_phase, coarse_peak,
        coarse_var, coarse_det
    );

    doppler_out = coarse_doppler;
    phase_out   = coarse_phase;
    peak_out    = (power_t)coarse_peak;
    detected_out = coarse_det;

    if (!coarse_det) {
        max_power_out  = 0;
        mean_power_out = 0;
        return;
    }

    doppler_t doppler_offsets_fine[NB_DOPPLER_FINE];
    int fine_tau_begin = 0, fine_tau_count = NB_PHASES;
    int fine_doppler_count = NB_DOPPLER_FINE;

    BUILD_FINE_DOPPLER: for (int d = 0; d < NB_DOPPLER_FINE; d++) {
#pragma HLS PIPELINE II=1
        int offset = coarse_doppler +
                     (d - NB_DOPPLER_FINE / 2) * DOPPLER_BIN_FINE;
        doppler_offsets_fine[d] = (doppler_t)offset;
    }

    int fine_doppler = 0, fine_phase = 0;
    peak_power_t fine_peak = 0;
    bool fine_det = false;

    fine_pilot(
        signal,
        coarse_doppler, coarse_phase,
        doppler_offsets_fine,
        fine_doppler_count,
        fine_tau_begin, fine_tau_count,
        fine_doppler, fine_phase, fine_peak,
        fine_det, corr_out,
        max_power_out, mean_power_out
    );

    if (fine_det) {
        doppler_out  = fine_doppler;
        phase_out    = fine_phase;
        peak_out     = (power_t)fine_peak;
        detected_out = fine_det;
    }
}

void acquisition_stsa_top(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    hls::stream<axis_t> &corr_out,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    int &max_power_out,
    int &mean_power_out
) {
#pragma HLS INTERFACE axis         port=rx_stream
#pragma HLS INTERFACE axis         port=prn_stream
#pragma HLS INTERFACE axis         port=corr_out
#pragma HLS INTERFACE s_axilite    port=doppler_out
#pragma HLS INTERFACE s_axilite    port=phase_out
#pragma HLS INTERFACE s_axilite    port=peak_out
#pragma HLS INTERFACE s_axilite    port=detected_out
#pragma HLS INTERFACE s_axilite    port=max_power_out
#pragma HLS INTERFACE s_axilite    port=mean_power_out
#pragma HLS INTERFACE s_axilite    port=return

    power_t    peak_internal  = 0;
    bool       detected_internal = false;
    metric_t   threshold_coarse = (metric_t)SEUIL_VARIANCE_COARSE;
    metric_t   threshold_fine   = (metric_t)SEUIL_VARIANCE_FINE;

    acquisition_unified(
        rx_stream, prn_stream, corr_out,
        doppler_out, phase_out,
        peak_internal, detected_internal,
        threshold_coarse, threshold_fine,
        max_power_out, mean_power_out
    );

    peak_out     = peak_internal;
    detected_out = detected_internal;
}
