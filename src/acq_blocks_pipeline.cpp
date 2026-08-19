//==============================================================================
// acq_blocks_pipeline.cpp
//
// Refactorisation en 3 blocs synthetisables separement, demandee par Carlos
// (meme principe que le tutoriel RGB : plusieurs blocs interconnectes par
// flux au lieu d'un seul bloc qui fait tout).
//
//   BLOC 1 : dds_mixer_top()       -- melange DDS (Doppler x echantillons)
//   BLOC 2 : correlator_top()      -- corrélation signal x PRN (circuit fixe,
//                                      deja partage coarse/fine dans la
//                                      version precedente acq_unified.cpp)
//   BLOC 3 : decision_top()        -- decision finale, avec un switch
//                                      Option A (seuil de variance, comme
//                                      l'architecture originale) / Option B
//                                      (seuil fixe, version optimisee actuelle)
//
// Chaque bloc peut etre mis en "top" HLS et synthetise independamment pour
// obtenir son propre rapport BRAM/DSP/FF/LUT. Le wrapper acquisition_stsa_top
// les assemble ensuite avec #pragma HLS DATAFLOW, sur le meme principe que
// le tutoriel RGB mentionne par Carlos.
//==============================================================================
#include "acq_blocks.h"
#include "dds_lut_rom.h"
#ifndef __SYNTHESIS__
#include <iostream>
#include <cmath>
#endif

// ============================================================
// Constantes partagees (identiques a acq_unified.cpp)
// ============================================================
static const int STSA_DDS_LUT_BITS   = 10;
static const int STSA_DDS_PHASE_BITS = 32;
static const int STSA_FS_HZ          = 11999000;
static const int STSA_FC_HZ          = 3563000;
static const int PRN_VARIANTS        = 3;
static const int CORR_TILE           = 16;

#if defined(__SYNTHESIS__) && !defined(__INTELLISENSE__)
typedef ap_uint<1> prn_sign_t;
#else
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

// Table PRN partagee (identique au principe de acq_unified.cpp : ~6 BRAM
// au lieu des ~100 BRAM de prn_banks[3][16][2N] du design original)
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

//==============================================================================
// BLOC 1 -- dds_mixer_top()
//
// Melange DDS pour un balayage complet (coarse : NB_DOPPLER_COARSE bins x
// PRECISION passes STSA ; fine : NB_DOPPLER_FINE bins, tous les echantillons).
// Capture le signal brut une seule fois (premier appel), puis reutilise le
// buffer local pour tous les bins du balayage.
//==============================================================================
void dds_mixer_top(
    const sample_t signal[N],                // signal deja capture par le wrapper (1 seule lecture de rx_stream)
    sweep_mode_t mode,
    const doppler_t doppler_list[NB_DOPPLER_COARSE],  // taille max = coarse ; fine utilise les n_bins premiers
    int n_bins,
    hls::stream<mix_pkt_t> &mix_out
) {
#pragma HLS INLINE off

    MIXER_BIN_LOOP: for (int b = 0; b < n_bins; b++) {
#pragma HLS LOOP_TRIPCOUNT min=21 max=41
        doppler_t fd = doppler_list[b];
        dds_phase_t phase_inc = hz_to_phase_inc(fd);
        bool is_last_bin = (b == n_bins - 1);

        if (mode == SWEEP_COARSE) {
            // 8 passes STSA : indices n = pass, pass+PRECISION, pass+2*PRECISION, ...
            MIXER_COARSE_PASS: for (int pass = 0; pass < PRECISION; pass++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
                int M = (N - pass + PRECISION - 1) / PRECISION;
                dds_phase_u_t phase_acc  = (dds_phase_u_t)((long long)phase_inc * pass);
                dds_phase_u_t phase_step = (dds_phase_u_t)((long long)phase_inc * PRECISION);
                bool last_pass = (pass == PRECISION - 1);

                MIXER_COARSE_SAMPLE: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
                    int n = pass + i * PRECISION;
                    unsigned lut_idx = dds_lut_index(phase_acc);
                    trig_t c = DDS_COS_LUT[lut_idx];
                    trig_t s = DDS_SIN_LUT[lut_idx];
                    sample_t x = signal[n];

                    mix_pkt_t pkt;
                    pkt.mix_i      = x * c;
                    pkt.mix_q      = -(x * s);
                    pkt.idx        = n;
                    pkt.bin_id     = b;
                    pkt.bin_last   = last_pass && (i == M - 1);
                    pkt.sweep_last = is_last_bin && pkt.bin_last;
                    mix_out.write(pkt);

                    phase_acc = (dds_phase_u_t)(phase_acc + phase_step);
                }
            }
        } else {
            // SWEEP_FINE : tous les N echantillons, une seule passe
            dds_phase_u_t phase_acc = 0;
            MIXER_FINE_SAMPLE: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
                unsigned lut_idx = dds_lut_index(phase_acc);
                trig_t c = DDS_COS_LUT[lut_idx];
                trig_t s = DDS_SIN_LUT[lut_idx];
                sample_t x = signal[n];

                mix_pkt_t pkt;
                pkt.mix_i      = x * c;
                pkt.mix_q      = -(x * s);
                pkt.idx        = n;
                pkt.bin_id     = b;
                pkt.bin_last   = (n == N - 1);
                pkt.sweep_last = is_last_bin && pkt.bin_last;
                mix_out.write(pkt);

                phase_acc = (dds_phase_u_t)(phase_acc + (dds_phase_u_t)phase_inc);
            }
        }
    }
}

//==============================================================================
// BLOC 2 -- correlator_top()
//
// Corrélation signal x PRN. Consomme le flux du Bloc 1 un bin a la fois
// (bufferise localement), calcule la corrélation par tuiles de CORR_TILE
// phases, et emet un resultat par (bin, tau, variant). En mode fine, la
// meme portion bufferisee est reutilisee pour les 3 variantes PRN (0, +1/2,
// -1/2 chip) sans redemander le flux au Bloc 1.
//==============================================================================
void correlator_top(
    hls::stream<mix_pkt_t> &mix_in,
    sweep_mode_t mode,
    int tau_begin,
    int tau_count,
    hls::stream<corr_result_pkt_t> &corr_out
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_prn_var cyclic factor=16 dim=2

    sample_t buf_i[N];
    sample_t buf_q[N];
    int      buf_idx[N];
#pragma HLS BIND_STORAGE variable=buf_i type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=buf_q type=ram_1p impl=bram

    int variant_count = (mode == SWEEP_FINE) ? PRN_VARIANTS : 1;

    bool sweep_done = false;
    CORR_BIN_LOOP: while (!sweep_done) {
#pragma HLS LOOP_TRIPCOUNT min=21 max=41

        // --- Etape A : recevoir un bin complet du Bloc 1 et le bufferiser ---
        int count = 0;
        int cur_bin = -1;
        bool bin_done = false;
        CORR_BUFFER: while (!bin_done) {
#pragma HLS PIPELINE II=2
#pragma HLS LOOP_TRIPCOUNT min=1500 max=11999
            mix_pkt_t pkt = mix_in.read();
            buf_i[count]   = pkt.mix_i;
            buf_q[count]   = pkt.mix_q;
            buf_idx[count] = pkt.idx;
            cur_bin        = pkt.bin_id;
            count++;
            bin_done  = pkt.bin_last;
            sweep_done = pkt.sweep_last;
        }

        // --- Etape B : correler ce bin pour chaque variante demandee ---
        CORR_VARIANT_LOOP: for (int v = 0; v < variant_count; v++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
            CORR_TAU_TILE_LOOP: for (int t = 0; t < tau_count; t += CORR_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=64

                acc_t accI[CORR_TILE];
                acc_t accQ[CORR_TILE];
#pragma HLS ARRAY_PARTITION variable=accI complete dim=1
#pragma HLS ARRAY_PARTITION variable=accQ complete dim=1

                CORR_INIT_TILE: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS UNROLL
                    accI[k] = 0;
                    accQ[k] = 0;
                }

                CORR_SAMPLE_LOOP: for (int i = 0; i < count; i++) {
#pragma HLS PIPELINE II=1
                    sample_t bi = buf_i[i];
                    sample_t bq = buf_q[i];
                    int n_orig  = buf_idx[i];

                    CORR_TILE_LOOP: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS UNROLL
                        int tau = t + k + tau_begin;
                        if (tau < NB_PHASES) {
                            int prn_idx = g_tau_start_tbl[tau] + n_orig;
                            if (prn_idx >= N) prn_idx -= N;
                            prn_sign_t s = g_prn_var[v][prn_idx];
                            sample_t si = bi;
                            sample_t sq = bq;
                            if (!s) { si = -si; sq = -sq; }
                            accI[k] += si;
                            accQ[k] += sq;
                        }
                    }
                }

                CORR_WRITE_TILE: for (int k = 0; k < CORR_TILE; k++) {
#pragma HLS PIPELINE II=2
                    int tau = t + k + tau_begin;
                    if (tau < NB_PHASES) {
                        corr_result_pkt_t pkt;
                        pkt.power   = (peak_power_t)(accI[k] * accI[k] + accQ[k] * accQ[k]);
                        pkt.bin_id  = cur_bin;
                        pkt.tau     = tau;
                        pkt.variant = v;
                        // bin_last se declenche sur le DERNIER tau valide du bin
                        // (pas sur k==CORR_TILE-1, qui peut correspondre a un
                        // tau hors bornes si tau_count n'est pas multiple de
                        // CORR_TILE -- c'etait la cause du blocage initial).
                        pkt.bin_last   = (tau == tau_begin + tau_count - 1);
                        pkt.sweep_last = sweep_done && (v == variant_count - 1) && pkt.bin_last;
                        corr_out.write(pkt);
                    }
                }
            }
        }
    }
}

//==============================================================================
// BLOC 3 -- decision_top()
//
// Decision finale. Deux modes, memes entrees/sorties :
//   - DECISION_FIXED_THRESHOLD   (Option B, version optimisee actuelle) :
//       detecte des qu'un pic non nul existe.
//   - DECISION_VARIANCE_RELATIVE (Option A, comme l'architecture originale) :
//       conserve les DECISION_TOPK plus grands pics du balayage, calcule la
//       variance normalisee entre le plus grand et le plus petit d'entre eux
//       (formule identique a celle de la these de Glauberto, Eq. 8), et ne
//       detecte que si cette variance depasse le seuil fourni.
//==============================================================================
void decision_top(
    hls::stream<corr_result_pkt_t> &corr_in,
    decision_mode_t mode,
    metric_t threshold,
    int &best_bin_out,
    int &best_tau_out,
    peak_power_t &best_peak_out,
    bool &detected_out
) {
#pragma HLS INLINE off

    peak_power_t topk[DECISION_TOPK];
#pragma HLS ARRAY_PARTITION variable=topk complete dim=1
    DEC_INIT_TOPK: for (int k = 0; k < DECISION_TOPK; k++) {
#pragma HLS UNROLL
        topk[k] = 0;
    }

    peak_power_t global_best = 0;
    int global_best_bin = 0;
    int global_best_tau = 0;

    bool sweep_done = false;
    DEC_READ_LOOP: while (!sweep_done) {
#pragma HLS PIPELINE II=2
#pragma HLS LOOP_TRIPCOUNT min=1023 max=64449
        corr_result_pkt_t pkt = corr_in.read();

        if (pkt.power > global_best) {
            global_best     = pkt.power;
            global_best_bin = pkt.bin_id;
            global_best_tau = pkt.tau;
        }

        if (mode == DECISION_VARIANCE_RELATIVE) {
            // Insertion dans la liste des DECISION_TOPK plus grands pics,
            // en ne gardant que les valeurs qui depassent le plus petit
            // element courant (liste triee de facon decroissante).
            if (pkt.power > topk[DECISION_TOPK - 1]) {
                topk[DECISION_TOPK - 1] = pkt.power;
                DEC_SORT_TOPK: for (int k = DECISION_TOPK - 1; k > 0; k--) {
#pragma HLS UNROLL
                    if (topk[k] > topk[k - 1]) {
                        peak_power_t tmp = topk[k - 1];
                        topk[k - 1] = topk[k];
                        topk[k] = tmp;
                    }
                }
            }
        }

        sweep_done = pkt.sweep_last;
    }

    best_bin_out  = global_best_bin;
    best_tau_out  = global_best_tau;
    best_peak_out = global_best;

    if (mode == DECISION_FIXED_THRESHOLD) {
        // Option B : comportement actuel, inchange.
        detected_out = (global_best > (peak_power_t)0);
    } else {
        // Option A : variance normalisee entre le plus grand et le plus
        // petit des DECISION_TOPK pics (cf. these de Glauberto, Eq. 8).
        // Calcul en virgule fixe (metric_t), pas en double -- le double
        // genere des unites flottantes 64 bits tres couteuses en HLS
        // (latence ddiv=58 cycles, violation de timing constatee en synthese).
        metric_t p1 = (metric_t)topk[0];
        metric_t p2 = (metric_t)topk[DECISION_TOPK - 1];
        metric_t rho1 = (metric_t)1.0f;                      // p1 / p1
        metric_t rho2 = (p1 > (metric_t)0) ? (metric_t)(p2 / p1) : (metric_t)0;
        metric_t mean = (metric_t)((rho1 + rho2) / (metric_t)2.0f);
        metric_t diff1 = rho1 - mean;
        metric_t diff2 = rho2 - mean;
        metric_t var  = (metric_t)((metric_t)0.5f * (diff1 * diff1 + diff2 * diff2));
        detected_out = ((metric_t)var > threshold) && (global_best > (peak_power_t)0);
    }
}

//==============================================================================
// WRAPPER TOP-LEVEL -- assemble les 3 blocs (coarse puis fine)
//
// Meme interface AXI que l'ancien acquisition_stsa_top, pour compatibilite
// avec le notebook PYNQ existant. Le parametre decision_mode permet de
// basculer entre Option A et Option B sans toucher au reste du pipeline --
// c'est le point cle demande par Carlos pour la comparaison equitable.
//==============================================================================
void acquisition_pipeline_top(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    decision_mode_t decision_mode,
    metric_t threshold_coarse,
    metric_t threshold_fine
) {
#pragma HLS INTERFACE axis      port=rx_stream
#pragma HLS INTERFACE axis      port=prn_stream
#pragma HLS INTERFACE s_axilite port=doppler_out
#pragma HLS INTERFACE s_axilite port=phase_out
#pragma HLS INTERFACE s_axilite port=peak_out
#pragma HLS INTERFACE s_axilite port=detected_out
#pragma HLS INTERFACE s_axilite port=decision_mode
#pragma HLS INTERFACE s_axilite port=threshold_coarse
#pragma HLS INTERFACE s_axilite port=threshold_fine
#pragma HLS INTERFACE s_axilite port=return

    // --- Capture RX (une seule fois) et PRN, initialisation des tables ---
    sample_t signal[N];
    sample_t prn_raw[N];
#pragma HLS BIND_STORAGE variable=signal  type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=prn_raw type=ram_1p impl=bram
    CAPTURE_RX_PRN: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        axis_t rx_val  = rx_stream.read();
        axis_t prn_val = prn_stream.read();
        signal[i]   = (sample_t)rx_val.data;
        prn_raw[i]  = (sample_t)prn_val.data;
    }
    init_prn_var(prn_raw);
    init_tau_start_tbl();

    // ================= BALAYAGE COARSE =================
    doppler_t doppler_list_coarse[NB_DOPPLER_COARSE];
    BUILD_COARSE_LIST: for (int d = 0; d < NB_DOPPLER_COARSE; d++) {
#pragma HLS PIPELINE II=1
        int offset = (d - NB_DOPPLER_COARSE / 2) * DOPPLER_BIN_COARSE;
        doppler_list_coarse[d] = (doppler_t)offset;
    }

    hls::stream<mix_pkt_t> mix_s_coarse("mix_s_coarse");
    hls::stream<corr_result_pkt_t> corr_s_coarse("corr_s_coarse");
#pragma HLS STREAM variable=mix_s_coarse depth=64
#pragma HLS STREAM variable=corr_s_coarse depth=64
#pragma HLS DATAFLOW

    dds_mixer_top(signal, SWEEP_COARSE, doppler_list_coarse, NB_DOPPLER_COARSE, mix_s_coarse);
    correlator_top(mix_s_coarse, SWEEP_COARSE, 0, NB_PHASES, corr_s_coarse);

    int coarse_bin = 0, coarse_tau = 0;
    peak_power_t coarse_peak = 0;
    bool coarse_detected = false;
    decision_top(corr_s_coarse, decision_mode, threshold_coarse,
                 coarse_bin, coarse_tau, coarse_peak, coarse_detected);

    int coarse_doppler = (int)doppler_list_coarse[coarse_bin];

    doppler_out  = coarse_doppler;
    phase_out    = coarse_tau;
    peak_out     = (power_t)coarse_peak;
    detected_out = coarse_detected;

    if (!coarse_detected) {
        return;
    }

    // ================= BALAYAGE FINE =================
    doppler_t doppler_list_fine[NB_DOPPLER_COARSE]; // reutilise le meme tableau (taille max)
    BUILD_FINE_LIST: for (int d = 0; d < NB_DOPPLER_FINE; d++) {
#pragma HLS PIPELINE II=1
        int offset = coarse_doppler + (d - NB_DOPPLER_FINE / 2) * DOPPLER_BIN_FINE;
        doppler_list_fine[d] = (doppler_t)offset;
    }

    hls::stream<mix_pkt_t> mix_s_fine("mix_s_fine");
    hls::stream<corr_result_pkt_t> corr_s_fine("corr_s_fine");
#pragma HLS STREAM variable=mix_s_fine depth=64
#pragma HLS STREAM variable=corr_s_fine depth=64

    dds_mixer_top(signal, SWEEP_FINE, doppler_list_fine, NB_DOPPLER_FINE, mix_s_fine);
    correlator_top(mix_s_fine, SWEEP_FINE, 0, NB_PHASES, corr_s_fine);

    int fine_bin = 0, fine_tau = 0;
    peak_power_t fine_peak = 0;
    bool fine_detected = false;
    decision_top(corr_s_fine, decision_mode, threshold_fine,
                 fine_bin, fine_tau, fine_peak, fine_detected);

    if (fine_detected) {
        doppler_out  = (int)doppler_list_fine[fine_bin];
        phase_out    = fine_tau;
        peak_out     = (power_t)fine_peak;
        detected_out = fine_detected;
    }
}
