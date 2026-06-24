//==============================================================================
// acq_stsaV2.h
// Interface publique du pipeline STSA d'acquisition GNSS (coarse + fine).
// Ce fichier centralise:
//  - les types fixes/flottants selon le mode (synthèse vs simulation),
//  - les constantes algorithmiques,
//  - l'API top-level utilisée par le testbench et Vitis HLS.
//==============================================================================
#ifndef ACQ_STSA_H
#define ACQ_STSA_H

// CSIM (hors __SYNTHESIS__) : choisir UN mode ci-dessous.
//
// Mode A — Virgule fixe, aligne sur synthese / bitstream (recommandé apres diagnostic) :
//   Decommenter : #define STSA_CSIM_FIXED
//
// Mode B — Diagnostic float (proche CPU LUT pour isoler ecarts coarse_var / peak) :
//   Laisser STSA_CSIM_FIXED commente (etat actuel de cette copie pour LABO01).
//
// La synthese Vitis ignore ce bloc : __SYNTHESIS__ force toujours ap_fixed.
#ifndef __SYNTHESIS__
#define STSA_CSIM_FIXED
#endif

// [BRANCHEMENT TEST CARLOS/TAZI] Switch fine search streaming :
// version actuelle (signal[N] stocke en BRAM, defaut) vs version streaming
// (signal jamais stocke dans un buffer, rx_stream relu directement dans
// fine_mix_to_mem a chaque bin Doppler).
// Decommenter UNIQUEMENT pour la synthese de mesure comparative
// (rapport BRAM/DSP/Fmax vs version buffer), puis recommenter avant
// tout flash FPGA.
 #define STSA_USE_FINE_STREAMING

// ========== INCLUSIONS SYSTEME ==========
#if (defined(__SYNTHESIS__) && !defined(__INTELLISENSE__)) || defined(STSA_CSIM_FIXED)
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>  // Indispensable pour axis_t
#endif

// ========== CONSTANTES ==========
// Paramètres globaux de l'acquisition GNSS.
#define FS 11999000.0f
#define N 11999
#define NB_PHASES 1023
#define FREQUENCE_CENTRALE 3563000.0f

// Paramètrage recherche coarse (grille large).
#define NB_DOPPLER_COARSE 41
#define DOPPLER_BIN_COARSE 500
#define PRECISION 8
#define MAX_NUM_PEAKS 5
#define SEUIL_VARIANCE_COARSE 0.039f

// Paramètrage recherche fine (raffinement local).
#define NB_DOPPLER_FINE 21
#define DOPPLER_BIN_FINE 250
#define MAX_NUM_PEAKS_FS 16
#define SEUIL_VARIANCE_FINE 0.04f

// Profondeurs AXIS pour COSIM (utilisees par les pragmas top-level).
#define AXIS_IN_DEPTH (N + 1)
#define AXIS_OUT_DEPTH (NB_DOPPLER_FINE * 192 * 3 + 1)

#define PI 3.141592653589793f

#if (defined(__SYNTHESIS__) && !defined(__INTELLISENSE__)) || defined(STSA_CSIM_FIXED)
    typedef ap_axis<32, 0, 0, 0> axis_t;
    typedef ap_fixed<16, 4>   sample_t;
    typedef ap_fixed<18, 2>   trig_t;
    typedef ap_fixed<24, 12>  acc_t;
    // Puissance de correlation I^2 + Q^2 (toujours >= 0).
    typedef ap_ufixed<48, 32> power_t;
    typedef ap_ufixed<24, 4>  metric_t;
    typedef ap_fixed<24, 4>   metric_signed_t;
    
    typedef ap_fixed<40, 8>   metric_acc_t;
    
    typedef ap_fixed<32, 8>   metric_sq_lane_t;
    
    typedef ap_fixed<24, 6>   mix_t;
    typedef ap_fixed<32, 14>  acc_coarse_t;
    
    typedef ap_fixed<24, 14>  acc_coarse_mul_t;
    typedef ap_ufixed<32, 28> pow_tile_t;
    
    typedef pow_tile_t peak_power_t;
    typedef ap_int<24>        doppler_t;
    typedef ap_fixed<32, 10>  angle_t;
    typedef trig_t            osc_t;
#else
    // Mode simulation C
    #if __has_include("hls_stream.h")
    #include "hls_stream.h"
    #endif
    #if __has_include("ap_int.h")
    #include "ap_int.h"
    #endif
    #if __has_include("ap_axi_sdata.h")
    #include "ap_axi_sdata.h"
    #endif
    #include <iostream>
    #include <cmath>
    #include <queue>
    #include <cstdint>
    #include <algorithm>
    #include <vector>
    #include <complex>
    
    using namespace std;
    
#if __has_include("ap_axi_sdata.h") && __has_include("ap_int.h")
    typedef ap_axis<32, 0, 0, 0> axis_t;
#else
    typedef struct {
        int32_t data;
        int32_t keep;
        int32_t strb;
        bool last;
    } axis_t;
#endif
    typedef float sample_t;
    typedef float trig_t;
    typedef float acc_t;
    typedef float power_t;
    typedef float metric_t;
    typedef float metric_signed_t;
    typedef float metric_acc_t;
    typedef float metric_sq_lane_t;
    typedef float mix_t;
    typedef float acc_coarse_t;
    typedef float acc_coarse_mul_t;
    typedef float pow_tile_t;
    typedef float peak_power_t;
    typedef int   doppler_t;
    typedef float angle_t;
    typedef trig_t osc_t;
    
    // Fallback: mock local uniquement si hls_stream.h est indisponible.
    // En cosim Vitis, hls_stream.h est present et doit etre utilise pour
    // garder la meme ABI que le wrapper apatb_*.
    #if !__has_include("hls_stream.h")
    namespace hls {
        template<typename T>
        class stream {
        private:
            std::queue<T> q;
        public:
            void write(const T &v) { q.push(v); }
            T read() {
                if (q.empty()) {
                    std::cerr << "[HLS STREAM ERROR] read() sur stream vide" << std::endl;
                    std::abort();
                }
                T v = q.front();
                q.pop();
                return v;
            }
            bool empty() const { return q.empty(); }
            bool full() const { return false; }
            std::size_t size() const { return q.size(); }
        };
    }
    #endif
#endif

// ========== STRUCTURES ==========
// Pic de corrÃ©lation candidat (Doppler, phase, puissance).
struct PeakInfo {
    doppler_t doppler;
    int phase;
    peak_power_t power;
};

// ========== FONCTIONS ==========
// Insere un pic candidat dans une liste de taille limitee en conservant les plus forts.
void insert_peak(
    PeakInfo ListOfPeaks[MAX_NUM_PEAKS_FS],
    int &num_peaks,
    doppler_t doppler,
    int phase,
    peak_power_t power,
    int max_size
);

// Pipeline principal: capture -> coarse -> fine -> decision finale.
void acquisition_stsa(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    hls::stream<axis_t> &corr_out,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    metric_t threshold_coarse,
    metric_t threshold_fine,
    bool compare_mode,
    metric_t &metric_coarse_var_out,
    metric_t &metric_fine_var_out,
    power_t &peak_coarse_out,
    power_t &peak_fine_out,
    int &max_power_out,
    int &mean_power_out
);

// Wrapper top HLS (interfaces AXIS/AXI-Lite) expose a l'integration IP.
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
);

#endif 
