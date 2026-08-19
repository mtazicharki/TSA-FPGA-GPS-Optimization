//==============================================================================
// acq_blocks.h
// Types partages entre les 3 blocs synthetisables separement :
//   Bloc 1 - dds_mixer_top      (melange DDS)
//   Bloc 2 - correlator_top     (corrélation, circuit fixe reutilise)
//   Bloc 3 - decision_top       (decision : seuil fixe OU seuil de variance)
//
// Ces 3 blocs communiquent par flux AXI-Stream (hls::stream), au meme titre
// que des IP Vivado independantes, sur le meme principe que le tutoriel RGB
// mentionne par Carlos : chaque bloc peut etre synthetise et rapporte
// separement (BRAM/DSP/FF/LUT propres a chaque bloc), puis assembles dans
// le wrapper top-level via #pragma HLS DATAFLOW.
//==============================================================================
#ifndef ACQ_BLOCKS_H
#define ACQ_BLOCKS_H

#include "acq_stsaV2.h"

// ------------------------------------------------------------------
// Paquet emis par le Bloc 1 (mixer) et consomme par le Bloc 2 (correlateur)
// ------------------------------------------------------------------
struct mix_pkt_t {
    sample_t mix_i;
    sample_t mix_q;
    int      idx;        // indice original de l'echantillon (pour le lookup PRN)
    int      bin_id;     // identifiant du bin Doppler courant (0..NB_DOPPLER_COARSE-1
                          // ou 0..NB_DOPPLER_FINE-1 selon le mode)
    bool     bin_last;   // vrai sur le dernier echantillon de ce bin
    bool     sweep_last; // vrai sur le tout dernier paquet du balayage complet
};

// ------------------------------------------------------------------
// Paquet emis par le Bloc 2 (correlateur) et consomme par le Bloc 3 (decision)
// ------------------------------------------------------------------
struct corr_result_pkt_t {
    peak_power_t power;
    int          bin_id;    // bin Doppler d'origine
    int          tau;       // phase de code associee
    int          variant;   // 0/1/2 (nominal, +1/2 chip, -1/2 chip) -- 0 en coarse
    bool         bin_last;
    bool         sweep_last;
};

// ------------------------------------------------------------------
// Mode de balayage : coarse (grille large, accumulation multi-passes STSA)
// ou fine (grille fine, 3 variantes PRN, pas d'accumulation)
// ------------------------------------------------------------------
enum sweep_mode_t { SWEEP_COARSE = 0, SWEEP_FINE = 1 };

// ------------------------------------------------------------------
// Mode de decision : reproduit le choix Option A / Option B discute avec
// Carlos et Jerome -- meme bloc, meme interface, un seul parametre change.
// ------------------------------------------------------------------
enum decision_mode_t {
    DECISION_FIXED_THRESHOLD   = 0,  // Option B : seuil fixe (peak > 0) - version actuelle
    DECISION_VARIANCE_RELATIVE = 1   // Option A : variance normalisee entre pics (comme l'original)
};

// Nombre de meilleurs pics conserves pour le calcul de variance (Option A)
#define DECISION_TOPK 5

#endif
