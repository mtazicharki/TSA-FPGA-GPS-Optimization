//============================================
// acq_stsaV2.cpp
// ==========================================
#include "acq_stsaV2.h"
#include "dds_lut_rom.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

// Nombre de bits utilisés pour indexer la LUT DDS (1024 entrées = 2^10).
// Compromis résolution spectrale / taille BRAM : 10 bits -> LUT de 1024 points.
static const int STSA_DDS_LUT_BITS   = 10;
// Accumulateur de phase sur 32 bits : résolution de fréquence = Fs/2^32 ≈ 0.0028 Hz.
static const int STSA_DDS_PHASE_BITS = 32;
// Fréquence d'échantillonnage du signal reçu (Hz).
static const int STSA_FS_HZ          = 11999000;
// Fréquence porteuse intermédiaire à annuler lors du mélange DDS (Hz).
static const int STSA_FC_HZ          = 3563000;

// Nombre de variantes PRN générées : nominale (v0), avancée +1/2 chip (v1), retardée -1/2 chip (v2).
// Ces trois corrélateurs simultanés permettent la discrimination sub-chip en phase.
static const int PRN_VARIANTS  = 3;
// Taille d'une tuile de phases traitées en parallèle dans la boucle fine.
// Doit être une puissance de 2 pour l'alignement et l'UNROLL complet.
static const int FINE_TAU_TILE = 16;
// Nombre maximal de pics conservés dans le top-K de la recherche coarse.
static const int COARSE_TOPK   = 16;
// 3 variantes PRN: nominale, +1/2 chip, -1/2 chip.

// Sous synthèse, on utilise des types ap_uint/ap_int à largeur minimale
// pour économiser les ressources LUT/FF. En simulation C, on replie sur
// des types C standard compatibles pour éviter la dépendance à ap_types.
#if defined(__SYNTHESIS__) && !defined(__INTELLISENSE__)
typedef ap_uint<2> variant_t;
typedef ap_uint<1> prn_sign_t;
#else
typedef unsigned char variant_t;
typedef unsigned char prn_sign_t;
#endif

// Sous synthèse, les types ap_uint/ap_int permettent à Vitis HLS d'inférer
// exactement la largeur de bus nécessaire, évitant les mux/registres superflus.
// dds_lut_index extrait les STSA_DDS_LUT_BITS MSB de l'accumulateur de phase,
// ce qui revient à prendre la partie entière de (phase / 2^(32-10)).
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
// Équivalent logiciel du range() HLS : décalage à droite pour garder les 10 MSB.
static inline unsigned dds_lut_index(dds_phase_u_t phase_acc) {
    return phase_acc >> (STSA_DDS_PHASE_BITS - STSA_DDS_LUT_BITS);
}
#endif

// Convertit une frequence Doppler (Hz) en increment de phase DDS.
// Le signe negatif conserve la convention de de-rotation utilisee ici.
static inline dds_phase_t hz_to_phase_inc(doppler_t fd_hz) {
#pragma HLS INLINE
    // Convertit une fréquence (Hz) en incrément de phase DDS.
    // freq_total_hz = -(Fc + fd) : fréquence totale à annuler (porteuse + Doppler).
    // Le décalage gauche de 32 bits puis la division par Fs réalise :
    //   inc = round(freq / Fs * 2^32), en arithmétique entière 64 bits pour éviter
    //   la perte de précision qu'introduirait une division flottante.
    const int freq_total_hz = -(STSA_FC_HZ + (int)fd_hz);
    long long num = ((long long)freq_total_hz << STSA_DDS_PHASE_BITS);
    return (dds_phase_t)(num / STSA_FS_HZ);
}

// Quantifie le PRN en signe binaire (1 => +1, 0 => -1).
static inline prn_sign_t to_prn_sign(sample_t x) {
#pragma HLS INLINE
    // Binarise le PRN: >=0 -> +1, sinon -1.
    // La binarisation du PRN permet de remplacer la multiplication corrélateur
    // par un simple changement de signe (±1), économisant des DSP en fine search.
    return (x >= (sample_t)0) ? (prn_sign_t)1 : (prn_sign_t)0;
}

// Paquet de puissance élémentaire produit par process_*_all_tau et consommé
// par reduce_fine_stream_core via la FIFO pow_s.
struct fine_pow_pkt_t {
    // Puissance élémentaire pour un couple (doppler, tau, variante PRN).
    peak_power_t power;
    doppler_t fd;
    variant_t variant;
    int       tau;
};

// Paquet de maximum par variante et par bin Doppler, émis par reduce_fine_stream_core
// vers consume_fine_maxima_and_insert via la FIFO max_s.
struct fine_max_pkt_t {
    // Maximum trouvé pour une variante PRN Ã  Doppler donné.
    peak_power_t max_power;
    doppler_t fd;
    variant_t variant;
    int       phase_idx;
};

// Paquet de sortie brute vers l'AXI-Stream corr_out (surface de corrélation complète).
struct corr_pkt_t {
    peak_power_t power;
};

// Paquet de statistiques globales (max et somme) calculé en une seule passe
// dans reduce_fine_stream_core et transmis vers read_fine_stats via stats_s.
struct fine_stats_pkt_t {
    // Statistiques globales utiles pour max/mean en sortie.
    power_t max_power;
    power_t sum_power;
};

// ===== [NOUVEAU BLOC 1] ParamÃ¨tres + Helpers thÃ¨se =====
// Rang de référence basse pour le calcul du ratio p1/p5 en coarse (critère thèse).
static const int CS_REF_RANK = 5;   // Coarse: p1 vs p5
// Rang de référence basse pour le calcul de la variance normalisée en fine (critère thèse Eq.2).
static const int FS_REF_RANK = 4;   // Fine:   p1 vs p4
// Séparation minimale en bins de phase entre deux pics acceptés dans le top-K espacé.
// Évite de compter plusieurs fois le même pic réel étalé sur plusieurs phases.
static const int MIN_PEAK_SEP = 16; // séparation minimale en phase (Ã  tuner)
static const int MIN_DOPPLER_SEP_FINE = DOPPLER_BIN_FINE; // séparation minimale en Doppler pour le filtre fine
// Seuil bas de variance fine = 85 % du seuil nominal, utilisé pour le fallback borderline.
static const metric_t FS_THRESHOLD_LOW_SCALE = (metric_t)0.85; // seuil bas = 0.85 * seuil fin
// Ratio de garde : si le pic global dépasse 1.2× le pic de meilleure variance,
// on suspecte un conflit entre variantes et on bascule sur le pic de meilleure variance.
static const metric_t FS_GLOBAL_PEAK_GUARD_RATIO = (metric_t)1.20; // garde-fou amplitude pic global
// Dominance locale minimale : ratio pic / meilleur voisin >= 1.0 (le pic doit être au moins
// aussi fort que son plus proche concurrent dans la fenêtre locale phase+Doppler).
static const metric_t FS_LOCAL_DOMINANCE_MIN = (metric_t)1.00;
// Fenêtre de voisinage local en phase (±1 bin) pour le calcul de dominance.
static const int FS_LOCAL_PHASE_WIN = 1; // voisinage local en phase
// Fenêtre de voisinage local en Doppler pour le calcul de dominance (= 1 bin Doppler fin).
static const int FS_LOCAL_DOPPLER_WIN = DOPPLER_BIN_FINE; // voisinage local en Doppler
static const int MIN_DOPPLER_SEP_COARSE = DOPPLER_BIN_COARSE; // séparation min Doppler coarse (Hz)
// Tolérance maximale de dérive Doppler entre l'estimation coarse et le pic fine retenu.
// Au-delà, le pic fine est jugé incohérent avec la coarse et ignoré.
static const int FINE_REFINE_DOPPLER_MAX = 2 * DOPPLER_BIN_FINE; // +/- 500 Hz
// Tolérance maximale de dérive en phase entre l'estimation coarse et le pic fine retenu.
static const int FINE_REFINE_PHASE_MAX = 64; // +/- 64 phases

// Variables de debug csim uniquement (exclues de la synthèse par #ifndef).
// Permettent d'inspecter les pics p1/p5 coarse et p1/p4 fine sans instrumenter le RTL.
#ifndef __SYNTHESIS__
static power_t dbg_cs_p1 = 0;
static power_t dbg_cs_p5 = 0;
static int     dbg_cs_top_count = 0;

static power_t dbg_fs_p1 = 0;
static power_t dbg_fs_p4 = 0;
static int     dbg_fs_top_spaced_count = 0;
#endif

static inline int phase_circular_dist(int a, int b) {
#pragma HLS INLINE
    // Distance circulaire sur [0, NB_PHASES).
    // Le min(d, NB_PHASES-d) gère le cas où deux phases sont séparées par
    // le bord circulaire (ex : phase 0 et phase NB_PHASES-1 sont adjacentes).
    int d = a - b;
    if (d < 0) d = -d;
    int wrap = NB_PHASES - d;
    return (d < wrap) ? d : wrap;
}

static inline int doppler_abs_dist(doppler_t a, doppler_t b) {
#pragma HLS INLINE
    // Distance absolue en bins Doppler.
    // Le cast en int est nécessaire car doppler_t peut être un ap_fixed signé
    // dont la soustraction directe produirait un résultat tronqué.
    int d = (int)a - (int)b;
    return (d < 0) ? -d : d;
}

static inline bool is_local_neighbor(
    const PeakInfo &ref_peak,
    const PeakInfo &cand_peak,
    int phase_win,
    int doppler_win
) {
#pragma HLS INLINE
    // Vrai si candidat dans le voisinage local du pic de référence.
    // Utilisé pour calculer la dominance locale : on cherche le concurrent
    // le plus fort dans une fenêtre (phase_win × doppler_win) autour du pic.
    int d_phase = phase_circular_dist(ref_peak.phase, cand_peak.phase);
    int d_doppler = doppler_abs_dist(ref_peak.doppler, cand_peak.doppler);
    return (d_phase <= phase_win) && (d_doppler <= doppler_win);
}

// NVar sur 2 valeurs normalisées par p1 (p1 >= pk)
// esprit Eq.(2): variance des points {1, pk/p1}
// Plus la variance est élevée, plus les pics sont dissemblables,
// ce qui traduit la présence d'un vrai pic dominant (signal acquis).
static inline metric_t nvar_from_p1_pk(power_t p1, power_t pk) {
#pragma HLS INLINE
    if (p1 <= (power_t)0) return (metric_t)0;
    // Solution A: widening chirurgical du calcul interne pour eviter la
    // sous-estimation par quantification (ap_fixed<24,4> -> ap_fixed<40,4>).
    // Les types d'interface (metric_t / metric_signed_t) restent inchanges,
    // donc aucun impact sur Fmax dans les autres etages.
    metric_acc_t r  = (metric_acc_t)(pk / p1);                         // r = pk/p1
    metric_acc_t m  = ((metric_acc_t)1 + r) * (metric_acc_t)0.5;       // moyenne
    // |d1|, |d2| <= 0.5 (r dans [0,1]) -> 32 bits (8.24) suffisent largement.
    // Mul 32x32 sur DSP avec latency=4 -> meme timing closure que SQ_LOOP coarse.
    metric_sq_lane_t d1 = (metric_sq_lane_t)((metric_acc_t)1 - m);
    metric_sq_lane_t d2 = (metric_sq_lane_t)(r - m);
    // Carrés forcés sur DSP (latency=4) : évite l'inférence d'un chemin
    // combinatoire long qui casserait le timing closure à haute fréquence.
    metric_sq_lane_t d1sq = d1 * d1;
    metric_sq_lane_t d2sq = d2 * d2;
#pragma HLS BIND_OP variable=d1sq op=mul impl=dsp latency=4
#pragma HLS BIND_OP variable=d2sq op=mul impl=dsp latency=4
    metric_acc_t var = (metric_acc_t)0.5 * ((metric_acc_t)d1sq + (metric_acc_t)d2sq);
    // Clip à 0 : metric_acc_t est signé pour gérer les écarts négatifs internes,
    // mais la variance ne peut être négative ; on sature pour la sortie.
    return (var > (metric_acc_t)0) ? (metric_t)var : (metric_t)0;
}

// Selection top-K avec separation minimale (phase + doppler).
// Algorithme glouton : à chaque itération, on sélectionne le meilleur pic non utilisé
// et on le rejette s'il est trop proche d'un pic déjà retenu (critère AND phase+doppler).
static void select_spaced_topk(
    const PeakInfo top_in[MAX_NUM_PEAKS_FS],
    int n_top_in,
    int k_target,
    int phase_sep,
    int doppler_sep,
    PeakInfo out_spaced[MAX_NUM_PEAKS_FS],
    int &n_spaced_out
) {
#pragma HLS INLINE off
    // Tableau de marquage pour éviter de revisiter un pic déjà traité.
    bool used[MAX_NUM_PEAKS_FS];

    INIT_USED: for (int i = 0; i < MAX_NUM_PEAKS_FS; i++) {
#pragma HLS PIPELINE II=1
        used[i] = false;
    }

    int count = 0;

    // Boucle externe non pipelinée car elle contient un accès conditionnel
    // à out_spaced[] dont la taille dépend de l'itération (dépendance RAW).
    SELECT_SPACED_LOOP: for (int r = 0; r < MAX_NUM_PEAKS_FS; r++) {
#pragma HLS PIPELINE off
        int best_idx = -1;
        peak_power_t best_p = 0;

        // Recherche linéaire du pic de puissance maximale non encore utilisé.
        FIND_BEST_UNUSED: for (int i = 0; i < MAX_NUM_PEAKS_FS; i++) {
#pragma HLS PIPELINE II=1
            if ((i < n_top_in) && (!used[i])) {
                if ((best_idx < 0) || (top_in[i].power > best_p)) {
                    best_idx = i;
                    best_p = top_in[i].power;
                }
            }
        }

        if (best_idx >= 0) {
            bool far_enough = true;
            // Vérifie la séparation minimale par rapport à tous les pics déjà retenus.
            // Condition AND (phase ET doppler) : un pic trop proche dans les DEUX
            // dimensions est rejeté comme doublon probable du même trajet.
            CHECK_SEP_LOOP: for (int j = 0; j < MAX_NUM_PEAKS_FS; j++) {
#pragma HLS PIPELINE II=1
                if (j < count) {
                    int d_phase = phase_circular_dist(top_in[best_idx].phase, out_spaced[j].phase);
                    int d_dopp  = doppler_abs_dist(top_in[best_idx].doppler, out_spaced[j].doppler);
                    if ((d_phase < phase_sep) && (d_dopp < doppler_sep)) {
                        far_enough = false;
                    }
                }
            }

            // Marque toujours le pic comme traité, même s'il est rejeté pour séparation,
            // pour ne pas le réévaluer lors des prochaines itérations.
            used[best_idx] = true;

            if (far_enough && (count < k_target) && (count < MAX_NUM_PEAKS_FS)) {
                out_spaced[count] = top_in[best_idx];
                count++;
            }
        }
    }

    n_spaced_out = count;
}

// Primitive élémentaire de tri : échange a et b si b > a (ordre décroissant).
// Inlinée systématiquement pour former un réseau de tri sans appel de fonction en RTL.
static inline void cmp_swap_desc(PeakInfo &a, PeakInfo &b) {
#pragma HLS INLINE
    // Compare/swap pour tri décroissant de puissance.
    if (b.power > a.power) {
        PeakInfo t = a;
        a = b;
        b = t;
    }
}

// Réseau de tri optimal à 5 comparaisons pour 4 éléments (tri par insertion sans boucle).
// Entièrement combinatoire après inlining : 0 cycle de latence supplémentaire.
static inline void sort4_desc(PeakInfo top[4]) {
#pragma HLS INLINE
    // Petit réseau de tri fixe (4 éléments).
    cmp_swap_desc(top[0], top[1]);
    cmp_swap_desc(top[2], top[3]);
    cmp_swap_desc(top[0], top[2]);
    cmp_swap_desc(top[1], top[3]);
    cmp_swap_desc(top[1], top[2]);
}

// Réseau de tri optimal à 9 comparaisons pour 5 éléments.
// Nombre minimal de comparaisons connu pour 5 éléments, entièrement déroulé.
static inline void sort5_desc(PeakInfo top[5]) {
#pragma HLS INLINE
    // Petit réseau de tri fixe (5 éléments).
    cmp_swap_desc(top[0], top[1]);
    cmp_swap_desc(top[3], top[4]);
    cmp_swap_desc(top[2], top[4]);
    cmp_swap_desc(top[2], top[3]);
    cmp_swap_desc(top[1], top[4]);
    cmp_swap_desc(top[0], top[3]);
    cmp_swap_desc(top[0], top[2]);
    cmp_swap_desc(top[1], top[3]);
    cmp_swap_desc(top[1], top[2]);
}

// Insertion dans un top-5 coarse avec contrainte de séparation en phase.
// Si le candidat est proche d'un pic existant (distance < min_sep), on garde
// le plus fort des deux (fusion de pics) plutôt que d'en créer un nouveau.
static inline void insert_top5_cs(
    PeakInfo top[5], int &num_top, const PeakInfo &cand, int min_sep
) {
#pragma HLS INLINE
    // Garde un top-5 coarse avec contrainte de séparation en phase.
    int near_idx = -1;
    if (num_top > 0 && phase_circular_dist(top[0].phase, cand.phase) < min_sep) near_idx = 0;
    else if (num_top > 1 && phase_circular_dist(top[1].phase, cand.phase) < min_sep) near_idx = 1;
    else if (num_top > 2 && phase_circular_dist(top[2].phase, cand.phase) < min_sep) near_idx = 2;
    else if (num_top > 3 && phase_circular_dist(top[3].phase, cand.phase) < min_sep) near_idx = 3;
    else if (num_top > 4 && phase_circular_dist(top[4].phase, cand.phase) < min_sep) near_idx = 4;

    if (near_idx >= 0) {
        // Fusion : mise à jour du pic existant si le candidat est plus fort.
        if (cand.power > top[near_idx].power) top[near_idx] = cand;
    } else {
        if (num_top < 5) {
            top[num_top] = cand;
            num_top++;
        } else if (cand.power > top[4].power) {
            // Le tableau est plein : éviction du 5e (le moins bon) si le candidat est meilleur.
            top[4] = cand;
        } else {
            return;
        }
    }
    // Re-tri après chaque insertion pour maintenir l'ordre décroissant et
    // garantir que top[4] est toujours le minimum (cible d'éviction).
    sort5_desc(top);
}

// Version simplifiée sans contrainte de séparation : garde les 4 plus fortes
// puissances toutes positions confondues. Utilisée pour les statistiques globales.
static inline void insert_top4_power_only(
    PeakInfo top[4], int &num_top, const PeakInfo &cand
) {
#pragma HLS INLINE
    // Garde uniquement les 4 plus fortes puissances.
    if (num_top < 4) {
        top[num_top] = cand;
        num_top++;
    } else if (cand.power > top[3].power) {
        top[3] = cand;
    } else {
        return;
    }
    sort4_desc(top);
}

// Insertion dans un top-5 fixe sans contrainte de séparation ni tri complet.
// Remplace uniquement le minimum courant si le candidat est meilleur.
// Plus économique en ressources que sort5 car ne maintient pas l'ordre global.
static inline void insert_peak_top5_fixed(
    PeakInfo top[5], int &num_top, const PeakInfo &cand
) {
#pragma HLS INLINE
    // Version top-5 fixe: remplace le minimum si meilleur candidat.
    if (num_top < 5) {
        top[num_top] = cand;
        num_top++;
        return;
    }

    int min_idx = 0;
    peak_power_t min_power = top[0].power;

    // Optimisation sÃ»re : UNROLL sur boucle trÃ¨s courte (5)
    // UNROLL total sur 4 itérations : génère 4 comparateurs parallèles en LUT,
    // élimine le chemin critique séquentiel et réduit la latence à 1 niveau logique.
    FIND_MIN_TOP5: for (int i = 1; i < 5; i++) {
#pragma HLS UNROLL // Ajouté : boucle trÃ¨s courte, sans risque pour la synthÃ¨se
        if (top[i].power < min_power) {
            min_power = top[i].power;
            min_idx = i;
        }
    }

    if (cand.power > min_power) {
        top[min_idx] = cand;
    }
}

// Filtre post-sélection sur un top-4 : élimine les doublons trop proches
// (condition AND phase+doppler) en parcourant les candidats dans l'ordre décroissant
// de puissance. Utilisé uniquement en simulation (boucles non pipelinées).
static inline int post_filter_top4_spacing(
    const PeakInfo in_top[4],
    int in_count,
    PeakInfo out_top[4],
    int min_phase_sep,
    int min_doppler_sep
) {
#pragma HLS INLINE
    int out_count = 0;

    for (int i = 0; i < in_count; i++) {
#pragma HLS PIPELINE off
        bool keep = true;
        int reject_j = -1;
        int reject_phase_dist = -1;
        int reject_doppler_dist = -1;
        // Pour chaque candidat, on vérifie s'il est trop proche d'un pic déjà accepté.
        for (int j = 0; j < out_count; j++) {
#pragma HLS PIPELINE off
            int d_phase = phase_circular_dist(in_top[i].phase, out_top[j].phase);
            int d_doppler = doppler_abs_dist(in_top[i].doppler, out_top[j].doppler);
            if ((d_phase < min_phase_sep) && (d_doppler < min_doppler_sep)) {
                keep = false;
                reject_j = j;
                reject_phase_dist = d_phase;
                reject_doppler_dist = d_doppler;
                break;
            }
        }

#ifndef __SYNTHESIS__
        if (keep) {
            std::cout << "  [DBG FS FILTER] keep  in[" << i << "]"
                      << " doppler=" << (int)in_top[i].doppler
                      << " phase=" << in_top[i].phase
                      << " power=" << in_top[i].power << std::endl;
        } else {
            std::cout << "  [DBG FS FILTER] reject in[" << i << "]"
                      << " doppler=" << (int)in_top[i].doppler
                      << " phase=" << in_top[i].phase
                      << " power=" << in_top[i].power
                          << " : trop proche (phase+doppler) de out[" << reject_j << "]"
                          << " (doppler=" << (int)out_top[reject_j].doppler
                      << " (phase=" << out_top[reject_j].phase << ")"
                          << " d_phase=" << reject_phase_dist
                          << " < min_phase_sep=" << min_phase_sep
                          << " et d_doppler=" << reject_doppler_dist
                          << " < min_doppler_sep=" << min_doppler_sep
                      << std::endl;
        }
#endif

        if (keep && out_count < 4) {
            out_top[out_count] = in_top[i];
            out_count++;
        }
    }
    return out_count;
}

// ========== INPUT CAPTURE ==========
// Premier étage de la chaîne de capture : lit N échantillons depuis les deux
// flux AXI-Stream (signal reçu + PRN) et les redirige vers deux FIFO internes.
// Séparé en fonction distincte pour que le DATAFLOW puisse recouvrir la lecture
// et le stockage en mémoire (producteur de la paire de FIFO sig_s / prn_s).
static void read_inputs_to_stream_stsa(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    hls::stream<sample_t> &sig_s,
    hls::stream<sample_t> &prn_s
) {
#pragma HLS INLINE off

    // Pipeline II=1 : lecture et écriture FIFO chaque cycle, débit = Fs.
    READ_INPUTS_STREAM: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
        if (i < 4) {
            std::cout << "[TRACE CAPTURE STEP] i=" << i << " begin" << std::endl;
        }
        if (rx_stream.empty() || prn_stream.empty()) {
            std::cout << "[TRACE CAPTURE ERROR] input underflow i=" << i
                      << " rx_empty=" << (rx_stream.empty() ? 1 : 0)
                      << " prn_empty=" << (prn_stream.empty() ? 1 : 0)
                      << std::endl;
            std::abort();
        }
        if (i < 4) {
            std::cout << "[TRACE CAPTURE STEP] i=" << i << " after empty-check" << std::endl;
        }
#endif
        axis_t rx_val  = rx_stream.read();
        axis_t prn_val = prn_stream.read();

#ifndef __SYNTHESIS__
        if (i < 4) {
            std::cout << "[TRACE CAPTURE STEP] i=" << i << " after input read" << std::endl;
        }
        if (i == 0 || i == (N - 1)) {
            std::cout << "[TRACE CAPTURE IN] i=" << i
                      << " rx_data=" << rx_val.data
                      << " prn_data=" << prn_val.data
                      << " rx_last=" << (rx_val.last ? 1 : 0)
                      << " prn_last=" << (prn_val.last ? 1 : 0)
                      << std::endl;
        }
#endif

        // Cast explicite : axis_t.data est un entier générique (AP_INT ou int),
        // on le projette sur le type sample_t (ap_fixed) attendu par les étages suivants.
        sig_s.write((sample_t)rx_val.data);
#ifndef __SYNTHESIS__
        if (i < 4) {
            std::cout << "[TRACE CAPTURE STEP] i=" << i << " after sig_s.write" << std::endl;
        }
#endif
        prn_s.write((sample_t)prn_val.data);
#ifndef __SYNTHESIS__
        if (i < 4) {
            std::cout << "[TRACE CAPTURE STEP] i=" << i << " after prn_s.write" << std::endl;
        }
#endif
    }
}

// Second étage de capture : consomme les FIFO sig_s / prn_s et écrit dans
// les tableaux RAM signal[] et prn[]. Maintenu séparé pour le DATAFLOW :
// la lecture AXI-Stream et le stockage RAM s'exécutent en recouvrement.
static void store_stream_to_mem_stsa(
    hls::stream<sample_t> &sig_s,
    hls::stream<sample_t> &prn_s,
    sample_t signal[N],
    sample_t prn[N]
) {
#pragma HLS INLINE off

    STORE_INPUTS_MEM: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
        if (sig_s.empty() || prn_s.empty()) {
            std::cout << "[TRACE CAPTURE ERROR] internal underflow i=" << i
                      << " sig_empty=" << (sig_s.empty() ? 1 : 0)
                      << " prn_empty=" << (prn_s.empty() ? 1 : 0)
                      << std::endl;
            std::abort();
        }
#endif
        signal[i] = sig_s.read();
        prn[i]    = prn_s.read();

#ifndef __SYNTHESIS__
        if (i == 0 || i == (N - 1)) {
            std::cout << "[TRACE CAPTURE STORE] i=" << i
                      << " signal=" << signal[i]
                      << " prn=" << prn[i]
                      << std::endl;
        }
#endif
    }
}

// Orchestrateur de capture : lit les deux flux AXIS et remplit les buffers RAM.
// En cosim (#ifndef __SYNTHESIS__), la boucle directe évite le DATAFLOW interne
// qui peut provoquer un crash du wrapper C généré par Vitis HLS.
// En synthèse, le DATAFLOW recouvre read et store pour réduire la latence totale.
static void capture_inputs_to_mem_stsa(
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream,
    sample_t signal[N],
    sample_t prn[N]
) {
#pragma HLS INLINE off
    // Copie RX/PRN AXIS vers des buffers mémoire locaux.

#ifndef __SYNTHESIS__
    // Chemin cosim robuste: éviter DATAFLOW+streams internes qui peuvent provoquer un crash cÃ´té wrapper C.
    std::cout << "[TRACE CAPTURE DIRECT] initial stream sizes: rx=" << rx_stream.size()
              << " prn=" << prn_stream.size() << std::endl;
    READ_STORE_DIRECT: for (int i = 0; i < N; i++) {
        if (rx_stream.empty() || prn_stream.empty()) {
            std::cout << "[TRACE CAPTURE ERROR] direct input underflow i=" << i
                      << " rx_empty=" << (rx_stream.empty() ? 1 : 0)
                      << " prn_empty=" << (prn_stream.empty() ? 1 : 0)
                      << std::endl;
            std::abort();
        }

        axis_t rx_val = rx_stream.read();
        axis_t prn_val = prn_stream.read();
        signal[i] = (sample_t)rx_val.data;
        prn[i] = (sample_t)prn_val.data;

        if (i == 0 || i == (N - 1)) {
            std::cout << "[TRACE CAPTURE DIRECT] i=" << i
                      << " rx_data=" << rx_val.data
                      << " prn_data=" << prn_val.data
                      << " rx_last=" << (rx_val.last ? 1 : 0)
                      << " prn_last=" << (prn_val.last ? 1 : 0)
                      << std::endl;
        }
    }
    return;
#endif

    // Deux FIFO internes de profondeur 64 (SRL16 pour économiser les BRAM) :
    // elles découplent le producteur (read_inputs) du consommateur (store_stream)
    // et absorbent les bulles dues aux latences de lecture AXIS.
    hls::stream<sample_t> sig_s;
    hls::stream<sample_t> prn_s;
#pragma HLS STREAM variable=sig_s depth=64
#pragma HLS STREAM variable=prn_s depth=64
// SRL : implémentation en registres à décalage (Shift Register LUT) au lieu de BRAM,
// économique pour des FIFO courtes (≤ 32-64 entrées).
#pragma HLS BIND_STORAGE variable=sig_s type=FIFO impl=SRL
#pragma HLS BIND_STORAGE variable=prn_s type=FIFO impl=SRL
// DATAFLOW : Vitis HLS génère un ordonnancement de type pipeline de tâches ;
// read_inputs et store_stream s'exécutent en recouvrement dès que la FIFO
// contient au moins un élément, réduisant la latence totale de N cycles.
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
    std::cout << "[TRACE CAPTURE] before read_inputs_to_stream_stsa" << std::endl;
#endif
    read_inputs_to_stream_stsa(rx_stream, prn_stream, sig_s, prn_s);
#ifndef __SYNTHESIS__
    std::cout << "[TRACE CAPTURE] after read_inputs_to_stream_stsa" << std::endl;
    std::cout << "[TRACE CAPTURE] before store_stream_to_mem_stsa" << std::endl;
#endif
    store_stream_to_mem_stsa(sig_s, prn_s, signal, prn);
#ifndef __SYNTHESIS__
    std::cout << "[TRACE CAPTURE] after store_stream_to_mem_stsa" << std::endl;
#endif
}

// ========== INSERT PEAK ==========
// Maintient une liste des max_size meilleurs pics (doppler, phase, power).
// Pendant le remplissage (num_peaks < max_size), on ajoute directement.
// Une fois plein, on cherche le minimum et on l'évince si le candidat est meilleur.
// Pas de tri : le coût O(max_size) de recherche du min est acceptable ici.
void insert_peak(
    PeakInfo ListOfPeaks[MAX_NUM_PEAKS_FS],
    int &num_peaks,
    doppler_t doppler,
    int phase,
    peak_power_t power,
    int max_size
) {
#pragma HLS INLINE
    // Remplace le plus petit pic si le candidat est meilleur.

    if(num_peaks < max_size) {
        ListOfPeaks[num_peaks].doppler = doppler;
        ListOfPeaks[num_peaks].phase   = phase;
        ListOfPeaks[num_peaks].power   = power;
        num_peaks++;
        return;
    }

    // Recherche linéaire du minimum : la liste n'est pas triée, donc O(max_size).
    // Acceptable car max_size = MAX_NUM_PEAKS_FS est petit (≤ 16).
    int min_idx = 0;
    peak_power_t min_power = ListOfPeaks[0].power;

    for(int i = 1; i < num_peaks; i++) {
        if(ListOfPeaks[i].power < min_power) {
            min_power = ListOfPeaks[i].power;
            min_idx = i;
        }
    }

    if(power > min_power) {
        ListOfPeaks[min_idx].doppler = doppler;
        ListOfPeaks[min_idx].phase   = phase;
        ListOfPeaks[min_idx].power   = power;
    }
}

// ========== DDS MIX ==========
// Mélangeur numérique (DDS) : multiplie chaque échantillon du signal par
// cos(-ω_c·n) et sin(-ω_c·n) pour ramener le signal en bande de base complexe.
// ============================================================
// DDS PARTAGÉ — une seule instance BRAM pour coarse ET fine
//
// Remplace les deux DDS séparés :
//   - fine_mix_to_mem()  : boucle MIX_LOOP (pas = phase_inc × 1)
//   - coarse MIX_SUB_PASS : boucle inline  (pas = phase_inc × PRECISION)
//
// #pragma HLS INLINE off : HLS génère UN SEUL bloc matériel
// -> une seule BRAM pour DDS_COS_LUT et DDS_SIN_LUT
//
// Paramètres :
//   signal[]    : signal GPS d'entrée
//   indices[]   : indices des échantillons à traiter
//   n_count     : nombre d'échantillons
//   phase_init  : phase initiale de l'accumulateur
//                 fine   : 0
//                 coarse : phase_inc * pass
//   phase_step  : incrément entre deux échantillons consécutifs
//                 fine   : phase_inc * 1
//                 coarse : phase_inc * PRECISION (= phase_inc * 8)
//   out_i[]     : composante I en sortie
//   out_q[]     : composante Q en sortie
// ============================================================
static void dds_mix_shared(
    const sample_t  signal[N],
    const int       indices[],
    int             n_count,
    dds_phase_u_t   phase_init,
    dds_phase_u_t   phase_step,
    sample_t        out_i[],
    sample_t        out_q[]
) {
#pragma HLS INLINE off
#pragma HLS DEPENDENCE variable=out_i inter false
#pragma HLS DEPENDENCE variable=out_q inter false

    dds_phase_u_t phase_acc = phase_init;

    DDS_SHARED_LOOP: for (int k = 0; k < n_count; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=out_i inter false
#pragma HLS DEPENDENCE variable=out_q inter false

        unsigned lut_idx = dds_lut_index(phase_acc);
        // Lecture unique de la LUT partagée (une seule BRAM)
        trig_t c = DDS_COS_LUT[lut_idx];
        trig_t s = DDS_SIN_LUT[lut_idx];

        sample_t x  = signal[indices[k]];
        sample_t xc = x * c;
        sample_t xs = x * s;
#pragma HLS BIND_OP variable=xc op=mul impl=dsp latency=3
#pragma HLS BIND_OP variable=xs op=mul impl=dsp latency=3

        out_i[k] = xc;
        out_q[k] = -xs;

        // Avancement de l'accumulateur avec le pas correct
        // fine   : phase_step = phase_inc × 1
        // coarse : phase_step = phase_inc × PRECISION
        phase_acc = (dds_phase_u_t)(phase_acc + phase_step);
    }
}

// Les composantes I et Q sont stockées en BRAM pour être relues par les
// corrélateurs de la fine search (réutilisation pour toutes les phases τ).
static void fine_mix_to_mem(
    const sample_t signal[N],
    sample_t mix_i[N],
    sample_t mix_q[N],
    doppler_t fd
) {
#pragma HLS INLINE off

    dds_phase_t   phase_inc  = hz_to_phase_inc(fd);
    // Phase initiale = 0 : la fine démarre toujours à phase nulle
    dds_phase_u_t phase_init = 0;
    // Pas = phase_inc × 1 : tous les échantillons consécutifs
    dds_phase_u_t phase_step = (dds_phase_u_t)phase_inc;

    // Indices séquentiels 0..N-1 pour la fine
    static int fine_indices[N];
    FINE_IDX_LOOP: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
        fine_indices[n] = n;
    }

    // Appel DDS partagé — même BRAM LUT que la coarse
    // phase_step = phase_inc × 1 (pas de sous-échantillonnage)
    dds_mix_shared(signal, fine_indices, N,
                   phase_init, phase_step,
                   mix_i, mix_q);
}

// ========== PRN TABLES ==========
// Précalcule l'indice de début de chaque phase τ dans le tableau signal[].
// tau_start_tbl[τ] = floor(τ * N / NB_PHASES) : échantillon de signal correspondant
// au décalage de τ chips, en tenant compte du rapport N/NB_PHASES (oversampling).
static void build_tau_start_table(int tau_start_tbl[NB_PHASES]) {
#pragma HLS INLINE off

    BUILD_TAU_START: for (int tau = 0; tau < NB_PHASES; tau++) {
#pragma HLS PIPELINE II=1
        // Multiplication 64 bits pour éviter le débordement si tau * N > 2^31.
        tau_start_tbl[tau] = (int)(((long long)tau * (long long)N) / NB_PHASES);
    }
}

// Génère les 3 variantes binarisées du PRN et les duplique sur [0, 2N) pour
// permettre un accès circulaire sans modulo dans les corrélateurs (prn[n + shift]
// avec shift < N ne nécessite pas de test de bord si le tableau fait 2N).
static void build_prn_variants(
    const sample_t prn[N],
    prn_sign_t prn_var[PRN_VARIANTS][2 * N]
) {
#pragma HLS INLINE off

    BUILD_PRN_BASE: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        // Calcul de l'index suivant/précédent modulo N pour les variantes ±½ chip.
        int next_idx = (i + 1) % N;
        int prev_idx = (i - 1 + N) % N;

        // Variante nominale : chip centré à i.
        sample_t base      = prn[i];
        // Variante +½ chip : interpolation linéaire entre chip i et chip i+1.
        sample_t half_plus = (sample_t)0.5f * (prn[i] + prn[next_idx]);
        // Variante -½ chip : interpolation linéaire entre chip i-1 et chip i.
        sample_t half_minus= (sample_t)0.5f * (prn[i] + prn[prev_idx]);

        prn_var[0][i] = to_prn_sign(base);
        prn_var[1][i] = to_prn_sign(half_plus);
        prn_var[2][i] = to_prn_sign(half_minus);
    }

    // Duplication de la première moitié dans la seconde : permet prn_var[v][k + N]
    // sans test de débordement dans les boucles de corrélation.
    DUP_PRN_ALL: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        prn_var[0][i + N] = prn_var[0][i];
        prn_var[1][i + N] = prn_var[1][i];
        prn_var[2][i + N] = prn_var[2][i];
    }
}

// Crée FINE_TAU_TILE copies identiques de chaque variante PRN dans une dimension
// supplémentaire (dimension 1 = banc). But : permettre à Vitis HLS de lire
// FINE_TAU_TILE accès simultanés en un cycle (II=1 dans SAMPLE_LOOP_ALL)
// sans conflit de port BRAM, grâce au partitionnement complet sur dim=2.
static void build_prn_banks(
    const prn_sign_t prn_var[PRN_VARIANTS][2 * N],
    prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N]
) {
#pragma HLS INLINE off
// ram_2p impl=bram : deux ports de lecture/écriture distincts, mappés sur BRAM18/36.
// Nécessaire pour maintenir II=1 avec les accès en lecture depuis SAMPLE_LOOP.
#pragma HLS BIND_STORAGE variable=prn_banks type=ram_2p impl=bram
    BANK_VARIANT_LOOP: for (int v = 0; v < PRN_VARIANTS; v++) {
// LOOP_FLATTEN off : empêche HLS de fusionner les deux boucles imbriquées en une
// seule, ce qui écraserait le pipeline II=1 de BUILD_BANKS par la variabilité de v.
#pragma HLS LOOP_FLATTEN off
        BUILD_BANKS: for (int n = 0; n < 2 * N; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
            // UNROLL complet sur FINE_TAU_TILE : toutes les copies sont écrites
            // en un cycle, ce qui permet II=1 malgré FINE_TAU_TILE ports à remplir.
            TILE_COPY: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
                prn_banks[v][k][n] = prn_var[v][n];
            }
        }
    }
}

// ========== FINE PROCESSING ==========
// Corrèle le signal mélangé (mix_i / mix_q) avec UNE variante PRN sur une tuile
// de FINE_TAU_TILE phases consécutives. Produit FINE_TAU_TILE paquets de puissance
// écrits dans pow_s. Traitée séparément de process_all_variants_all_tau pour
// les cas où l'on veut une seule variante (mode legacy, non utilisé en main path).
static void process_tau_tile_variant(
    const sample_t mix_i[N],
    const sample_t mix_q[N],
    const prn_sign_t prn_bank[FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    int tau_base,
    doppler_t fd,
    variant_t variant,
    hls::stream<fine_pow_pkt_t> &pow_s
) {
#pragma HLS INLINE off
// Partitionnement complet de la dimension 1 : chaque ligne prn_bank[k] est
// accessible indépendamment, permettant FINE_TAU_TILE lectures simultanées
// en un seul cycle (nécessaire pour l'UNROLL de TAU_TILE_LOOP).
#pragma HLS ARRAY_PARTITION variable=prn_bank complete dim=1
    // Accumulateurs I et Q distincts pour chaque phase de la tuile.
    // Partitionnés complètement pour qu'ils vivent dans des registres FF indépendants
    // (pas de BRAM), évitant les conflits de port lors des mises à jour parallèles.
    acc_t accI[FINE_TAU_TILE];
    acc_t accQ[FINE_TAU_TILE];
#pragma HLS ARRAY_PARTITION variable=accI complete dim=1
#pragma HLS ARRAY_PARTITION variable=accQ complete dim=1

    // Initialisation des accumulateurs : séparé de SAMPLE_LOOP pour que HLS
    // puisse les placer dans la même région pipeline sans dépendance fictive.
    INIT_ACC: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
        accI[k] = 0;
        accQ[k] = 0;
    }

    // Boucle principale d'accumulation : un échantillon mélangé par cycle (II=1).
    // TAU_TILE_LOOP est déroulé : à chaque cycle on accumule FINE_TAU_TILE phases.
    SAMPLE_LOOP: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
        sample_t bi = mix_i[n];
        sample_t bq = mix_q[n];

        TAU_TILE_LOOP: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
            int tau = tau_base + k;
            if (tau < NB_PHASES) {
                // tau_start_tbl[tau] + n : index circulaire du chip PRN
                // correspondant au décalage τ et à l'échantillon n.
                prn_sign_t s = prn_bank[k][tau_start_tbl[tau] + n];

                // Multiplication PRN binaire par changement de signe :
                // si s=0 (PRN=-1), on inverse i_s et q_s ; si s=1 (PRN=+1), inchangé.
                // Économise 2×FINE_TAU_TILE DSP en remplacement de vraies multiplications.
                sample_t i_s = bi;
                sample_t q_s = bq;
                if (!s) {
                    i_s = -i_s;
                    q_s = -q_s;
                }

                accI[k] += i_s;
                accQ[k] += q_s;
            }
        }
    }

    // Calcul de puissance et émission dans la FIFO pow_s.
    OUT_TILE: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS PIPELINE II=1
        int tau = tau_base + k;
        if (tau < NB_PHASES) {
            fine_pow_pkt_t pkt;
            // FIX SIGNED->UNSIGNED: squarer EN SIGNÃ‰ d'abord, puis caster.
            // (power_t)accI avant multiplication réinterprete un accI négatif
            // comme une énorme valeur positive et fausse la puissance.
            // On conserve le type signé acc_t pour le carré, puis on caste le résultat
            // positif en peak_power_t non-signé : I²+Q² est toujours ≥ 0.
            acc_t iI = accI[k];
            acc_t qQ = accQ[k];
            pkt.power   = (peak_power_t)(iI * iI + qQ * qQ);
            pkt.fd      = fd;
            pkt.variant = variant;
            pkt.tau     = tau;
            pow_s.write(pkt);
        }
    }
}

// Itère process_tau_tile_variant sur toutes les phases pour une seule variante.
// Utilisée uniquement dans le chemin legacy (non appelée dans le main path fine).
static void process_variant_all_tau(
    const sample_t mix_i[N],
    const sample_t mix_q[N],
    const prn_sign_t prn_bank[FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    doppler_t fd,
    variant_t variant,
    hls::stream<fine_pow_pkt_t> &pow_s
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=prn_bank complete dim=1

    // NB_PHASES / FINE_TAU_TILE = 64 tuiles pour NB_PHASES=1024.
    TAU_BASE_LOOP: for (int tau_base = 0; tau_base < NB_PHASES; tau_base += FINE_TAU_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
        process_tau_tile_variant(mix_i, mix_q, prn_bank, tau_start_tbl, tau_base, fd, variant, pow_s);
    }
}

// Version optimisée traitant les 3 variantes PRN en une seule passe sur les données.
// Évite de relire mix_i / mix_q 3 fois : les N échantillons sont lus une seule fois
// et distribués aux 3 variantes simultanément (facteur 3 de réduction des accès BRAM).
static void process_all_variants_all_tau(
    const sample_t mix_i[N],
    const sample_t mix_q[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    int fine_tau_begin,
    int fine_tau_count,
    doppler_t fd,
    hls::stream<fine_pow_pkt_t> &pow_s
) {
#pragma HLS INLINE off
// Partitionnement complet de la dimension des bancs (16 bancs indépendants)
// pour permettre FINE_TAU_TILE accès simultanés à prn_banks[v][k][...] en un cycle.
#pragma HLS ARRAY_PARTITION variable=prn_banks complete dim=2
#pragma HLS BIND_STORAGE variable=prn_banks type=ram_2p impl=bram

    // fine_tau_count is guaranteed to be a multiple of FINE_TAU_TILE by the
    // caller (see fine_search), so there is no partial tail tile to process.
    const int full_tiles = fine_tau_count / FINE_TAU_TILE;

    // Boucle sur les tuiles complètes uniquement (pas de tuile partielle grâce
    // à l'alignement imposé par fine_search).
    TAU_BASE_LOOP_ALL_FULL: for (int t = 0; t < full_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=12 avg=8
        int tau_base = t * FINE_TAU_TILE;
        // Accumulateurs I et Q pour chaque combinaison (variante × phase dans la tuile).
        // Stockés en registres (partition complète sur les deux dimensions) pour
        // permettre 3 × 16 = 48 mises à jour parallèles par cycle de sample.
        acc_t accI[PRN_VARIANTS][FINE_TAU_TILE];
        acc_t accQ[PRN_VARIANTS][FINE_TAU_TILE];
        // Indices de début PRN pour chaque phase de la tuile, précalculés une fois
        // avant SAMPLE_LOOP_ALL pour éviter la recalculation à chaque cycle.
        int   tau_idx_tile[FINE_TAU_TILE];
#pragma HLS ARRAY_PARTITION variable=accI complete dim=2
#pragma HLS ARRAY_PARTITION variable=accQ complete dim=2
#pragma HLS ARRAY_PARTITION variable=tau_idx_tile complete dim=1

        INIT_ACC_ALL: for (int v = 0; v < PRN_VARIANTS; v++) {
#pragma HLS UNROLL
            INIT_ACC_TILE: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
                accI[v][k] = 0;
                accQ[v][k] = 0;
            }
        }

        // Précalcul des offsets : tau_idx_tile[k] = tau_start_tbl[tau_base + k + fine_tau_begin].
        // Mémorisé dans des registres (UNROLL) pour éviter un accès BRAM par cycle dans SAMPLE_LOOP.
        PREP_TAU_TILE_ALL: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
            int tau = tau_base + k + fine_tau_begin;
            tau_idx_tile[k] = tau_start_tbl[tau];
        }

        // Cœur du corrélateur : un échantillon (bi, bq) par cycle, accumulé
        // dans les 3 × 16 = 48 canaux simultanément grâce aux UNROLL imbriqués.
        SAMPLE_LOOP_ALL: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
// LOOP_FLATTEN off : interdit la fusion de SAMPLE_LOOP avec TAU_TILE_LOOP
// afin de préserver le pipeline II=1 (la fusion augmenterait le trip count
// et empêcherait HLS de planifier les accès BRAM correctement).
#pragma HLS LOOP_FLATTEN off
            sample_t bi = mix_i[n];
            sample_t bq = mix_q[n];

            TAU_TILE_LOOP_ALL: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS UNROLL
                // Lecture simultanée des 3 variantes pour la même phase k et le même n.
                // Possible car chaque prn_banks[v][k][...] est un banc BRAM indépendant.
                VARIANT_LOOP_ALL: for (int v = 0; v < PRN_VARIANTS; v++) {
                    prn_sign_t s = prn_banks[v][k][tau_idx_tile[k] + n];
                    sample_t i_s = bi;
                    sample_t q_s = bq;
                    if (!s) {
                        i_s = -i_s;
                        q_s = -q_s;
                    }
                    accI[v][k] += i_s;
                    accQ[v][k] += q_s;
                }
            }
        }

        // Émission des puissances dans pow_s après la fin de l'accumulation.
        // PIPELINE off : la boucle est courte (FINE_TAU_TILE × PRN_VARIANTS = 48 cycles max)
        // et contient des appels à pow_s.write() qui imposent des dépendances.
        STORE_TILE_ALL: for (int k = 0; k < FINE_TAU_TILE; k++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
            int tau = tau_base + k + fine_tau_begin;
            STORE_VARIANT_ALL: for (int v = 0; v < PRN_VARIANTS; v++) {
                // FIX SIGNED->UNSIGNED: squarer EN SIGNÃ‰ d'abord, puis caster.
                // Même précaution que dans process_tau_tile_variant :
                // iI et qQ restent en acc_t signé pour le carré, évitant la réinterprétation
                // d'une valeur négative en énorme valeur positive après cast prématuré.
                acc_t iI = accI[v][k];
                acc_t qQ = accQ[v][k];
                fine_pow_pkt_t pkt;
                pkt.power   = (peak_power_t)(iI * iI + qQ * qQ);
                pkt.fd      = fd;
                pkt.variant = (variant_t)v;
                pkt.tau     = tau;
                pow_s.write(pkt);
            }
        }
    }

}

// Producteur principal du pipeline fine : pour chaque bin Doppler actif,
// effectue le mélange DDS puis corrèle les 3 variantes sur toutes les phases.
// Les puissances sont empilées dans pow_s dans l'ordre (doppler, tuile, variante).
static void produce_fine_powers(
    const sample_t signal[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    const doppler_t doppler_offsets_abs[NB_DOPPLER_FINE],
    int fine_tau_begin,
    int fine_tau_count,
    int fine_doppler_count,
    hls::stream<fine_pow_pkt_t> &pow_s
) {
#pragma HLS INLINE off

    // Buffers de sortie du mélangeur : réutilisés pour chaque bin Doppler.
    // Partition cyclique factor=2 : divise le tableau en 2 banques interleaved,
    // permettant deux accès simultanés en lecture depuis process_all_variants_all_tau
    // sans conflit de port BRAM (utile quand SAMPLE_LOOP lit bi et bq en même cycle).
    sample_t mix_i[N];
    sample_t mix_q[N];
#pragma HLS ARRAY_PARTITION variable=mix_i cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=mix_q cyclic factor=2 dim=1
// ram_1p impl=bram latency=2 : un seul port de lecture, latence 2 cycles.
// Compromis : ram_2p permettrait 2 accès/cycle mais doublerait la surface BRAM.
#pragma HLS BIND_STORAGE variable=mix_i type=ram_1p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=mix_q type=ram_1p impl=bram latency=2

    // Boucle séquentielle sur les bins Doppler : le mélange doit être complet
    // avant de démarrer la corrélation (pas de DATAFLOW ici, mix_i/mix_q partagés).
    DOPPLER_LOOP: for (int d = 0; d < fine_doppler_count; d++) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=21
        doppler_t fd = doppler_offsets_abs[d];

        // Étape 1 : remplissage des buffers mix_i / mix_q pour ce bin Doppler.
        fine_mix_to_mem(signal, mix_i, mix_q, fd);
        // Étape 2 : corrélation des 3 variantes sur fine_tau_count phases,
        // résultats poussés dans pow_s pour consommation par reduce_fine_stream_core.
        process_all_variants_all_tau(mix_i, mix_q, prn_banks, tau_start_tbl, fine_tau_begin, fine_tau_count, fd, pow_s);
    }
}

// Consommateur de pow_s : calcule le maximum par variante et par bin Doppler,
// agrège la somme totale et le maximum global pour les statistiques,
// et relaie les puissances brutes vers corr_pkt_s (surface de corrélation complète).
// PIPELINE II=3 dans REDUCE_TAU : les 3 variantes arrivent groupées par tau,
// permettant 3 reads consécutifs dans un seul pipeline déroulé.
static void reduce_fine_stream_core(
    hls::stream<fine_pow_pkt_t> &pow_s,
    hls::stream<fine_max_pkt_t> &max_s,
    hls::stream<corr_pkt_t> &corr_pkt_s,
    int fine_tau_count,
    int fine_doppler_count,
    hls::stream<fine_stats_pkt_t> &stats_s
) {
#pragma HLS INLINE off

    // Narrow chirurgical pour timing closure (slack 0.03 -> ~0.5+ ns):
    // power_t<48,32> donne des comparateurs 48-bit en chaine (3x cmp + 3x add)
    // qui depassent le budget combinatoire a II=3.
    // pow_tile_t<32,28> suffit pour le chemin max: max(I^2+Q^2) fine ~ 23 bits int,
    // et p0_n/p1_n/p2_n permettent un mux/cmp 32-bit au lieu de 48.
    // sum_corr reste en power_t (overflow a partir de ~32 int bits inchange).
    pow_tile_t max_val = 0;
    power_t    sum_corr = 0;

    // Boucle externe sur les bins Doppler : réinitialise les max par variante
    // à chaque nouveau bin et émet un paquet fine_max_pkt_t à la fin du bin.
    REDUCE_D: for (int d = 0; d < fine_doppler_count; d++) {
        pow_tile_t max_power_0 = 0, max_power_1 = 0, max_power_2 = 0;
        int        phase_idx_0 = 0, phase_idx_1 = 0, phase_idx_2 = 0;
        doppler_t  fd_0 = 0,        fd_1 = 0,        fd_2 = 0;
        bool       has_0 = false,   has_1 = false,   has_2 = false;

        // Pipeline II=3 : les 3 variantes pour un même tau arrivent groupées dans pow_s
        // (ordre de production : v0, v1, v2 pour chaque tau dans STORE_VARIANT_ALL).
        // Lire 3 paquets par itération respecte exactement cet ordre.
        REDUCE_TAU: for (int tau = 0; tau < fine_tau_count; tau++) {
#pragma HLS PIPELINE II=3
            fine_pow_pkt_t pkt0 = pow_s.read();
            fine_pow_pkt_t pkt1 = pow_s.read();
            fine_pow_pkt_t pkt2 = pow_s.read();

            // Narrow local: comparateurs/mux 32-bit au lieu de 48-bit.
            // Le cast vers pow_tile_t tronque les bits de précision excédentaires
            // mais préserve toute la plage utile pour le calcul du maximum.
            pow_tile_t p0_n = (pow_tile_t)pkt0.power;
            pow_tile_t p1_n = (pow_tile_t)pkt1.power;
            pow_tile_t p2_n = (pow_tile_t)pkt2.power;

            // Mise à jour des maxima par variante avec suivi de l'index de phase
            // et du Doppler correspondant (nécessaires pour remplir fine_max_pkt_t).
            if (!has_0 || (p0_n > max_power_0)) {
                max_power_0 = p0_n;
                phase_idx_0 = pkt0.tau;
                fd_0        = pkt0.fd;
                has_0       = true;
            }
            if (!has_1 || (p1_n > max_power_1)) {
                max_power_1 = p1_n;
                phase_idx_1 = pkt1.tau;
                fd_1        = pkt1.fd;
                has_1       = true;
            }
            if (!has_2 || (p2_n > max_power_2)) {
                max_power_2 = p2_n;
                phase_idx_2 = pkt2.tau;
                fd_2        = pkt2.fd;
                has_2       = true;
            }

            // Mise à jour du maximum global toutes variantes confondues.
            pow_tile_t p01_max = (p0_n > p1_n) ? p0_n : p1_n;
            pow_tile_t p_max   = (p01_max > p2_n) ? p01_max : p2_n;
            // sum_corr accumulé en power_t (large) pour éviter le débordement
            // sur la somme de fine_tau_count × fine_doppler_count × PRN_VARIANTS termes.
            sum_corr += pkt0.power + pkt1.power + pkt2.power;
            if (p_max > max_val) {
                max_val = p_max;
            }

            // Relais vers corr_pkt_s : la surface de corrélation brute est transmise
            // à write_corr_out_stream pour export vers le PS via AXI-Stream.
            corr_pkt_t c0, c1, c2;
            c0.power = pkt0.power;
            c1.power = pkt1.power;
            c2.power = pkt2.power;
            corr_pkt_s.write(c0);
            corr_pkt_s.write(c1);
            corr_pkt_s.write(c2);
        }

        // Émission du maximum par variante à la fin de chaque bin Doppler.
        // consume_fine_maxima_and_insert appellera insert_peak sur ces paquets.
        fine_max_pkt_t out0, out1, out2;
        out0.max_power = max_power_0; out0.fd = fd_0; out0.variant = (variant_t)0; out0.phase_idx = phase_idx_0;
        out1.max_power = max_power_1; out1.fd = fd_1; out1.variant = (variant_t)1; out1.phase_idx = phase_idx_1;
        out2.max_power = max_power_2; out2.fd = fd_2; out2.variant = (variant_t)2; out2.phase_idx = phase_idx_2;
        max_s.write(out0);
        max_s.write(out1);
        max_s.write(out2);
    }

    // Émission unique des statistiques globales (max + somme) après la fin
    // de tous les bins Doppler. read_fine_stats les lira depuis stats_s.
    fine_stats_pkt_t st;
    st.max_power = (power_t)max_val;
    st.sum_power = sum_corr;
    stats_s.write(st);
}

// Transfert de la surface de corrélation brute vers le flux AXI-Stream de sortie.
// Marque le dernier échantillon avec out.last=1 pour respecter le protocole AXIS.
// La profondeur de corr_pkt_s (2048) doit absorber le délai entre produce et write
// dans le pipeline DATAFLOW pour éviter le blocage de reduce_fine_stream_core.
static void write_corr_out_stream(
    hls::stream<corr_pkt_t> &corr_pkt_s,
    int fine_tau_count,
    int fine_doppler_count,
    hls::stream<axis_t> &corr_out
) {
#pragma HLS INLINE off

    // Total de paquets = doppler_count × tau_count × 3 variantes.
    const int total = fine_doppler_count * fine_tau_count * PRN_VARIANTS;
#ifndef __SYNTHESIS__
    std::cout << "[TRACE WRITE_CORR] begin total=" << total << std::endl;
#endif
    WRITE_CORR: for (int i = 0; i < total; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1344 max=12096 avg=6720
#pragma HLS PIPELINE II=1
        corr_pkt_t pkt = corr_pkt_s.read();

        axis_t out;
        // Cast vers int : le bus AXI-Stream (axis_t.data) est un entier générique ;
        // la puissance (ap_fixed non-signé) est tronquée aux bits inférieurs.
        out.data = (int)pkt.power;
        // keep=-1 et strb=-1 : tous les octets du mot sont valides (convention AXIS).
        out.keep = -1;
        out.strb = -1;
        // Marquage AXIS last : le récepteur (PS DMA) détecte la fin du paquet.
        out.last = (i == (total - 1));
        corr_out.write(out);
#ifndef __SYNTHESIS__
        if (i == 0 || i == (total / 2) || i == (total - 1)) {
            std::cout << "[TRACE WRITE_CORR] i=" << i
                      << " power=" << pkt.power
                      << " last=" << (out.last ? 1 : 0)
                      << std::endl;
        }
#endif
    }
#ifndef __SYNTHESIS__
    std::cout << "[TRACE WRITE_CORR] end" << std::endl;
#endif
}

// Lecture unique du paquet de statistiques depuis stats_s.
// Fonction séparée pour satisfaire la contrainte DATAFLOW : chaque consommateur
// d'une FIFO doit être une fonction distincte dans la région DATAFLOW.
static void read_fine_stats(
    hls::stream<fine_stats_pkt_t> &stats_s,
    power_t &max_power,
    power_t &sum_power
) {
#pragma HLS INLINE off
    fine_stats_pkt_t st = stats_s.read();
    max_power = st.max_power;
    sum_power = st.sum_power;
}

// Consomme max_s (fine_doppler_count × PRN_VARIANTS paquets) et insère chaque
// maximum dans la liste de pics de la variante correspondante.
// Utilise une copie locale pour éviter les conflits d'accès avec le DATAFLOW,
// puis copie vers les tableaux de sortie après consommation complète.
static void consume_fine_maxima_and_insert(
    hls::stream<fine_max_pkt_t> &max_s,
    int fine_doppler_count,
    PeakInfo ListOfPeaksFS[3][MAX_NUM_PEAKS_FS],
    int num_peaks_fs[3]
) {
#pragma HLS INLINE off

    // Copies locales : évitent les alias mémoire entre la FIFO et les sorties,
    // garantissant que HLS peut analyser les dépendances correctement.
    PeakInfo ListOfPeaksFS_local[3][MAX_NUM_PEAKS_FS];
    int num_peaks_fs_local[3];

    INIT_NUM_PEAKS_FS: for (int p = 0; p < 3; p++) {
#pragma HLS PIPELINE II=1
        num_peaks_fs_local[p] = 0;
    }

    // fine_doppler_count × PRN_VARIANTS paquets au total dans max_s.
    const int total = fine_doppler_count * PRN_VARIANTS;

    CONSUME_MAX: for (int i = 0; i < total; i++) {
#pragma HLS LOOP_TRIPCOUNT min=21 max=63
// LOOP_FLATTEN off : interdit la fusion avec une boucle externe inexistante,
// mais surtout préserve l'ordonnancement séquentiel requis par insert_peak
// (dépendance RAW sur num_peaks_fs_local[p]).
#pragma HLS LOOP_FLATTEN off
        fine_max_pkt_t pkt = max_s.read();
        // Dispatch vers la liste de la variante correspondante (0, 1 ou 2).
        int p = (int)pkt.variant;
        insert_peak(
            ListOfPeaksFS_local[p],
            num_peaks_fs_local[p],
            pkt.fd,
            pkt.phase_idx,
            pkt.max_power,
            MAX_NUM_PEAKS_FS
        );
    }

    // Copie des compteurs vers les sorties (séparée pour pipeline propre).
    COPY_NUM_PEAKS_FS_OUT: for (int p = 0; p < 3; p++) {
#pragma HLS PIPELINE II=1
        num_peaks_fs[p] = num_peaks_fs_local[p];
    }

    // Copie conditionnelle des pics vers les sorties : évite d'écrire des
    // entrées non initialisées (indices au-delà de num_peaks_fs_local[p]).
    COPY_PEAKS_FS_OUT_P: for (int p = 0; p < 3; p++) {
#pragma HLS LOOP_FLATTEN off
        COPY_PEAKS_FS_OUT_I: for (int i = 0; i < MAX_NUM_PEAKS_FS; i++) {
#pragma HLS PIPELINE off
            if (i < num_peaks_fs_local[p]) {
                ListOfPeaksFS[p][i] = ListOfPeaksFS_local[p][i];
            }
        }
    }
}

// Coeur dataflow de la fine:
//   produce -> reduce -> write corr_out -> collect maxima -> stats.
// Les profondeurs de FIFO sont reglees pour garder un debit stable en HLS.
// Le DATAFLOW de Vitis HLS génère ici un pipeline de 5 tâches concurrentes :
// produce_fine_powers et reduce_fine_stream_core s'exécutent en recouvrement
// dès que pow_s contient au moins un paquet (profondeur=64 = compromis latence/BRAM).
static void run_fine_dataflow_core(
    const sample_t signal[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    const doppler_t doppler_offsets_abs[NB_DOPPLER_FINE],
    int fine_tau_begin,
    int fine_tau_count,
    int fine_doppler_count,
    PeakInfo ListOfPeaksFS_local[3][MAX_NUM_PEAKS_FS],
    int num_peaks_fs_local[3],
    hls::stream<axis_t> &corr_out,
    power_t &max_power,
    power_t &sum_power
) {
#pragma HLS INLINE off

    // pow_s : FIFO de puissances élémentaires (profondeur 64, SRL pour économiser BRAM).
    //         Producteur : produce_fine_powers. Consommateur : reduce_fine_stream_core.
    hls::stream<fine_pow_pkt_t> pow_s;
    // max_s : FIFO des maxima par variante et par Doppler (profondeur 32).
    //         Producteur : reduce_fine_stream_core. Consommateur : consume_fine_maxima_and_insert.
    hls::stream<fine_max_pkt_t> max_s;
    // corr_pkt_s : FIFO large (profondeur 2048) pour absorber le débit de la surface
    //              de corrélation complète entre reduce et write_corr_out_stream.
    //              Taille fixée pour que reduce ne se bloque pas sur l'écriture AXIS.
    hls::stream<corr_pkt_t> corr_pkt_s;
    // stats_s : FIFO de 2 entrées suffisante (1 seul paquet émis en fin de traitement).
    hls::stream<fine_stats_pkt_t> stats_s;
#pragma HLS STREAM variable=pow_s depth=64
#pragma HLS STREAM variable=max_s depth=32
#pragma HLS STREAM variable=corr_pkt_s depth=2048
#pragma HLS STREAM variable=stats_s depth=2
// SRL (Shift Register LUT) : implémentation FIFO en registres à décalage,
// économique pour les petites FIFO (≤64 entrées) vs BRAM.
#pragma HLS BIND_STORAGE variable=pow_s type=FIFO impl=SRL
#pragma HLS BIND_STORAGE variable=max_s type=FIFO impl=SRL
// DATAFLOW : les 5 fonctions ci-dessous s'exécutent en pipeline de tâches.
// Chaque fonction démarre dès que sa FIFO d'entrée contient des données,
// sans attendre que la fonction précédente soit complètement terminée.
#pragma HLS DATAFLOW

    produce_fine_powers(signal, prn_banks, tau_start_tbl, doppler_offsets_abs, fine_tau_begin, fine_tau_count, fine_doppler_count, pow_s);
    reduce_fine_stream_core(pow_s, max_s, corr_pkt_s, fine_tau_count, fine_doppler_count, stats_s);
    write_corr_out_stream(corr_pkt_s, fine_tau_count, fine_doppler_count, corr_out);
    consume_fine_maxima_and_insert(max_s, fine_doppler_count, ListOfPeaksFS_local, num_peaks_fs_local);
    read_fine_stats(stats_s, max_power, sum_power);
}

// Wrapper transparent autour de run_fine_dataflow_core : nécessaire pour que
// le pragma DATAFLOW soit appliqué à une fonction sans INLINE off au niveau appelant.
// Isole la région DATAFLOW et évite que Vitis HLS ne tente de fusionner
// les boucles extérieures de fine_search avec le pipeline interne.
static void run_fine_dataflow_region(
    const sample_t signal[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    const doppler_t doppler_offsets_abs[NB_DOPPLER_FINE],
    int fine_tau_begin,
    int fine_tau_count,
    int fine_doppler_count,
    PeakInfo ListOfPeaksFS[3][MAX_NUM_PEAKS_FS],
    int num_peaks_fs[3],
    hls::stream<axis_t> &corr_out,
    power_t &max_power,
    power_t &sum_power
) {
#pragma HLS INLINE off

    run_fine_dataflow_core(
        signal,
        prn_banks,
        tau_start_tbl,
        doppler_offsets_abs,
        fine_tau_begin,
        fine_tau_count,
        fine_doppler_count,
        ListOfPeaksFS,
        num_peaks_fs,
        corr_out,
        max_power,
        sum_power
    );
}

// Produit la grille de corrélation coarse pour UN pass STSA et UN ensemble de
// bins Doppler. Les accumulateurs accI_grid / accQ_grid sont mis à jour de façon
// incrémentale : seule la contribution des échantillons du pass courant est ajoutée,
// permettant de recalculer la puissance après chaque sous-ensemble sans recommencer
// la corrélation depuis zéro (technique STSA = Sequential Time-domain Subtraction Acquisition).
static void coarse_compute_doppler_peaks(
    const sample_t signal[N],
    const sample_t prn[N],
    const int stsa_indices[PRECISION][N / PRECISION + 1],
    const int stsa_counts[PRECISION],
    const int shifts[NB_PHASES],
    const doppler_t doppler_offsets[NB_DOPPLER_COARSE],
    int pass,
    acc_t accI_grid[NB_DOPPLER_COARSE][NB_PHASES],
    acc_t accQ_grid[NB_DOPPLER_COARSE][NB_PHASES],
    hls::stream<PeakInfo> &coarse_peak_s
) {
#pragma HLS INLINE off

    // Boucle sur tous les bins Doppler coarse (NB_DOPPLER_COARSE = 41 typiquement).
    DOPPLER_PASS_LOOP: for(int d = 0; d < NB_DOPPLER_COARSE; d++) {
#pragma HLS LOOP_TRIPCOUNT min=41 max=41
        doppler_t fd = doppler_offsets[d];
        dds_phase_t phase_inc = hz_to_phase_inc(fd);
        int M = stsa_counts[pass];
        // Phase initiale du DDS alignée sur le début du pass (pas = 0, PRECISION, 2×PRECISION…).
        // Garantit la continuité de phase par rapport aux autres passes.
        dds_phase_u_t phase_acc = (dds_phase_u_t)((long long)phase_inc * pass);
        // Pas de phase = PRECISION × phase_inc : les échantillons d'un même pass
        // sont espacés de PRECISION dans le temps, d'où un pas de phase PRECISION fois plus grand.
        dds_phase_u_t phase_step = (dds_phase_u_t)((long long)phase_inc * PRECISION);

        // Buffers temporaires pour les échantillons mélangés de ce pass.
        // Taille N/PRECISION+1 : nombre d'échantillons dans un pass (sous-échantillonnage par PRECISION).
        sample_t mix_i[N / PRECISION + 1];
        sample_t mix_q[N / PRECISION + 1];
        int mix_n[N / PRECISION + 1];

        // Remplir mix_n avec les indices du pass courant
        FILL_MIX_N: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
            mix_n[i] = stsa_indices[pass][i];
        }

        // Appel DDS partagé pour ce pass coarse
        // phase_init = phase_inc * pass (alignement de phase correct)
        // phase_step = phase_inc * PRECISION (sous-échantillonnage × 8)
        dds_mix_shared(signal, mix_n, M,
                       phase_acc, phase_step,
                       mix_i, mix_q);

        // Accumulation incrémentale sur toutes les phases τ : lecture de l'accumulateur
        // persistant, ajout de la contribution du pass, puis réécriture et calcul de puissance.
        TAU_LOOP: for(int tau = 0; tau < NB_PHASES; tau++) {
#pragma HLS LOOP_TRIPCOUNT min=1023 max=1023
#pragma HLS LOOP_FLATTEN off
            int shift = shifts[tau];
            // Version incrémentale:
            // on ajoute uniquement la contribution de la classe 'pass'
            // aux accumulateurs persistants coarse.
            acc_t I = accI_grid[d][tau];
            acc_t Q = accQ_grid[d][tau];

            // Boucle interne sur les M échantillons du pass : corrélation avec le chip PRN
            // au décalage circulaire shift[tau] (équivalent à un décalage de tau chips).
            ACC_LOOP: for(int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
                int n = mix_n[i];
                // Décalage circulaire modulo N : le PRN est périodique de longueur N.
                int prn_idx = n + shift;
                if(prn_idx >= N) prn_idx -= N;
                sample_t prn_val = prn[prn_idx];

                I += mix_i[i] * prn_val;
                Q += mix_q[i] * prn_val;
            }

            // Mise à jour des accumulateurs persistants pour le prochain pass.
            accI_grid[d][tau] = I;
            accQ_grid[d][tau] = Q;

            // Puissance instantanée après ce pass : I²+Q² avec les accumulateurs mis à jour.
            peak_power_t power = (peak_power_t)(I * I + Q * Q);

            PeakInfo cand;
            cand.doppler = fd;
            cand.phase = tau;
            cand.power = power;
            // Émission dans la FIFO pour que coarse_reduce_doppler_peaks
            // puisse traiter les pics en parallèle (DATAFLOW).
            coarse_peak_s.write(cand);
        }
    }
}

// Consomme la FIFO coarse_peak_s et maintient :
//  - le meilleur pic global (running_best) pour la détection rapide,
//  - un top-COARSE_TOPK pour le calcul du ratio p1/p_lowest après le pass.
// Le ratio lowest/highest sur le top-K espacé est une approximation de la
// variance normalisée (Eq.2 thèse) calculable sans tri complet.
static void coarse_reduce_doppler_peaks(
    hls::stream<PeakInfo> &coarse_peak_s,
    power_t &pass_best_power,
    doppler_t &pass_best_doppler,
    int &pass_best_phase,
    bool &pass_has_ratio,
    metric_t &pass_ratio,
    power_t &pass_highest,
    power_t &pass_lowest,
    int &pass_top_count
) {
#pragma HLS INLINE off

    // top_cs[0..MAX_NUM_PEAKS_FS-1] : liste des COARSE_TOPK meilleurs pics non triés.
    // Partitionné complètement pour permettre l'UNROLL de FIND_LOW_HIGH.
    PeakInfo top_cs[MAX_NUM_PEAKS_FS];
#pragma HLS ARRAY_PARTITION variable=top_cs complete dim=1
    int n_top_cs = 0;

    // Initialisation à zéro : insert_peak compare toujours à min_power,
    // donc des valeurs nulles sont immédiatement évincées par tout vrai pic.
    INIT_TOP_CS: for (int i = 0; i < MAX_NUM_PEAKS_FS; i++) {
#pragma HLS UNROLL
        top_cs[i].power = 0;
        top_cs[i].phase = 0;
        top_cs[i].doppler = 0;
    }

    peak_power_t running_best = 0;
    pass_best_power = 0;
    pass_best_doppler = 0;
    pass_best_phase = 0;
    pass_has_ratio = false;
    pass_ratio = (metric_t)1;
    pass_highest = 0;
    pass_lowest = 0;
    pass_top_count = 0;

    // Consommation de tous les NB_DOPPLER_COARSE × NB_PHASES pics de ce pass.
    // PIPELINE off : insert_peak a une dépendance RAW sur n_top_cs et top_cs[].
    REDUCE_COARSE: for (int k = 0; k < NB_DOPPLER_COARSE * NB_PHASES; k++) {
#pragma HLS PIPELINE off
        PeakInfo peak = coarse_peak_s.read();

        if (peak.power > running_best) {
            running_best = peak.power;
            pass_best_doppler = peak.doppler;
            pass_best_phase = peak.phase;
        }

        insert_peak(top_cs, n_top_cs, peak.doppler, peak.phase, peak.power, COARSE_TOPK);
    }

    pass_best_power = (power_t)running_best;

    pass_top_count = n_top_cs;

    // Calcul du ratio lowest/highest sur le top-K pour approximer la variance.
    // Nécessite au moins 2 pics distincts ; sinon pass_has_ratio reste false.
    if (n_top_cs >= 2) {
        peak_power_t highest = top_cs[0].power;
        peak_power_t lowest = top_cs[0].power;

        // Recherche du min et max sur le top-K : UNROLL complet car MAX_NUM_PEAKS_FS ≤ 16.
        FIND_LOW_HIGH: for (int i = 1; i < MAX_NUM_PEAKS_FS; i++) {
#pragma HLS UNROLL
            if (i < n_top_cs) {
                peak_power_t p = top_cs[i].power;
                if (p > highest) highest = p;
                if (p < lowest) lowest = p;
            }
        }

        pass_highest = (power_t)highest;
        pass_lowest = (power_t)lowest;
        // ratio = lowest / highest ∈ [0, 1] : proche de 1 = tous les pics similaires
        // (pas de signal dominant), proche de 0 = un pic très fort parmi des bruits faibles.
        // Ce ratio alimente le calcul de variance inter-pass dans coarse_search.
        pass_ratio = (highest > (peak_power_t)0)
                         ? (metric_t)(((power_t)lowest) / (power_t)highest)
                         : (metric_t)1;
        pass_has_ratio = true;
    }
}

// Wrapper DATAFLOW pour un pass coarse : découple le calcul de la corrélation
// (compute) et la réduction des pics (reduce) en deux tâches concurrentes.
// La FIFO coarse_peak_s (profondeur=32, SRL) découple les deux fonctions.
static void coarse_pass_dataflow_region(
    const sample_t signal[N],
    const sample_t prn[N],
    const int stsa_indices[PRECISION][N / PRECISION + 1],
    const int stsa_counts[PRECISION],
    const int shifts[NB_PHASES],
    const doppler_t doppler_offsets[NB_DOPPLER_COARSE],
    int pass,
    acc_t accI_grid[NB_DOPPLER_COARSE][NB_PHASES],
    acc_t accQ_grid[NB_DOPPLER_COARSE][NB_PHASES],
    power_t &pass_best_power,
    doppler_t &pass_best_doppler,
    int &pass_best_phase,
    bool &pass_has_ratio,
    metric_t &pass_ratio,
    power_t &pass_highest,
    power_t &pass_lowest,
    int &pass_top_count
) {
#pragma HLS INLINE off

    hls::stream<PeakInfo> coarse_peak_s;
// Profondeur 32 : compromis entre la latence de remplissage de compute et la
// capacité de drain de reduce. 32 entrées SRL ≈ 4 LUT6 par bit de largeur.
#pragma HLS STREAM variable=coarse_peak_s depth=32
#pragma HLS BIND_STORAGE variable=coarse_peak_s type=FIFO impl=SRL

#pragma HLS DATAFLOW
    coarse_compute_doppler_peaks(signal, prn, stsa_indices, stsa_counts, shifts, doppler_offsets, pass, accI_grid, accQ_grid, coarse_peak_s);
    coarse_reduce_doppler_peaks(
        coarse_peak_s,
        pass_best_power,
        pass_best_doppler,
        pass_best_phase,
        pass_has_ratio,
        pass_ratio,
        pass_highest,
        pass_lowest,
        pass_top_count
    );
}

// ========== COARSE SEARCH ==========
// Tuile coarse = même taille que fine pour partager les banks PRN (cohérence d'accès).
static const int COARSE_TAU_TILE   = FINE_TAU_TILE; // 16, partage les banks fine
// Traitement de 2 bins Doppler en parallèle : double le débit sans doubler les BRAM
// (les mix_i/mix_q sont calculés en parallèle, mais les accumulateurs tau sont partagés).
static const int COARSE_DOPPLER_PAR = 2;            // 2 Doppler traites en parallele

// Recherche coarse: balayage large Doppler/phase + metrique de variance inter-pass.
// Cette etape porte la decision de detection principale.
// Stratégie STSA : PRECISION passes sur des sous-ensembles d'échantillons complémentaires.
// Après chaque pass, on calcule le ratio p5/p1 du top-K espacé (critère thèse Eq.2).
// La variance de ces ratios inter-passes croît quand un pic dominant apparaît.
// Sortie anticipée (early exit) dès que la variance dépasse threshold_coarse.
void coarse_search(
    const sample_t signal[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    metric_t threshold_coarse,
    bool compare_mode,
    metric_t &coarse_var_out
) {
#pragma HLS INLINE off
// Limite à 1 instance de select_spaced_topk : évite la duplication de ressources
// si HLS tente d'inliner plusieurs copies dans la même région.
#pragma HLS ALLOCATION function instances=select_spaced_topk limit=1
// Partitionnement complet de la dimension des bancs PRN :
// chaque banc est un tableau indépendant, permettant COARSE_TAU_TILE accès parallèles.
#pragma HLS ARRAY_PARTITION variable=prn_banks complete dim=2
    // Recherche rapide sur grille large Doppler/phase.

    // Précalcul des indices STSA : stsa_indices[p][i] = i-ème échantillon du pass p.
    // Chaque pass p sélectionne les échantillons n = p, p+PRECISION, p+2×PRECISION, …
    // (sous-échantillonnage uniforme par facteur PRECISION = décimation temporelle).
    int stsa_indices[PRECISION][N / PRECISION + 1];
    int stsa_counts[PRECISION];

    BUILD_STSA_INDICES: for(int p = 0; p < PRECISION; p++) {
// UNROLL : PRECISION est petit (typiquement 8), l'UNROLL évite la boucle séquentielle
// et permet au compilateur de calculer tous les indices en parallèle.
#pragma HLS UNROLL
        int count = 0;
        for(int n = p; n < N; n += PRECISION) {
            stsa_indices[p][count++] = n;
        }
        stsa_counts[p] = count;
    }

    // Grille de fréquences Doppler coarse : bins uniformément espacés de DOPPLER_BIN_COARSE Hz,
    // centrés autour de 0 Hz (plage symétrique de -(NB_DOPPLER_COARSE-1)/2 à +(NB_DOPPLER_COARSE-1)/2 bins).
    doppler_t doppler_offsets[NB_DOPPLER_COARSE];
    BUILD_DOPPLER: for(int d = 0; d < NB_DOPPLER_COARSE; d++) {
#pragma HLS UNROLL
        doppler_offsets[d] = (doppler_t)(DOPPLER_BIN_COARSE * (d - (NB_DOPPLER_COARSE - 1) / 2));
    }

    // Buffer circulaire des ratios p5/p1 des PRECISION+1 derniers passes.
    // La variance de ce buffer est la métrique de détection coarse (Eq.2 thèse).
    // Initialisé avec un ratio fictif de 1.0 pour éviter un démarrage à variance nulle.
    metric_t higherLowerPeaks[PRECISION + 1];
#pragma HLS ARRAY_PARTITION variable=higherLowerPeaks complete dim=1
    INIT_RATIOS_BUF: for (int i = 0; i < PRECISION + 1; i++) {
#pragma HLS UNROLL
        higherLowerPeaks[i] = (metric_t)0;
    }
    // La première entrée est initialisée à 1 (ratio neutre) pour amorcer le calcul de variance.
    higherLowerPeaks[0] = (metric_t)1;
    int ratios_head = 1;
    int ratios_valid = 1;

    peak_power_t best_power = 0;
    doppler_t best_doppler = 0;
    int best_phase = 0;
    bool coarse_detected = false;
    coarse_var_out = 0;

    // Tableaux inter-passes pour mémoriser le meilleur pic et le top-K de chaque pass.
    // Indexés par pass pour permettre l'évaluation différée dans PASS_EVAL_LOOP.
    peak_power_t pass_best_power_arr[PRECISION];
    doppler_t pass_best_doppler_arr[PRECISION];
    int pass_best_phase_arr[PRECISION];
    PeakInfo top_cs_pass[PRECISION][MAX_NUM_PEAKS_FS];
    int n_top_cs_pass[PRECISION];

    INIT_PASS_STATE: for (int p = 0; p < PRECISION; p++) {
#pragma HLS UNROLL
        pass_best_power_arr[p] = 0;
        pass_best_doppler_arr[p] = 0;
        pass_best_phase_arr[p] = 0;
        n_top_cs_pass[p] = 0;
    }

    // Boucle externe sur des paires de Doppler.
    // Traitement par paires (COARSE_DOPPLER_PAR=2) : les deux bins Doppler partagent
    // les mêmes indices d'échantillons mix_n, réduisant de moitié les accès mémoire.
    DOPPLER_LOOP_COARSE: for (int d_base = 0; d_base < NB_DOPPLER_COARSE; d_base += COARSE_DOPPLER_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=21 max=21

        // Fréquences et incréments de phase pour les 2 slots Doppler actifs.
        // doppler_active[p] gère le cas où NB_DOPPLER_COARSE est impair
        // (le dernier groupe peut n'avoir qu'un seul slot actif).
        doppler_t    fd_arr[COARSE_DOPPLER_PAR];
        dds_phase_t  phase_inc_arr[COARSE_DOPPLER_PAR];
        bool         doppler_active[COARSE_DOPPLER_PAR];
#pragma HLS ARRAY_PARTITION variable=fd_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=phase_inc_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=doppler_active complete dim=1

        INIT_DOPPLER_SLOTS: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
            int dd = d_base + p;
            doppler_active[p] = (dd < NB_DOPPLER_COARSE);
            // idx_safe évite un accès hors-borne sur doppler_offsets si dd >= NB_DOPPLER_COARSE.
            int idx_safe = doppler_active[p] ? dd : 0;
            fd_arr[p] = doppler_offsets[idx_safe];
            phase_inc_arr[p] = hz_to_phase_inc(fd_arr[p]);
        }

        // Accumulateurs persistants inter-passes (elargis pour preserver la
        // precision du mix elargi a travers les 8 passes d'accumulation).
        // Deux tableaux (un par slot Doppler) stockés en BRAM 2-ports :
        // l'accumulateur du pass précédent est lu (port A) et mis à jour (port B) en même cycle.
        acc_coarse_t accI_tau[COARSE_DOPPLER_PAR][NB_PHASES];
        acc_coarse_t accQ_tau[COARSE_DOPPLER_PAR][NB_PHASES];
// Partition complète sur dim=1 : les deux slots Doppler ont des accumulateurs
// dans des ressources BRAM distinctes, permettant 2 accès simultanés par cycle.
#pragma HLS ARRAY_PARTITION variable=accI_tau complete dim=1
#pragma HLS ARRAY_PARTITION variable=accQ_tau complete dim=1
#pragma HLS BIND_STORAGE variable=accI_tau type=ram_2p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=accQ_tau type=ram_2p impl=bram latency=2

        // Remise à zéro des accumulateurs pour la nouvelle paire Doppler.
        // NB_PHASES itérations avec II=1 grâce à la BRAM 2-ports et la partition dim=1.
        INIT_ACC_TAU: for (int tau = 0; tau < NB_PHASES; tau++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1023 max=1023
            INIT_ACC_TAU_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                accI_tau[p][tau] = 0;
                accQ_tau[p][tau] = 0;
            }
        }

        // Boucle interne sur les PRECISION passes STSA pour cette paire Doppler.
        // Après chaque pass, les accumulateurs contiennent la corrélation partielle
        // sur (pass+1)/PRECISION de la durée totale du signal.
        PASS_LOOP_COARSE: for (int pass = 0; pass < PRECISION; pass++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            int M = stsa_counts[pass];

            // DDS phase state, un par slot Doppler.
            // Un accumulateur et un pas par slot : les deux fréquences évoluent indépendamment.
            dds_phase_u_t phase_acc_arr[COARSE_DOPPLER_PAR];
            dds_phase_u_t phase_step_arr[COARSE_DOPPLER_PAR];
#pragma HLS ARRAY_PARTITION variable=phase_acc_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=phase_step_arr complete dim=1
            INIT_DDS: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                // Phase initiale alignée sur le premier échantillon du pass (offset = pass × phase_inc).
                phase_acc_arr[p]  = (dds_phase_u_t)((long long)phase_inc_arr[p] * pass);
                // Pas inter-échantillons du pass = PRECISION × phase_inc (échantillons espacés de PRECISION).
                phase_step_arr[p] = (dds_phase_u_t)((long long)phase_inc_arr[p] * PRECISION);
            }

            // Un buffer de mix par slot Doppler, un buffer de mix_n partage (les passes
            // sous-echantillonnent le meme signal).
            // mix_i[p][i] et mix_q[p][i] : i-ème échantillon mélangé du slot p.
            // mix_n[i] : index n de cet échantillon dans signal[] (partagé entre les deux slots).
            mix_t mix_i[COARSE_DOPPLER_PAR][N / PRECISION + 1];
            mix_t mix_q[COARSE_DOPPLER_PAR][N / PRECISION + 1];
            int      mix_n[N / PRECISION + 1];
// Partition complète sur dim=1 : les deux lignes mix_i[0] et mix_i[1] sont
// dans des banques de registres distinctes, permettant 2 écritures/lectures simultanées.
#pragma HLS ARRAY_PARTITION variable=mix_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=mix_q complete dim=1
// ram_1p bram latency=2 : un port unique en lecture, 2 cycles de latence.
// Acceptable car SAMPLE_LOOP_COARSE lit mix_n[i] et mix_i[p][i] en séquentiel.
#pragma HLS BIND_STORAGE variable=mix_i type=ram_1p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=mix_q type=ram_1p impl=bram latency=2
// lutram pour mix_n : tableau d'entiers de petite taille, plus efficace en LUT
// distribuée qu'en BRAM (évite de gaspiller un BRAM18 pour 1500 entiers).
#pragma HLS BIND_STORAGE variable=mix_n type=ram_1p impl=lutram

            // Mélange DDS des M échantillons du pass, pour les 2 slots Doppler simultanément.
            MIX_SUB_PASS: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1500 max=1500
                // DEPENDENCE inter false : pas de dépendance inter-itération sur mix_i/mix_q/mix_n
                // (écriture séquentielle à des adresses croissantes i=0,1,...,M-1).
#pragma HLS DEPENDENCE variable=mix_i inter false
#pragma HLS DEPENDENCE variable=mix_q inter false
#pragma HLS DEPENDENCE variable=mix_n inter false
                int n = stsa_indices[pass][i];
                sample_t x = signal[n];
                mix_n[i] = n;
                // Calcul du mélange pour les 2 slots Doppler en parallèle (UNROLL).
                MIX_SUB_PASS_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                    unsigned lut_idx = dds_lut_index(phase_acc_arr[p]);
                    trig_t c = DDS_COS_LUT[lut_idx];
                    trig_t s = DDS_SIN_LUT[lut_idx];
                    // Multiplications DSP (latency=3) : force l'inférence sur DSP48E1
                    // et non sur LUT pour préserver le timing closure à Fmax cible.
                    mix_t xc = (mix_t)(x * c);
                    mix_t xs = (mix_t)(x * s);
#pragma HLS BIND_OP variable=xc op=mul impl=dsp latency=3
#pragma HLS BIND_OP variable=xs op=mul impl=dsp latency=3
                    mix_i[p][i] = xc;
                    mix_q[p][i] = -xs;
                    phase_acc_arr[p] = (dds_phase_u_t)(phase_acc_arr[p] + phase_step_arr[p]);
                }
            }

            // Cur tuile: la boucle SAMPLE est externe (pipeline II=1), la boucle
            // TAU_TILE et DOPPLER_PAR sont internes (UNROLL). Par tuile, on effectue
            // TILE x DOPPLER_PAR = 32 corrélations simultanees par cycle de sample.
            // Chaque itération charge une tuile de COARSE_TAU_TILE phases depuis BRAM,
            // accumule M échantillons dans les registres I/Q, puis restitue en BRAM.
            TILE_LOOP_COARSE: for (int tau_base = 0; tau_base < NB_PHASES; tau_base += COARSE_TAU_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS LOOP_FLATTEN off

                // Accumulateurs locaux (registres) pour la tuile courante.
                // Partition complète sur toutes les dimensions (dim=0) : chaque élément
                // I[p][k] vit dans un registre FF indépendant, permettant 2×16=32 mises
                // à jour simultanées dans SAMPLE_LOOP_COARSE.
                acc_coarse_t I[COARSE_DOPPLER_PAR][COARSE_TAU_TILE];
                acc_coarse_t Q[COARSE_DOPPLER_PAR][COARSE_TAU_TILE];
                // Index de début PRN pour chaque phase de la tuile (précalculé).
                int   tau_idx_tile[COARSE_TAU_TILE];
                // Adresse τ absolue pour le stockage BRAM (précalculée pour casser le sparsemux).
                int   tau_addr_tile[COARSE_TAU_TILE];  // Adresse tau precalculee pour le stockage.
                // Masque de validité : les phases de la tuile de queue peuvent dépasser NB_PHASES.
                bool  tau_valid[COARSE_TAU_TILE];
#pragma HLS ARRAY_PARTITION variable=I complete dim=0
#pragma HLS ARRAY_PARTITION variable=Q complete dim=0
#pragma HLS ARRAY_PARTITION variable=tau_idx_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=tau_addr_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=tau_valid complete dim=1

                // Branche FULL : toutes les phases [tau_base, tau_base+TILE-1] sont valides.
                // Les indices sont constants à la compilation -> pas de sparsemux dans le chargement.
                bool full_tau_tile_load = (tau_base <= (NB_PHASES - COARSE_TAU_TILE));
                if (full_tau_tile_load) {
                    // Chargement des accumulateurs persistants depuis BRAM vers registres locaux.
                    // II=1 possible car accI_tau est une BRAM 2-ports avec partition dim=1.
                    LOAD_ACC_TILE_FULL: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS PIPELINE II=1
                        int tau = tau_base + k;
                        tau_valid[k] = true;
                        tau_addr_tile[k] = tau;
                        tau_idx_tile[k] = tau_start_tbl[tau];
                        LOAD_ACC_TILE_FULL_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                            I[p][k] = accI_tau[p][tau];
                            Q[p][k] = accQ_tau[p][tau];
                        }
                    }
                } else {
                    // Branche TAIL: indices constants pour casser le sparsemux,
                    // meme strategie que STORE_ACC_TILE_TAIL.
                    // NB_PHASES=1023 (non multiple de 16) -> TAIL_VALID_LD=15 phases valides,
                    // TAIL_BASE_LD=1008 : les 15 dernières phases commencent à l'index 1008.
                    const int TAIL_VALID_LD = NB_PHASES - (NB_PHASES / COARSE_TAU_TILE) * COARSE_TAU_TILE; // 15
                    const int TAIL_BASE_LD  = NB_PHASES - TAIL_VALID_LD;                                    // 1008
                    // UNROLL complet : les 16 phases sont toutes connues à la compilation
                    // (tau_s = constante), empêchant HLS de générer un sparsemux dynamique.
                    LOAD_ACC_TILE_TAIL: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS UNROLL
                        const bool valid = (k < TAIL_VALID_LD);
                        const int  tau_s = TAIL_BASE_LD + k;
                        tau_valid[k] = valid;
                        tau_addr_tile[k] = tau_s;
                        tau_idx_tile[k] = valid ? tau_start_tbl[tau_s] : 0;
                        LOAD_ACC_TILE_TAIL_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                            I[p][k] = valid ? accI_tau[p][tau_s] : (acc_coarse_t)0;
                            Q[p][k] = valid ? accQ_tau[p][tau_s] : (acc_coarse_t)0;
                        }
                    }
                }

                // Cœur du corrélateur tuilé : II=1, 32 accumulations par cycle.
                // Pour chaque échantillon i, on met à jour les 2 slots × 16 phases simultanément.
                SAMPLE_LOOP_COARSE: for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1500 max=1500
                    int n_i = mix_n[i];
                    // Boucle sur les 16 phases de la tuile (UNROLL) :
                    // prn_banks[0][k][tau_idx_tile[k]+n_i] donne le chip PRN pour la phase k.
                    // Variante 0 uniquement en coarse : la décision se fait sur la variance
                    // inter-passes, pas sur la discrimination sub-chip (réservée à la fine search).
                    SAMPLE_TILE_LOOP: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS UNROLL
                        prn_sign_t s = prn_banks[0][k][tau_idx_tile[k] + n_i];
                        // Application du signe PRN aux deux slots Doppler (UNROLL).
                        SAMPLE_DOPPLER_LOOP: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                            mix_t i_s = mix_i[p][i];
                            mix_t q_s = mix_q[p][i];
                            if (!s) {
                                i_s = -i_s;
                                q_s = -q_s;
                            }
                            // Guard sur tau_valid : les phases de la tuile de queue hors NB_PHASES
                            // ne doivent pas être accumulées (résultats indéfinis).
                            if (tau_valid[k]) {
                                I[p][k] += i_s;
                                Q[p][k] += q_s;
                            }
                        }
                    }
                }

                // Restitution des accumulateurs mis à jour depuis les registres vers BRAM.
                // Branche FULL : écriture séquentielle II=1 avec déroulage implicite des 2 slots.
                bool full_tau_tile = (tau_base <= (NB_PHASES - COARSE_TAU_TILE));
                if (full_tau_tile) {
                    STORE_ACC_TILE_FULL: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS PIPELINE II=1
                        int tau_s = tau_addr_tile[k];
                        accI_tau[0][tau_s] = I[0][k];
                        accQ_tau[0][tau_s] = Q[0][k];
                        accI_tau[1][tau_s] = I[1][k];
                        accQ_tau[1][tau_s] = Q[1][k];
                    }
                } else {
                    // Branche de fin de tuile: ecriture des phases restantes.
                    // UNROLL avec constante tau_s : chaque écriture cible une adresse BRAM
                    // fixe connue à la compilation, évitant le sparsemux et garantissant II=1.
                    const int TAIL_VALID = NB_PHASES - (NB_PHASES / COARSE_TAU_TILE) * COARSE_TAU_TILE; // 15
                    const int TAIL_BASE  = NB_PHASES - TAIL_VALID;                                        // 1008
                    STORE_ACC_TILE_TAIL: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS UNROLL
                        if (k < TAIL_VALID) {
                            const int tau_s = TAIL_BASE + k;
                            accI_tau[0][tau_s] = I[0][k];
                            accQ_tau[0][tau_s] = Q[0][k];
                            accI_tau[1][tau_s] = I[1][k];
                            accQ_tau[1][tau_s] = Q[1][k];
                        }
                    }
                }

                // Calcul de puissance I²+Q² pour chaque cellule (p, k) de la tuile.
                // Narrow cast avant le carré : réduit la multiplication de 32×32 (4 DSP)
                // à 24×24 (2 DSP) en tronquant les bits de précision excédentaires.
                pow_tile_t pow_tile[COARSE_DOPPLER_PAR][COARSE_TAU_TILE];
#pragma HLS ARRAY_PARTITION variable=pow_tile complete dim=1
                COMPUTE_POW_TILE: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS UNROLL
                    COMPUTE_POW_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
#pragma HLS UNROLL
                        // Narrow uniquement pour le carre: 32x32 (4 DSP) -> 24x24 (2 DSP).
                        // Acc. preservee en <32,14>; cast = simple selection de bits.
                        acc_coarse_mul_t iI_n = (acc_coarse_mul_t)I[p][k];
                        acc_coarse_mul_t qQ_n = (acc_coarse_mul_t)Q[p][k];
                        pow_tile_t i_sq = (pow_tile_t)(iI_n * iI_n);
                        pow_tile_t q_sq = (pow_tile_t)(qQ_n * qQ_n);
// Latence max DSP mul valide Vitis 2023.2 / Zynq-7020 = 4 (voir HLS 214-383).
#pragma HLS BIND_OP variable=i_sq op=mul impl=dsp latency=4
#pragma HLS BIND_OP variable=q_sq op=mul impl=dsp latency=4
                        pow_tile[p][k] = i_sq + q_sq;
                    }
                }

                // insert_peak a une dependance read-modify-write sur top_cs_pass[pass];
                // on sequence les k et les p, HLS choisira le meilleur II (~1-2).
                // Mise à jour du meilleur pic global du pass et du top-COARSE_TOPK.
                INSERT_PEAK_TILE: for (int k = 0; k < COARSE_TAU_TILE; k++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                    int tau = tau_base + k;
                    if (tau >= NB_PHASES) continue;
                    INSERT_PEAK_P: for (int p = 0; p < COARSE_DOPPLER_PAR; p++) {
                        if (!doppler_active[p]) continue;
                        peak_power_t power = pow_tile[p][k];
                        if (power > pass_best_power_arr[pass]) {
                            pass_best_power_arr[pass] = power;
                            pass_best_doppler_arr[pass] = fd_arr[p];
                            pass_best_phase_arr[pass] = tau;
                        }
                        insert_peak(
                            top_cs_pass[pass],
                            n_top_cs_pass[pass],
                            fd_arr[p],
                            tau,
                            power,
                            COARSE_TOPK
                        );
                    }
                }
            }
        }
    }

    // Évaluation des passes après la fin des boucles Doppler/pass :
    // calcul du ratio espacé p5/p1, mise à jour du buffer circulaire de ratios,
    // calcul de la variance inter-passes et décision de détection.
    PASS_EVAL_LOOP: for(int pass = 0; pass < PRECISION; pass++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
// PIPELINE off : cette boucle contient des appels à select_spaced_topk (non inlinable)
// et des mises à jour conditionnelles de coarse_var_out (dépendances RAW complexes).
#pragma HLS PIPELINE off
#ifndef __SYNTHESIS__
    std::cout << "[PASS_TRACE] pass=" << pass
              << " coarse_var_out=" << coarse_var_out
              << " threshold=" << threshold_coarse
              << " detected=" << (coarse_var_out > threshold_coarse ? 1 : 0)
              << std::endl;
#endif
        peak_power_t pass_best_power = pass_best_power_arr[pass];
        doppler_t pass_best_doppler = pass_best_doppler_arr[pass];
        int pass_best_phase = pass_best_phase_arr[pass];
        int pass_top_count = n_top_cs_pass[pass];
        bool pass_has_ratio = false;
        metric_t pass_ratio = (metric_t)1;
        peak_power_t pass_highest = 0;
        peak_power_t pass_lowest = 0;

        // Calcul du ratio p5/p1 uniquement si le top-K contient assez de pics espacés.
        if (pass_top_count >= CS_REF_RANK) {
            PeakInfo spaced_cs[MAX_NUM_PEAKS_FS];
// Partition complète pour permettre l'accès simultané à tous les éléments
// dans select_spaced_topk (dépendances CHECK_SEP_LOOP sur out_spaced[j]).
#pragma HLS ARRAY_PARTITION variable=spaced_cs complete dim=1
            int n_spaced_cs = 0;
            // Sélection des CS_REF_RANK meilleurs pics espacés (séparation MIN_PEAK_SEP en phase
            // et MIN_DOPPLER_SEP_COARSE en Doppler) pour calculer le ratio p1/p5 sans biais
            // dû aux pics voisins du pic principal.
            select_spaced_topk(
                top_cs_pass[pass],
                pass_top_count,
                CS_REF_RANK,
                MIN_PEAK_SEP,
                MIN_DOPPLER_SEP_COARSE,
                spaced_cs,
                n_spaced_cs
            );
#ifndef __SYNTHESIS__
            // Top-5 espaces (critere these p1..p5) sur les 3 derniers pass — debug csim uniquement.
            if (pass >= PRECISION - 3) {
                const int RAW_K = COARSE_TOPK < 8 ? COARSE_TOPK : 8;
                std::cout << "[DBG COARSE top/spaced] pass=" << pass
                          << " raw_top_count=" << pass_top_count
                          << " min_phase_sep=" << MIN_PEAK_SEP
                          << " min_fd_sep_Hz=" << MIN_DOPPLER_SEP_COARSE
                          << std::endl;
                std::cout << "  raw top-K (ordre insert_peak, jusqu'a " << RAW_K << "):" << std::endl;
                for (int j = 0; j < RAW_K && j < pass_top_count; ++j) {
                    std::cout << "    raw[" << j << "] fd_hz=" << (int)top_cs_pass[pass][j].doppler
                              << " phase=" << top_cs_pass[pass][j].phase
                              << " power=" << top_cs_pass[pass][j].power << std::endl;
                }
                std::cout << "  apres select_spaced_topk: n_spaced=" << n_spaced_cs
                          << " (besoin " << CS_REF_RANK << " pour ratio p5/p1)" << std::endl;
                for (int j = 0; j < CS_REF_RANK; ++j) {
                    if (j < n_spaced_cs) {
                        std::cout << "    spaced[" << j << "] fd_hz="
                                  << (int)spaced_cs[j].doppler
                                  << " phase=" << spaced_cs[j].phase
                                  << " power=" << spaced_cs[j].power << std::endl;
                    } else {
                        std::cout << "    spaced[" << j << "] --- (absent)" << std::endl;
                    }
                }
            }
#endif
            if (n_spaced_cs >= CS_REF_RANK) {
                // spaced_cs n'est pas trié : highest et lowest ne sont pas nécessairement
                // aux extrémités ; on pourrait les trouver par un min/max linéaire.
                // Ici on utilise les extrémités spaced_cs[0] et spaced_cs[CS_REF_RANK-1]
                // car select_spaced_topk retourne les pics dans l'ordre de puissance décroissante.
                pass_highest = spaced_cs[0].power;
                pass_lowest  = spaced_cs[CS_REF_RANK - 1].power;
                pass_ratio = (pass_highest > (peak_power_t)0)
                                 ? (metric_t)(((power_t)pass_lowest) / (power_t)pass_highest)
                                 : (metric_t)1;
                pass_has_ratio = true;
            }
        } else {
#ifndef __SYNTHESIS__
            if (pass >= PRECISION - 3) {
                std::cout << "[DBG COARSE top/spaced] pass=" << pass
                          << " raw_top_count=" << pass_top_count
                          << " (insuffisant pour select_spaced_topk, besoin >= "
                          << CS_REF_RANK << ")" << std::endl;
            }
#endif
        }

#ifndef __SYNTHESIS__
        dbg_cs_p1 = (power_t)pass_highest;
        dbg_cs_p5 = (power_t)pass_lowest;
        dbg_cs_top_count = pass_top_count;
#endif

        // Mise à jour du meilleur pic global toutes passes confondues.
        if (pass_best_power > best_power) {
            best_power = pass_best_power;
            best_doppler = pass_best_doppler;
            best_phase = pass_best_phase;
        }

        if (pass_has_ratio) {
            // Insertion du ratio du pass courant dans le buffer circulaire (taille PRECISION+1).
            // Le buffer circulaire évite de décaler tous les éléments à chaque pass.
            higherLowerPeaks[ratios_head] = pass_ratio;
            ratios_head++;
            if (ratios_head >= (PRECISION + 1)) ratios_head = 0;
            if (ratios_valid < (PRECISION + 1)) ratios_valid++;

            // Solution A: variance coarse en metric_acc_t (ap_fixed<40,8> pour
            // ratios_valid jusqu'a PRECISION+1 sans wrap du diviseur).
            // Calcul de la variance empirique des ratios sur la fenêtre courante :
            // mean = (1/N) Σ ratios[i], var = (1/N) Σ (ratios[i] - mean)².
            metric_acc_t mean = 0;
            // UNROLL sur PRECISION+1 = 9 itérations : génère une somme parallèle en arbre
            // (3 niveaux de LUT) au lieu d'une accumulation séquentielle de 9 cycles.
            MEAN_LOOP: for (int i = 0; i < PRECISION + 1; i++) {
#pragma HLS UNROLL
                if (i < ratios_valid) mean += (metric_acc_t)higherLowerPeaks[i];
            }
            mean /= (metric_acc_t)ratios_valid;

            metric_acc_t diff_buf[PRECISION + 1];
            metric_acc_t sq_buf[PRECISION + 1];
#pragma HLS ARRAY_PARTITION variable=diff_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=sq_buf complete dim=1

            // Calcul des écarts à la moyenne : mis à zéro pour les indices non valides
            // afin de ne pas biaiser la somme des carrés.
            DIFF_LOOP: for (int i = 0; i < PRECISION + 1; i++) {
#pragma HLS UNROLL
                metric_acc_t d = (metric_acc_t)higherLowerPeaks[i] - mean;
                diff_buf[i] = (i < ratios_valid) ? d : (metric_acc_t)0;
            }

            SQ_LOOP: for (int i = 0; i < PRECISION + 1; i++) {
#pragma HLS UNROLL
                // Indice "i" devient constant a la compilation -> pas de sparsemux,
                // chemin critique = juste mul 32x32 (PRECISION+1 mul DSP en parallele).
                // Chaque carré est mappé sur un DSP48E1 indépendant (PRECISION+1 = 9 DSP).
                metric_sq_lane_t dn = (metric_sq_lane_t)diff_buf[i];
                metric_sq_lane_t sq_n = dn * dn;
#pragma HLS BIND_OP variable=sq_n op=mul impl=dsp latency=4
                sq_buf[i] = (metric_acc_t)sq_n;
            }

            metric_acc_t var = 0;
            // Somme parallèle en arbre des carrés (PRECISION+1 termes, UNROLL complet).
            ACCUM_LOOP: for (int i = 0; i < PRECISION + 1; i++) {
#pragma HLS UNROLL
                var += sq_buf[i];
            }
            var /= (metric_acc_t)ratios_valid;
            // Clip à 0 : var est en metric_acc_t signé pour absorber les écarts négatifs
            // intermédiaires, mais la variance finale est toujours positive.
            coarse_var_out = (var > (metric_acc_t)0) ? (metric_t)var : (metric_t)0;
        }

        // Décision de détection : si la variance dépasse le seuil, un pic dominant
        // a été détecté. En mode normal (!compare_mode), on sort immédiatement (early exit)
        // pour éviter les passes restantes inutiles et réduire la latence.
        if (coarse_var_out > threshold_coarse) {
            doppler_out = (int)best_doppler;
            phase_out = best_phase;
            peak_out = (power_t)best_power;
            detected_out = true;
            coarse_detected = true;
#ifndef __SYNTHESIS__
            std::cout << "[EARLY_EXIT] Quitting at pass " << pass << std::endl;
#endif
            // compare_mode=true : on continue malgré la détection pour évaluer tous les passes
            // (utile pour les tests de performance et la comparaison des métriques).
            if(!compare_mode) return;
        }
    }

    // Mise à jour des sorties avec le meilleur pic global (toutes passes).
    // Si aucun pass n'a déclenché la détection, detected_out reste false.
    doppler_out = (int)best_doppler;
    phase_out = best_phase;
    peak_out = (power_t)best_power;
    detected_out = coarse_detected;
}

// ========== FINE SEARCH ==========
// Affine l'estimation Doppler et phase autour du résultat coarse.
// Politique adaptative : la fenêtre fine (doppler_count × tau_count) est réduite
// proportionnellement à la confiance coarse (coarse_var_in / threshold_coarse_in).
// Plus la coarse est confiante, plus la fine est étroite → latence réduite.
void fine_search(
    const sample_t signal[N],
    const prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N],
    const int tau_start_tbl[NB_PHASES],
    doppler_t doppler_coarse,
    int &doppler_out,
    int &phase_out,
    power_t &peak_out,
    bool &detected_out,
    metric_t threshold_fine,
    metric_t coarse_var_in,
    metric_t threshold_coarse_in,
    metric_t &fine_var_out,
    power_t &peak_fine_out,
    int &fine_doppler_count_out,
    int &fine_tau_count_out,
    hls::stream<axis_t> &corr_out,
    power_t &max_power,
    power_t &sum_power
) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=select_spaced_topk limit=1
    // Raffine autour du meilleur Doppler/phase coarse.

#ifndef __SYNTHESIS__
    std::cout << "[TRACE FINE] enter coarse_doppler=" << (int)doppler_coarse << std::endl;
#endif

    // Politique adaptative resserree (T3).
    // La coarse balaye deja toutes les phases (NB_PHASES), donc la fine
    // n'a pas besoin d'une fenetre tau tres large pour se recaler; on garde
    // une marge conservative mais fortement plafonnee pour reduire la latence.
    // Valeurs par défaut (confiance coarse faible ou nulle) : fenêtre maximale.
    int fine_doppler_count = NB_DOPPLER_FINE;
    int fine_tau_count     = 192;
    if (threshold_coarse_in > (metric_t)0) {
        // Trois niveaux de confiance : ultra-haute (3×seuil), haute (2×seuil), moyenne (1.5×seuil).
        // À chaque niveau, la fenêtre fine est réduite pour limiter la latence du DATAFLOW.
        metric_t ultra_high_conf = threshold_coarse_in * (metric_t)3;
        metric_t high_conf       = threshold_coarse_in * (metric_t)2;
        metric_t mid_conf        = threshold_coarse_in + (threshold_coarse_in / (metric_t)2);
        if (coarse_var_in >= ultra_high_conf) {
            // Confiance très haute : fenêtre minimale (7 bins × 64 phases).
            fine_doppler_count = 7;
            fine_tau_count     = 64;
        } else if (coarse_var_in >= high_conf) {
            fine_doppler_count = 9;
            fine_tau_count     = 96;
        } else if (coarse_var_in >= mid_conf) {
            fine_doppler_count = 11;
            fine_tau_count     = 128;
        }
    }
    fine_doppler_count_out = fine_doppler_count;
    if (fine_tau_count < 1) fine_tau_count = 1;
    if (fine_tau_count > NB_PHASES) fine_tau_count = NB_PHASES;
    // Align fine_tau_count down to a multiple of FINE_TAU_TILE(=16). This
    // eliminates the partial tail tile in process_all_variants_all_tau, which
    // Alignement vers le bas au multiple inférieur de FINE_TAU_TILE :
    // garantit qu'il n'y a pas de tuile partielle dans process_all_variants_all_tau,
    // ce qui simplifie le contrôle de boucle et améliore le timing HLS.
    {
        int aligned = (fine_tau_count / FINE_TAU_TILE) * FINE_TAU_TILE;
        // Assure au moins une tuile complète même si fine_tau_count était < FINE_TAU_TILE.
        if (aligned < FINE_TAU_TILE) aligned = FINE_TAU_TILE;
        fine_tau_count = aligned;
    }
    fine_tau_count_out = fine_tau_count;
    // Centrage de la fenêtre tau autour du pic coarse, avec clipping aux bornes [0, NB_PHASES-fine_tau_count].
    int phase_center = phase_out;
    int fine_tau_begin = phase_center - (fine_tau_count / 2);
    if (fine_tau_begin < 0) fine_tau_begin = 0;
    int max_begin = NB_PHASES - fine_tau_count;
    if (fine_tau_begin > max_begin) fine_tau_begin = max_begin;

    // Construction de la grille Doppler fine : bins uniformément espacés de DOPPLER_BIN_FINE Hz,
    // centrés sur doppler_coarse (résultat de la coarse search).
    doppler_t doppler_offsets_abs[NB_DOPPLER_FINE];
    int center = NB_DOPPLER_FINE / 2;
    int active_half = fine_doppler_count / 2;
    int first_idx = center - active_half;

    // PIPELINE off : la boucle est courte et contient un cast doppler_t qui peut
    // ne pas être pipelinable à II=1 selon le type ap_fixed utilisé.
    BUILD_FINE_DOPPLER: for(int d = 0; d < fine_doppler_count; d++) {
#pragma HLS PIPELINE off
        int full_idx = first_idx + d;
        // doppler_offsets_abs[d] = doppler_coarse + offset relatif en Hz.
        doppler_offsets_abs[d] = doppler_coarse + (doppler_t)((full_idx - center) * DOPPLER_BIN_FINE);
    }

    // Tableaux de pics par variante PRN, remplis par le DATAFLOW fine.
    PeakInfo ListOfPeaksFS[3][MAX_NUM_PEAKS_FS];
    int num_peaks_fs[3];

    // Lancement du DATAFLOW fine : produce → reduce → corr_out → maxima → stats.
    // À la sortie, ListOfPeaksFS[v] contient le top-MAX_NUM_PEAKS_FS des pics
    // de la variante v, et max_power / sum_power contiennent les statistiques globales.
    run_fine_dataflow_region(
        signal,
        prn_banks,
        tau_start_tbl,
        doppler_offsets_abs,
        fine_tau_begin,
        fine_tau_count,
        fine_doppler_count,
        ListOfPeaksFS,
        num_peaks_fs,
        corr_out,
        max_power,
        sum_power
    );

#ifndef __SYNTHESIS__
    std::cout << "[TRACE FINE] after dataflow max_power=" << max_power
              << " sum_power=" << sum_power
              << " fine_bins=" << fine_doppler_count
              << " fine_tau=" << fine_tau_count
              << " peaks_count={" << num_peaks_fs[0] << "," << num_peaks_fs[1] << "," << num_peaks_fs[2] << "}"
              << std::endl;
#endif

    // Pic de puissance maximale et minimale (rang FS_REF_RANK) pour chaque variante.
    // Utilisés pour calculer la variance normalisée nvar(p1, p_FS_REF_RANK).
    PeakInfo highest_peak[3];
    PeakInfo lowest_peak[3];
    bool has_peak[3];
#pragma HLS ARRAY_PARTITION variable=highest_peak complete dim=1
#pragma HLS ARRAY_PARTITION variable=lowest_peak complete dim=1
#pragma HLS ARRAY_PARTITION variable=has_peak complete dim=1

    // Initialisation des structures de résultat avec des valeurs neutres.
    // Le doppler est initialisé à doppler_coarse pour garantir une valeur cohérente
    // même si la fine ne trouve pas de pic valide.
    INIT_VARIANT_STATS: for (int p = 0; p < 3; p++) {
#pragma HLS UNROLL
        highest_peak[p].power = 0;
        highest_peak[p].phase = 0;
        highest_peak[p].doppler = doppler_coarse;
        lowest_peak[p].power = 0;
        lowest_peak[p].phase = 0;
        lowest_peak[p].doppler = doppler_coarse;
        has_peak[p] = false;
    }

    // Selection p1 / p_FS_REF_RANK avec separation (phase + doppler).
    // Pour chaque variante, on extrait les FS_REF_RANK meilleurs pics espacés
    // afin de calculer la variance normalisée sans biais dû aux pics voisins.
    COLLECT_VARIANT_STATS_P: for (int p = 0; p < 3; p++) {
#pragma HLS LOOP_FLATTEN off
        PeakInfo spaced_fs[MAX_NUM_PEAKS_FS];
#pragma HLS ARRAY_PARTITION variable=spaced_fs complete dim=1
        int n_spaced_fs = 0;
        select_spaced_topk(
            ListOfPeaksFS[p],
            num_peaks_fs[p],
            FS_REF_RANK,
            MIN_PEAK_SEP,
            MIN_DOPPLER_SEP_FINE,
            spaced_fs,
            n_spaced_fs
        );
        if (n_spaced_fs >= FS_REF_RANK) {
            // spaced_fs est trié par puissance décroissante par select_spaced_topk.
            highest_peak[p] = spaced_fs[0];
            lowest_peak[p]  = spaced_fs[FS_REF_RANK - 1];
            has_peak[p]     = true;
        }
    }

    // best_peak : pic de la variante ayant la plus grande variance normalisée (meilleure discrimination).
    // global_peak : pic de puissance absolue maximale toutes variantes confondues.
    // Les deux peuvent différer si une variante à forte puissance a une faible variance
    // (tous ses pics sont comparables → signal absent ou bruit fort).
    PeakInfo best_peak;
    best_peak.power = 0;
    best_peak.phase = 0;
    best_peak.doppler = doppler_coarse;

    PeakInfo global_peak;
    global_peak.power = 0;
    global_peak.phase = 0;
    global_peak.doppler = doppler_coarse;
    int global_peak_variant = -1;

    metric_t higher_var = 0;
    int best_var_variant = -1;

    // Sélection de la meilleure variante par variance normalisée (critère principal thèse)
    // et du pic global de puissance maximale (utilisé pour le garde-fou global_guard).
    VARIANT_SCORE_LOOP: for (int p = 0; p < 3; p++) {
#pragma HLS UNROLL
        if (has_peak[p]) {
            metric_t var_p = nvar_from_p1_pk((power_t)highest_peak[p].power, (power_t)lowest_peak[p].power);
            if ((best_var_variant < 0) || (var_p > higher_var)) {
                higher_var = var_p;
                best_var_variant = p;
            }
            if ((global_peak_variant < 0) || (highest_peak[p].power > global_peak.power)) {
                global_peak = highest_peak[p];
                global_peak_variant = p;
            }
        }
    }

    if (best_var_variant >= 0) {
        best_peak = highest_peak[best_var_variant];
    }

    // Calcul de la dominance locale : ratio entre le pic principal et son meilleur
    // concurrent dans la fenêtre (FS_LOCAL_PHASE_WIN, FS_LOCAL_DOPPLER_WIN).
    // Une dominance >= 1.0 signifie que le pic est au moins aussi fort que son voisin
    // (condition nécessaire mais non suffisante pour la détection).
    peak_power_t local_comp_best = 0;
    peak_power_t local_comp_global = 0;

    LOCAL_COMP_P: for (int p = 0; p < 3; p++) {
        LOCAL_COMP_I: for (int i = 0; i < num_peaks_fs[p]; i++) {
#pragma HLS PIPELINE off
            PeakInfo cand = ListOfPeaksFS[p][i];

            // Exclusion du pic lui-même (comparaison exacte en triplet doppler/phase/power).
            bool same_best = (cand.phase == best_peak.phase) && ((int)cand.doppler == (int)best_peak.doppler) && (cand.power == best_peak.power);
            if (!same_best && is_local_neighbor(best_peak, cand, FS_LOCAL_PHASE_WIN, FS_LOCAL_DOPPLER_WIN)) {
                if (cand.power > local_comp_best) local_comp_best = cand.power;
            }

            bool same_global = (cand.phase == global_peak.phase) && ((int)cand.doppler == (int)global_peak.doppler) && (cand.power == global_peak.power);
            if (!same_global && is_local_neighbor(global_peak, cand, FS_LOCAL_PHASE_WIN, FS_LOCAL_DOPPLER_WIN)) {
                if (cand.power > local_comp_global) local_comp_global = cand.power;
            }
        }
    }

    // Dominance = pic / concurrent_local ; si aucun concurrent, on divise par 1
    // (dominance = puissance du pic, non bornée) pour éviter une division par zéro.
    metric_t dominance_best = (best_peak.power > (peak_power_t)0)
                                  ? (metric_t)(((power_t)best_peak.power)
                                      / ((local_comp_best > (peak_power_t)0) ? (power_t)local_comp_best : (power_t)1))
                                  : (metric_t)0;
    metric_t dominance_global = (global_peak.power > (peak_power_t)0)
                                    ? (metric_t)(((power_t)global_peak.power)
                                        / ((local_comp_global > (peak_power_t)0) ? (power_t)local_comp_global : (power_t)1))
                                    : (metric_t)0;

    // Deux niveaux de seuil de variance fine : haut (threshold_fine) et bas (85% de threshold_fine).
    // Le seuil bas est utilisé dans le fallback borderline de acquisition_stsa.
    metric_t threshold_high = threshold_fine;
    metric_t threshold_low = threshold_fine * FS_THRESHOLD_LOW_SCALE;

    bool var_pass_high = (higher_var > threshold_high);
    bool var_pass_low = (higher_var > threshold_low);
    bool dom_pass_best = (dominance_best >= FS_LOCAL_DOMINANCE_MIN);
    bool dom_pass_global = (dominance_global >= FS_LOCAL_DOMINANCE_MIN);
    // global_guard : actif si le pic global (forte puissance) appartient à une variante différente
    // de la variante de meilleure variance ET dépasse 1.2× le pic de meilleure variance.
    // Traduit une ambiguïté inter-variante : le pic le plus fort n'est pas le plus discriminant.
    bool global_guard = (best_var_variant >= 0) && (global_peak_variant >= 0) &&
                        (global_peak_variant != best_var_variant) &&
                        (global_peak.power > (peak_power_t)((best_peak.power * FS_GLOBAL_PEAK_GUARD_RATIO)));

    // ==========================================================
    // Sauvegarde des résultats coarse pour fallback si la fine échoue.
    int     coarse_doppler_saved = doppler_out;
    int     coarse_phase_saved   = phase_out;
    power_t coarse_peak_saved    = peak_out;

    // Candidat de raffinage: on privilegie global_peak (plus haute puissance),
    // mais on bascule sur best_peak si le global_guard est actif et que la
    // dominance best est plus sure que global.
    PeakInfo chosen_peak = best_peak;
    bool has_fine_candidate = (best_var_variant >= 0) || (global_peak_variant >= 0);
    if (global_peak_variant >= 0) {
        if (global_guard && dom_pass_best && !dom_pass_global) {
            // Conflit inter-variante résolu en faveur du pic de meilleure variance.
            chosen_peak = best_peak;
        } else {
            // Cas nominal : on retient le pic de puissance maximale.
            chosen_peak = global_peak;
        }
    }

    // Cohérence coarse/fine : vérifie que le pic fine retenu est dans les marges
    // FINE_REFINE_DOPPLER_MAX et FINE_REFINE_PHASE_MAX autour de l'estimation coarse.
    // Un pic fine incohérent indique une fausse alarme ou une instabilité numérique.
    int fine_dopp_dist  = doppler_abs_dist(chosen_peak.doppler, (doppler_t)coarse_doppler_saved);
    int fine_phase_dist = phase_circular_dist(chosen_peak.phase, coarse_phase_saved);
    bool coherent = has_fine_candidate &&
                    (fine_dopp_dist  <= FINE_REFINE_DOPPLER_MAX) &&
                    (fine_phase_dist <= FINE_REFINE_PHASE_MAX);

    // Dominance du pic choisi (global ou best selon la logique ci-dessus).
    metric_t dom_chosen = (chosen_peak.power == global_peak.power)
                              ? dominance_global
                              : dominance_best;
    bool dom_ok = (dom_chosen >= FS_LOCAL_DOMINANCE_MIN);

    // Aligned with thesis: acquisition = variance fine > threshold (Eq.2 / Listing 2 line 39)
    // Critère de détection finale : uniquement la variance fine dépasse le seuil haut.
    // La dominance et la cohérence sont des indicateurs complémentaires (log uniquement).
    bool fine_refine_ok = has_fine_candidate && var_pass_high;

    // Variables de tracage (inchangees en nom pour garder les prints existants).
    bool acquired_hybrid = fine_refine_ok; // pour les logs uniquement

#ifndef __SYNTHESIS__
    std::cout << "[DBG FINE THESIS] per-variant min/max" << std::endl;
    for (int p = 0; p < 3; p++) {
        if (!has_peak[p]) {
            std::cout << "  var=" << p << " count=0" << std::endl;
        } else {
            metric_t ratio = (highest_peak[p].power > (peak_power_t)0)
                                 ? (metric_t)(((power_t)lowest_peak[p].power) / (power_t)highest_peak[p].power)
                                 : (metric_t)0;
            metric_t var_p = nvar_from_p1_pk((power_t)highest_peak[p].power, (power_t)lowest_peak[p].power);
            std::cout << "  var=" << p
                      << " count=" << num_peaks_fs[p]
                      << " high=" << highest_peak[p].power
                      << " low=" << lowest_peak[p].power
                      << " low/high=" << ratio
                      << " nvar=" << var_p
                      << std::endl;
        }
    }

    if (best_var_variant >= 0) {
        dbg_fs_top_spaced_count = num_peaks_fs[best_var_variant];
        dbg_fs_p1 = (power_t)highest_peak[best_var_variant].power;
        dbg_fs_p4 = (power_t)lowest_peak[best_var_variant].power;
        std::cout << "[DBG FINE THESIS] selected_variant=" << best_var_variant
                  << " selected_doppler=" << (int)best_peak.doppler
                  << " selected_phase=" << best_peak.phase
                  << " selected_peak=" << best_peak.power
                  << std::endl;
    } else {
        dbg_fs_top_spaced_count = 0;
        dbg_fs_p1 = 0;
        dbg_fs_p4 = 0;
    }
    std::cout << "[DBG FINE HYBRID] global_variant=" << global_peak_variant
              << " global_peak=" << global_peak.power
              << " global_doppler=" << (int)global_peak.doppler
              << " global_phase=" << global_peak.phase
              << " var_high=" << (var_pass_high ? 1 : 0)
              << " var_low=" << (var_pass_low ? 1 : 0)
              << " dom_best=" << dominance_best
              << " dom_global=" << dominance_global
              << " global_guard=" << (global_guard ? 1 : 0)
              << " chosen_doppler=" << (int)chosen_peak.doppler
              << " chosen_phase=" << chosen_peak.phase
              << " chosen_peak=" << chosen_peak.power
              << " acquired=" << (acquired_hybrid ? 1 : 0)
              << std::endl;

    std::cout << std::flush;
#endif

    // Mise à jour des sorties : si la fine valide la détection, on retourne
    // les coordonnées du pic choisi ; sinon on restaure les valeurs coarse
    // et on signale une non-détection (detected_out=false).
    if (fine_refine_ok) {
        doppler_out   = (int)chosen_peak.doppler;
        phase_out     = chosen_peak.phase;
        peak_out      = (power_t)chosen_peak.power;
        peak_fine_out = (power_t)chosen_peak.power;
        detected_out  = true;
    } else {
        doppler_out   = coarse_doppler_saved;
        phase_out     = coarse_phase_saved;
        peak_out      = coarse_peak_saved;
        peak_fine_out = coarse_peak_saved;
        detected_out  = false;
    }
    fine_var_out = higher_var;
#ifndef __SYNTHESIS__
    std::cout << "[TRACE FINE] exit detected=" << (detected_out ? 1 : 0)
              << " refine_ok=" << (fine_refine_ok ? 1 : 0)
              << " doppler=" << doppler_out
              << " phase=" << phase_out
              << " peak=" << peak_out
              << std::endl;
#endif
}

// ========== TOP FUNCTION ==========
// Orchestrateur complet de la chaine STSA.
// Séquence : capture AXIS → PRN variants → coarse search → (optionnel) fine search.
// La fine search est sautée si la coarse n'est pas détectée ET non borderline,
// économisant la latence du DATAFLOW fine pour les signaux clairement absents.
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
) {
#pragma HLS INLINE off
    // ChaÃ®ne complÃ¨te d'acquisition (coarse puis fine).

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] enter" << std::endl;
#endif

    // Buffers RAM locaux pour le signal reçu et le PRN brut.
    // Déclarés en variables locales (pas static) : alloués en BRAM dédiée,
    // réinitialisés implicitement à chaque appel de la fonction.
    sample_t signal[N];
    sample_t prn[N];

#ifndef __SYNTHESIS__
    dbg_cs_p1 = 0;
    dbg_cs_p5 = 0;
    dbg_cs_top_count = 0;
    dbg_fs_p1 = 0;
    dbg_fs_p4 = 0;
    dbg_fs_top_spaced_count = 0;
#endif

    // Application des seuils par défaut si les valeurs fournies sont nulles ou négatives.
    // Permet à acquisition_stsa_top d'appeler cette fonction avec des seuils fixes
    // sans exposer les constantes SEUIL_* à l'interface AXI-Lite.
    if(threshold_coarse <= (metric_t)0) threshold_coarse = (metric_t)SEUIL_VARIANCE_COARSE;
    if(threshold_fine <= (metric_t)0) threshold_fine = (metric_t)SEUIL_VARIANCE_FINE;

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] thresholds coarse=" << threshold_coarse
              << " fine=" << threshold_fine << std::endl;
    std::cout << "[TRACE ACQ] before capture_inputs_to_mem_stsa" << std::endl;
#endif

    // Transfert des N échantillons depuis les flux AXIS vers les tableaux RAM locaux.
    capture_inputs_to_mem_stsa(rx_stream, prn_stream, signal, prn);

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] after capture_inputs_to_mem_stsa" << std::endl;
#endif

    // Déclarés static : persistent entre les appels, mappés sur BRAM dédiée par HLS.
    // Évite la réallocation et le rechargement des tables à chaque invocation,
    // mais impose que l'IP ne soit pas réentrante (usage unique dans le design).
    static prn_sign_t prn_var[PRN_VARIANTS][2 * N];
    static prn_sign_t prn_banks[PRN_VARIANTS][FINE_TAU_TILE][2 * N];
// Partition sur dim=2 : FINE_TAU_TILE bancs indépendants pour les accès parallèles.
#pragma HLS ARRAY_PARTITION variable=prn_banks complete dim=2
// ram_2p bram latency=2 : deux ports de lecture permettent les accès simultanés
// depuis les boucles SAMPLE_LOOP_ALL (fine) et SAMPLE_LOOP_COARSE (coarse).
#pragma HLS BIND_STORAGE variable=prn_banks type=ram_2p impl=bram latency=2
    static int tau_start_tbl[NB_PHASES];
// lutram : table de NB_PHASES entiers (≈ 1024 × 32 bits = 4 KB),
// plus efficace en LUT distribuée qu'en BRAM pour une table de petite taille à lecture fréquente.
#pragma HLS BIND_STORAGE variable=tau_start_tbl type=ram_1p impl=lutram

    // Préparation des tables PRN et de la LUT de départ tau :
    // ces trois fonctions sont séquentielles (chacune dépend de la précédente).
    build_prn_variants(prn, prn_var);
    build_tau_start_table(tau_start_tbl);
    build_prn_banks(prn_var, prn_banks);

    // Étape 1 : recherche coarse sur la grille complète Doppler × phase.
    // Remplit doppler_out / phase_out / peak_out avec la meilleure estimation
    // et metric_coarse_var_out avec la variance inter-passes (critère de détection).
    coarse_search(
        signal,
        prn_banks,
        tau_start_tbl,
        doppler_out,
        phase_out,
        peak_out,
        detected_out,
        threshold_coarse,
        compare_mode,
        metric_coarse_var_out
    );

    peak_coarse_out = peak_out;

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] after coarse_search detected=" << (detected_out ? 1 : 0)
              << " doppler=" << doppler_out
              << " phase=" << phase_out
              << " peak=" << peak_out
              << std::endl;
#endif

#ifndef __SYNTHESIS__
    std::cout << "[DBG SELECT] COARSE top_count=" << dbg_cs_top_count
              << " p1=" << dbg_cs_p1
              << " p5=" << dbg_cs_p5 << std::endl;
#endif

    // Fallback borderline: if coarse is just below threshold, still run fine search.
    // This recovers off-grid low-SNR cases without changing thesis thresholds.
    // coarse_promising = true si la variance coarse atteint 85% du seuil :
    // on tente quand même la fine pour récupérer les cas à faible SNR hors-grille.
    metric_t coarse_fallback_thresh = threshold_coarse * (metric_t)0.85f;
    bool coarse_promising = (metric_coarse_var_out >= coarse_fallback_thresh);

    // Court-circuit : si la coarse n'a pas détecté ET n'est pas borderline,
    // on skip la fine pour économiser ~N × fine_doppler_count × fine_tau_count cycles.
    if (!detected_out && !coarse_promising) {
        metric_fine_var_out = 0;
        peak_fine_out = 0;
        max_power_out = 0;
        mean_power_out = 0;
#ifndef __SYNTHESIS__
        std::cout << "[DBG SELECT] FINE   skipped (coarse non detectee et non borderline)"
                  << " list_count=" << dbg_fs_top_spaced_count
                  << " high=" << dbg_fs_p1
                  << " low=" << dbg_fs_p4 << std::endl;
#endif
        return;
    }

    power_t max_power = 0;
    power_t sum_power = 0;
    // Initialisés à leur valeur maximale pour que fine_search puisse les réduire
    // selon la confiance coarse (politique adaptative T3).
    int fine_doppler_count = NB_DOPPLER_FINE;
    int fine_tau_count = NB_PHASES;

    // Étape 2 : recherche fine autour de l'estimation coarse.
    // doppler_out / phase_out / peak_out sont mis à jour en place si la fine détecte.
    // fine_doppler_count et fine_tau_count sont retournés pour calculer mean_power.
    fine_search(
        signal,
        prn_banks,
        tau_start_tbl,
        (doppler_t)doppler_out,
        doppler_out,
        phase_out,
        peak_out,
        detected_out,
        threshold_fine,
        metric_coarse_var_out,
        threshold_coarse,
        metric_fine_var_out,
        peak_fine_out,
        fine_doppler_count,
        fine_tau_count,
        corr_out,
        max_power,
        sum_power
    );

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] after fine_search detected=" << (detected_out ? 1 : 0)
              << " doppler=" << doppler_out
              << " phase=" << phase_out
              << " peak=" << peak_out
              << " max_power=" << max_power
              << " sum_power=" << sum_power
              << std::endl;
#endif

    // Calcul de la puissance moyenne sur l'ensemble de la surface de corrélation fine.
    // mean_power = sum_power / (doppler_count × tau_count × 3 variantes).
    // Utile pour estimer le bruit de fond et calculer un SNR approximatif côté PS.
    int total_corr = fine_doppler_count * fine_tau_count * PRN_VARIANTS;
    power_t mean_power = (power_t)0;
    if (total_corr > 0) {
        mean_power = sum_power / (power_t)total_corr;
    }
    // Cast vers int : les ports AXI-Lite sont des entiers 32 bits ;
    // la précision fractionnaire de power_t est perdue mais suffisante pour le PS.
    max_power_out = (int)max_power;
    mean_power_out = (int)mean_power;

#ifndef __SYNTHESIS__
    std::cout << "[TRACE ACQ] outputs max_power_out=" << max_power_out
              << " mean_power_out=" << mean_power_out
              << std::endl;
#endif

#ifndef __SYNTHESIS__
    std::cout << "[DBG SELECT] FINE   list_count=" << dbg_fs_top_spaced_count
              << " high=" << dbg_fs_p1
              << " low=" << dbg_fs_p4 << std::endl;
    std::cout << "[TRACE ACQ] exit" << std::endl;
#endif
}

// Point d'entree HLS expose cote IP (AXIS + AXI-Lite).
// Cette fonction est le top-level synthétisé : elle définit l'interface RTL de l'IP.
// Les ports AXIS (rx_stream, prn_stream, corr_out) sont connectés au DMA PL-PS.
// Les ports s_axilite (scalaires) sont accessibles depuis le PS via le bus AXI4-Lite.
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
#ifndef __SYNTHESIS__
    std::cout << "[TRACE TOP] enter" << std::endl;
#endif
// Interfaces AXI-Stream pour les flux d'entrée (signal reçu et PRN) et de sortie (corrélation).
#pragma HLS INTERFACE axis port=rx_stream
#pragma HLS INTERFACE axis port=prn_stream
// corr_out : AXIS de sortie avec buffer de profondeur AXIS_OUT_DEPTH pour absorber
// les latences du DMA PS avant que la FIFO interne ne se remplisse.
#pragma HLS INTERFACE axis port=corr_out   depth=AXIS_OUT_DEPTH
// Registres AXI-Lite en lecture depuis le PS après la fin du traitement (return = ap_done).
// Tous groupés dans le même bundle CTRL pour minimiser les ressources d'interface.
#pragma HLS INTERFACE s_axilite port=doppler_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=phase_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=peak_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=detected_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=max_power_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=mean_power_out bundle=CTRL
// Le port return génère le signal ap_done/ap_idle/ap_ready du protocole HLS IP.
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Seuils par défaut et métriques internes non exposées.
    // Ces métriques sont calculées en interne mais non exposées sur l'interface AXI-Lite
    // pour simplifier le driver PS (réduction du nombre de registres à lire).
    metric_t metric_coarse_var_out = 0;
    metric_t metric_fine_var_out = 0;
    power_t peak_coarse_out = 0;
    power_t peak_fine_out = 0;

    // Appel avec les seuils constants (non configurables via AXI-Lite dans cette version).
    // compare_mode=false : sortie anticipée dès détection coarse, mode production normal.
    acquisition_stsa(
        rx_stream,
        prn_stream,
        corr_out,
        doppler_out,
        phase_out,
        peak_out,
        detected_out,
        (metric_t)SEUIL_VARIANCE_COARSE,
        (metric_t)SEUIL_VARIANCE_FINE,
        false,
        metric_coarse_var_out,
        metric_fine_var_out,
        peak_coarse_out,
        peak_fine_out,
        max_power_out,
        mean_power_out
    );

#ifndef __SYNTHESIS__
    std::cout << "[TRACE TOP] exit detected=" << (detected_out ? 1 : 0)
              << " doppler=" << doppler_out
              << " phase=" << phase_out
              << " peak=" << peak_out
              << std::endl;
#endif
}