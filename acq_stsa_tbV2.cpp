//==============================================================================
// acq_stsa_tbV2.cpp
// Testbench C/COSIM du top-level `acquisition_stsa_top`.
// Objectif: charger des vecteurs CSV, executer le top, puis verifier que
// le flux de correlation est bien integralement emis (jusqu'au flag `last`).
//==============================================================================
#include "acq_stsaV2.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

using namespace std;

// Utilitaires de chemins pour rendre l'execution robuste (IDE/CLI/Vitis).
static string directory_of_file(const string &file_path) {
    // Récupère le dossier parent d'un chemin fichier.
    size_t pos = file_path.find_last_of("\\/");
    if (pos == string::npos) {
        return ".";
    }
    return file_path.substr(0, pos);
}

static string join_path(const string &base, const string &name) {
    // Concatène proprement "dossier + nom de fichier".
    if (base.empty() || base == ".") {
        return name;
    }
    char last = base[base.size() - 1];
    if (last == '/' || last == '\\') {
        return base + name;
    }
    return base + "/" + name;
}

// Lecture CSV permissive: ignore les colonnes non numeriques (ex. en-tete).
static vector<int> read_csv(const string &file) {
    vector<int> data;
    // Lit une suite d'entiers (CSV simple).
    ifstream f(file.c_str());
    if (!f.is_open()) {
        cerr << "Erreur: Impossible d'ouvrir " << file << endl;
        return data;
    }

    string line;
    while (getline(f, line)) {
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            if (!val.empty()) {
                try {
                    data.push_back(stoi(val));
                } catch (...) {
                    // Ignore entetes/valeurs non numeriques
                }
            }
        }
    }
    return data;
}

static vector<int> read_csv_with_fallback(const vector<string> &paths) {
    // Essaye plusieurs chemins pour simplifier l'exécution locale.
    for (size_t i = 0; i < paths.size(); i++) {
        ifstream test(paths[i].c_str());
        if (test.is_open()) {
            test.close();
            cout << "[INFO] CSV trouve : " << paths[i] << endl;
            return read_csv(paths[i]);
        }
    }

    cerr << "[ERREUR] Aucun chemin CSV valide trouve." << endl;
    for (size_t i = 0; i < paths.size(); i++) {
        cerr << "         - " << paths[i] << endl;
    }
    return vector<int>();
}

// Convertit les tableaux d'entree en trames AXIS (TLAST sur le dernier echantillon).
// [BRANCHEMENT TEST CARLOS/TAZI] En mode STSA_USE_FINE_STREAMING :
// - Le signal est injecte 1 fois pour le coarse search (capture_inputs_to_mem_stsa)
// - Puis NB_DOPPLER_FINE fois supplementaires pour la fine search (fine_mix_streaming
//   appelle rx_stream.read() pour chaque bin Doppler, soit jusqu'a 21 fois).
// Le PRN est injecte 1 seule fois (non relu par la fine search en version streaming).
static void fill_input_streams(
    const vector<int> &signal_vec,
    const vector<int> &prn_vec,
    hls::stream<axis_t> &rx_stream,
    hls::stream<axis_t> &prn_stream
) {
#ifdef STSA_USE_FINE_STREAMING
    // [STREAMING] Injection du signal 1 + NB_DOPPLER_FINE fois :
    //   - Pass 0 : consomme par capture_inputs_to_mem_stsa (coarse search)
    //   - Pass 1..NB_DOPPLER_FINE : consomme par fine_mix_streaming (fine search,
    //     un passage par bin Doppler)
    // NB : en pratique fine_doppler_count <= NB_DOPPLER_FINE (7 a 21 selon confiance
    // coarse). On injecte NB_DOPPLER_FINE passages = pire cas pour couvrir tous les cas.
    const int n_rx_passes = 1 + NB_DOPPLER_FINE;
    cout << "[STREAMING TB] Injection rx_stream : " << n_rx_passes
         << " passes de " << N << " echantillons = "
         << (n_rx_passes * N) << " elements total" << endl;

    for (int pass = 0; pass < n_rx_passes; pass++) {
        for (int i = 0; i < N; i++) {
            axis_t rx_val;
            rx_val.data = static_cast<int32_t>(signal_vec[i]);
            rx_val.keep = -1;
            rx_val.strb = -1;
            // TLAST seulement sur le dernier echantillon du DERNIER pass
            rx_val.last = (pass == n_rx_passes - 1) && (i == N - 1);
            rx_stream.write(rx_val);
        }
    }

    // PRN : injecte une seule fois (identique a la version buffer)
    for (int i = 0; i < N; i++) {
        axis_t prn_val;
        prn_val.data = static_cast<int32_t>(prn_vec[i]);
        prn_val.keep = -1;
        prn_val.strb = -1;
        prn_val.last = (i == N - 1);
        prn_stream.write(prn_val);
    }
#else
    // [BUFFER - defaut] Injection standard : 1 seul passage signal + PRN.
    for (int i = 0; i < N; i++) {
        axis_t rx_val;
        axis_t prn_val;

        rx_val.data = static_cast<int32_t>(signal_vec[i]);
        rx_val.keep = -1;
        rx_val.strb = -1;
        rx_val.last = (i == N - 1);
        rx_stream.write(rx_val);

        prn_val.data = static_cast<int32_t>(prn_vec[i]);
        prn_val.keep = -1;
        prn_val.strb = -1;
        prn_val.last = (i == N - 1);
        prn_stream.write(prn_val);
    }
#endif
}

// Consomme corr_out jusqu'a TLAST et retourne le nombre d'echantillons lus.
static int drain_corr_stream(hls::stream<axis_t> &corr_out) {
    int corr_count = 0;
    // Vide le flux de corrélation jusqu'au paquet last.
    while (!corr_out.empty()) {
        axis_t pkt = corr_out.read();
        corr_count++;
        if (pkt.last) {
            break;
        }
    }
    return corr_count;
}

// power_t = ap_ufixed<48, 32> -> Q32.16. 48 bits tiennent dans un u64.
static unsigned long long peak_to_raw(const power_t &peak_val) {
#if defined(STSA_CSIM_FIXED)
    ap_uint<48> raw = peak_val.range(47, 0);
    return (unsigned long long)raw;
#else
    (void)peak_val;
    return 0ULL;
#endif
}

int main() {
    const string source_dir = directory_of_file(__FILE__);

    cout << "==================================================" << endl;
    cout << "   TESTBENCH STSA CORRECTED (DOUBLE APPEL)" << endl;
    cout << "==================================================" << endl;
    cout << "N         : " << N << endl;
    cout << "NB_PHASES : " << NB_PHASES << endl;
    cout << "FS        : " << FS << " Hz" << endl;
    cout << "==================================================" << endl << endl;

    // Chargement des 2 entrées depuis CSV (avec chemins de secours).
    vector<int> signal_vec = read_csv_with_fallback({
        "signal.csv",
        "../signal.csv",
        "../../STSA/signal.csv",
        join_path(source_dir, "signal.csv")
    });
    vector<int> prn_vec = read_csv_with_fallback({
        "PRN-25.csv",
        "../PRN-25.csv",
        "../../STSA/PRN-25.csv",
        join_path(source_dir, "PRN-25.csv")
    });

    // Validation minimale: il faut au moins N échantillons.
    if (signal_vec.size() < N || prn_vec.size() < N) {
        cerr << "[ERREUR] Taille CSV insuffisante !" << endl;
        cerr << "         Signal: " << signal_vec.size() << " < N=" << N << endl;
        cerr << "         PRN:    " << prn_vec.size() << " < N=" << N << endl;
        return -1;
    }

    // Appel unique du wrapper top pour valider la chaine complete.
    hls::stream<axis_t> rx_stream_top;
    hls::stream<axis_t> prn_stream_top;
    hls::stream<axis_t> corr_out_top;

    fill_input_streams(signal_vec, prn_vec, rx_stream_top, prn_stream_top);

    int doppler_top = 0;
    int phase_top = 0;
    power_t peak_top = 0;
    bool detected_top = false;
    int max_power_top = 0;
    int mean_power_top = 0;

    // Mesure simple du temps d'exécution du top.
    auto t3 = chrono::high_resolution_clock::now();

    cout << "[TRACE TB] avant appel acquisition_stsa_top" << endl;

    acquisition_stsa_top(
        rx_stream_top,
        prn_stream_top,
        corr_out_top,
        doppler_top,
        phase_top,
        peak_top,
        detected_top,
        max_power_top,
        mean_power_top
    );

    cout << "[TRACE TB] retour acquisition_stsa_top" << endl;

    // Lecture complete du flux de correlation produit.
    int corr_count_top = drain_corr_stream(corr_out_top);

    cout << "[TRACE TB] corr_out_top drain count=" << corr_count_top << endl;

    auto t4 = chrono::high_resolution_clock::now();
    double time_top_ms = chrono::duration<double, milli>(t4 - t3).count();

    cout << endl;
    cout << "================ RESULTATS acquisition_stsa_top ================" << endl;
    cout << "  Doppler        : " << doppler_top << endl;
    cout << "  Phase          : " << phase_top << endl;
    cout << "  Peak final     : " << peak_top << endl;
#if defined(STSA_CSIM_FIXED)
    unsigned long long peak_raw = peak_to_raw(peak_top);
    double peak_q32_16 = (double)peak_raw / (double)(1ULL << 16);
    cout << "  Peak raw (u48) : " << peak_raw << endl;
    cout << "  Peak q32.16    : " << fixed << setprecision(6) << peak_q32_16 << endl;
#endif
    cout << "  Detected       : " << (detected_top ? "OUI" : "NON") << endl;
    cout << "  max_power_out  : " << max_power_top << endl;
    cout << "  mean_power_out : " << mean_power_top << endl;
    cout << "  corr_out count : " << corr_count_top << endl;
    cout << "  Temps          : " << fixed << setprecision(2) << time_top_ms << " ms" << endl;

    return 0;
}

#else
#include <iostream>
using namespace std;

int main() {
    cout << "Ce testbench n'est pas destine a la synthese" << endl;
    return 0;
}
#endif
