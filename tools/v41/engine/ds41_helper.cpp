// tools/v41/engine/ds41_helper.cpp - the second GPU's process for STRATA_DS41_REMOTE (see ds41_remote.hpp).
//
//   ds41_helper -m <gguf> --rank <rank.bin> [--device 1] [--skip-gib 22] [--gib 13.5] [--name ds41_remote]
//               [--chunk-tok 12288]
//
// Holds the experts of the routing profile that come after the first --skip-gib (about what the engine's card keeps
// in its VRAM cache) until --gib of this card's VRAM is used, then serves the engine's requests until it says stop.
// Each request is computed with the engine's own kernels (native_quantize_q8_1, native_expert_grouped per format
// class), so a row is the one the engine's card would have produced from the same blob.
// --chunk-tok (0 = off): the widest prompt chunk it takes through the `<name>_chunk` segment (ds41_remote.hpp).
#include "ds41_moe.hpp"
#include "ds41_remote.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#if defined(DS41_HELPER_MMQ)
#include "ds41_mmq_plan.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <future>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::v41;
namespace sk = strata::kernels;

namespace {

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "ds41_helper: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Args {
    std::string model, rank, name = "ds41_remote";
    int device = 1;
    double skip_gib = 22.0, gib = 13.5;
    int64_t chunk_tok = 12288;
    double ram_gib = 0.0;   // --ram-gib G: a RAM tier after the VRAM set (pinned host memory, staged into the card per request)
    double ram_skip_gib = 0.0;   // --ram-skip-gib S: leave the next S GiB of the profile to the engine's arena; the RAM tier starts after
    int cpu_threads = 0;         // --cpu-threads N: compute part of the RAM tier's blobs on the CPU (the engine's own CPU expert kernels)
    double cpu_share = 0.45;     // --cpu-share F: that part, as a fraction of a request's distinct RAM-tier experts (rounded)
    int vram_cache = 0;          // --vram-cache 1: the card's set is an LRU cache over the RAM tier (which then also holds the set)
    int vram_cache_min_age = 8;  // --vram-cache-min-age N: a slot is a victim only if unused for N requests
    double chunk_stage_gib = 3.0;   // --chunk-stage-gib G: VRAM for the RAM tier's blobs during a prompt chunk (staged in waves)
    int ram_pin = 1;        // --ram-pin 0: plain (pageable) host memory for it
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--rank") a.rank = next();
        else if (k == "--device") a.device = std::atoi(next().c_str());
        else if (k == "--skip-gib") a.skip_gib = std::atof(next().c_str());
        else if (k == "--gib") a.gib = std::atof(next().c_str());
        else if (k == "--name") a.name = next();
        else if (k == "--chunk-tok") a.chunk_tok = std::atoll(next().c_str());
        else if (k == "--ram-gib") a.ram_gib = std::atof(next().c_str());
        else if (k == "--ram-pin") a.ram_pin = std::atoi(next().c_str());
        else if (k == "--ram-skip-gib") a.ram_skip_gib = std::atof(next().c_str());
        else if (k == "--cpu-threads") a.cpu_threads = std::atoi(next().c_str());
        else if (k == "--cpu-share") a.cpu_share = std::atof(next().c_str());
        else if (k == "--vram-cache") a.vram_cache = std::atoi(next().c_str());
        else if (k == "--vram-cache-min-age") a.vram_cache_min_age = std::atoi(next().c_str());
        else if (k == "--chunk-stage-gib") a.chunk_stage_gib = std::atof(next().c_str());
        else { std::fprintf(stderr, "ds41_helper: unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.rank.empty();
}

}  // namespace

int main(int argc, char** argv) {
    static const bool dbg = std::getenv("DS41_HELPER_DEBUG") != nullptr;
#define DBG(msg) do { if (dbg) { std::fprintf(stderr, "ds41_helper: [%s]\n", msg); std::fflush(stderr); } } while (0)
    DBG("start");
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: ds41_helper -m <gguf> --rank <rank.bin> [--device 1] [--skip-gib 22] [--gib 13.5] [--name ds41_remote] [--chunk-tok 12288] [--ram-gib 0 [--ram-pin 1] [--ram-skip-gib 0] [--cpu-threads 0 --cpu-share 0.45] [--vram-cache 1]]\n");
        return 2;
    }
    std::string err;
    Ds41RemoteMap map;
    if (!map.open(a.name, true, err)) { std::fprintf(stderr, "ds41_helper: %s\n", err.c_str()); return 1; }
    Ds41RemoteShm& S = *map.shm();
    S.ready.store(0);
    S.stop.store(0);
    S.req.store(0);
    S.done.store(0);
    auto fail = [&](const std::string& why) {
        std::fprintf(stderr, "ds41_helper: %s\n", why.c_str());
        S.ready.store(-1, std::memory_order_release);
        return 1;
    };

    DBG("map");
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) return fail("model: " + err);
    const Ds41MoeGeom& G = src.geom();
    if (G.n_embd > Ds41RemoteShm::kMaxEmbd || G.n_layers * G.n_experts > Ds41RemoteShm::kMaxPairs)
        return fail("the model is larger than the shared segment's limits");
    S.magic = Ds41RemoteShm::kMagic;
    S.version = Ds41RemoteShm::kVersion;
    S.n_layers = G.n_layers;
    S.n_experts = G.n_experts;
    S.n_embd = G.n_embd;
    std::memset(S.holds, 0, (size_t) (G.n_layers * G.n_experts));

    DBG("model");
    if (!ck(cudaSetDevice(a.device), "cudaSetDevice")) return fail("no CUDA device " + std::to_string(a.device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, a.device);

    DBG("device");
    // the engine's layouts, one per format class (Ds41MoeTier's init does the same)
    const int64_t NCL = G.n_classes();
    std::vector<sk::NativeExpertLayout> gll((size_t) NCL);
    for (int64_t c = 0; c < NCL; ++c) {
        if (!G.per_expert() && !G.routed(c)) continue;
        const int gt = G.per_expert() ? G.cls_gu[(size_t) c] : G.gate_type(c);
        const int ut = G.per_expert() ? gt : G.up_type(c);
        const int dt = G.per_expert() ? G.cls_d[(size_t) c] : G.down_type(c);
        if (!sk::native_expert_supported3(gt, ut, dt, G.n_embd, G.n_ff)) return fail("no kernel for class " + std::to_string(c));
        gll[(size_t) c] = sk::native_expert_layout3(gt, ut, dt, G.n_embd, G.n_ff);
        gll[(size_t) c].swiglu_limit = G.swiglu_limit;
    }

    DBG("layouts");
    // each class's slot stride in the slab: its blob, or with MMQ the blob rounded to the stride run_chunk's MMQ reads
    // experts at (ds41_mmq_plan.hpp), so a prompt chunk multiplies a class's experts of a layer in place
    std::vector<int64_t> cbl((size_t) NCL, 0), cstride((size_t) NCL, 0);
    for (int64_t l = 0; l < G.n_layers; ++l)
        for (int64_t e = 0; e < G.n_experts; ++e)
            if (G.routed(l) && !cbl[(size_t) G.cls(l, e)]) cbl[(size_t) G.cls(l, e)] = (int64_t) src.expert_bytes(l, e);
#if defined(DS41_HELPER_MMQ)
    std::vector<MmqPlan> mplan((size_t) NCL);
    const bool use_mmq = strata::prefill::mmq::built() && !(std::getenv("DS41_HELPER_MMQ") && std::getenv("DS41_HELPER_MMQ")[0] == '0');
#endif
    for (int64_t c = 0; c < NCL; ++c) {
        cstride[(size_t) c] = cbl[(size_t) c];
#if defined(DS41_HELPER_MMQ)
        if (use_mmq && cbl[(size_t) c] > 0) {
            mplan[(size_t) c] = mmq_plan(gll[(size_t) c], cbl[(size_t) c]);
            if (mplan[(size_t) c].ok) cstride[(size_t) c] = mplan[(size_t) c].SL;
        }
#endif
    }

    // the set: the profile after the first skip_gib, until gib
    std::vector<std::pair<int32_t, int32_t>> ranked = ds41_read_ranked(a.rank, err);
    if (ranked.empty()) return fail("rank: " + err);
    const double skip_b = a.skip_gib * 1073741824.0, take_b = a.gib * 1073741824.0;
    double seen = 0, held_b = 0, ram_b = 0, ram_skipped = 0;
    const double ram_cap = a.ram_gib * 1073741824.0, ram_skip = a.ram_skip_gib * 1073741824.0;
    std::vector<std::pair<int32_t, int32_t>> set, rset;   // the card's set, then the RAM tier's (the next of the profile)
    std::vector<char> picked((size_t) (G.n_layers * G.n_experts), 0);
    bool vram_full = false;
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= G.n_layers || le.second < 0 || le.second >= G.n_experts) continue;
        char& p = picked[(size_t) (le.first * G.n_experts + le.second)];
        if (p) continue;
        p = 1;
        const double b = (double) src.expert_bytes(le.first, le.second);
        seen += b;
        if (seen <= skip_b) continue;
        const double sb = (double) cstride[(size_t) G.cls(le.first, le.second)];   // its slot in the slab
        if (!vram_full && held_b + sb <= take_b) { held_b += sb; set.push_back(le); continue; }
        vram_full = true;
        if (ram_cap <= 0) break;
        if (ram_skipped < ram_skip) { ram_skipped += b; continue; }   // the engine's arena takes these (hotter) ranks
        const double rb = (double) ((src.expert_bytes(le.first, le.second) + 255) & ~(size_t) 255);
        if (ram_b + rb > ram_cap) break;
        ram_b += rb;
        rset.push_back(le);
    }

    if (std::getenv("DS41_HELPER_CLASSES")) {   // diagnostic: the set's experts per format class
        std::vector<int> nc((size_t) G.n_classes(), 0);
        for (const auto& le : set) ++nc[(size_t) G.cls(le.first, le.second)];
        for (int64_t c = 0; c < G.n_classes(); ++c)
            if (nc[(size_t) c]) std::fprintf(stderr, "ds41_helper: class %lld (gate/up %d, down %d): %d experts\n", (long long) c,
                                             G.per_expert() ? G.cls_gu[(size_t) c] : -1, G.per_expert() ? G.cls_d[(size_t) c] : -1, nc[(size_t) c]);
    }
    DBG("set");
    // one device slab for the set: a group per (layer, class) with its experts in ascending id order `cstride` apart
    // (the order the engine sorts a chunk's entries in), each group 4 KiB aligned, zeroed (MMQ reads a little past a
    // blob, and llama.cpp zero-pads too), plus 1 MiB of tail
    struct Grp { int32_t layer, c; size_t off; std::vector<int32_t> es; };
    std::vector<Grp> grps;
    std::vector<int32_t> grp_of((size_t) (G.n_layers * NCL), -1), slot_of((size_t) (G.n_layers * G.n_experts), -1);
    {
        std::vector<std::pair<int32_t, int32_t>> by = set;
        std::sort(by.begin(), by.end(), [&](const std::pair<int32_t, int32_t>& x, const std::pair<int32_t, int32_t>& y) {
            const int cx = G.cls(x.first, x.second), cy = G.cls(y.first, y.second);
            return x.first != y.first ? x.first < y.first : cx != cy ? cx < cy : x.second < y.second;
        });
        size_t off = 0;
        for (const auto& le : by) {
            const int c = G.cls(le.first, le.second);
            int32_t& g = grp_of[(size_t) (le.first * NCL + c)];
            if (g < 0) {
                off = (off + 4095) & ~(size_t) 4095;
                g = (int32_t) grps.size();
                grps.push_back({le.first, c, off, {}});
            }
            slot_of[(size_t) (le.first * G.n_experts + le.second)] = (int32_t) grps[(size_t) g].es.size();
            grps[(size_t) g].es.push_back(le.second);
            off += (size_t) cstride[(size_t) c];
        }
        held_b = (double) off;
    }
    const size_t slab_bytes = (size_t) held_b + ((size_t) 1 << 20);
    uint8_t* slab = nullptr;
    if (!ck(cudaMalloc(&slab, slab_bytes), "slab")) return fail("cannot allocate " + std::to_string(held_b / 1073741824.0) + " GiB");
    if (!ck(cudaMemset(slab, 0, slab_bytes), "slab zero")) return fail("slab zero");
    std::vector<unsigned long long> dptr((size_t) (G.n_layers * G.n_experts), 0);
    std::vector<uint8_t> host(src.blob_bytes());
    const double t0 = now_ms();
    for (const Grp& g : grps)
        for (size_t q = 0; q < g.es.size(); ++q) {
            uint8_t* d = slab + g.off + q * (size_t) cstride[(size_t) g.c];
            src.read(g.layer, g.es[q], host.data());
            if (!ck(cudaMemcpy(d, host.data(), src.expert_bytes(g.layer, g.es[q]), cudaMemcpyHostToDevice), "blob upload")) return fail("upload failed");
            dptr[(size_t) (g.layer * G.n_experts + g.es[q])] = (unsigned long long) d;
        }
    for (const auto& le : set) S.holds[(size_t) (le.first * G.n_experts + le.second)] = 1;
    S.held = (int64_t) set.size();
    S.held_bytes = (int64_t) held_b;

    // ---- the RAM tier (--ram-gib): one host arena, a slot per expert (256-aligned), read straight from the file ----
    std::vector<uint8_t*> hptr((size_t) (G.n_layers * G.n_experts), nullptr);
    uint8_t* harena = nullptr;
    bool ram_pinned = false;
    std::vector<std::pair<int32_t, int32_t>> rall = rset;   // what the RAM tier holds
    if (a.vram_cache && !rset.empty()) {
        for (const auto& le : set) { rall.push_back(le); ram_b += (double) ((src.expert_bytes(le.first, le.second) + 255) & ~(size_t) 255); }
    }
    if (!rall.empty()) {
        const size_t rbytes = (size_t) ram_b + ((size_t) 1 << 20);
        if (a.ram_pin && cudaHostAlloc((void**) &harena, rbytes, cudaHostAllocDefault) == cudaSuccess) ram_pinned = true;
        else {
            cudaGetLastError();
            if (a.ram_pin) std::fprintf(stderr, "ds41_helper: cannot pin %.1f GiB - the RAM tier is pageable (slower H2D)\n", ram_b / 1073741824.0);
            harena = (uint8_t*) std::malloc(rbytes);
            if (!harena) return fail("cannot allocate the RAM tier (" + std::to_string(ram_b / 1073741824.0) + " GiB)");
        }
        const double tr = now_ms();
        size_t off = 0;
        for (const auto& le : rall) {
            const size_t bytes = src.expert_bytes(le.first, le.second);
            src.read(le.first, le.second, harena + off);
            hptr[(size_t) (le.first * G.n_experts + le.second)] = harena + off;
            S.holds[(size_t) (le.first * G.n_experts + le.second)] = 1;
            off += (bytes + 255) & ~(size_t) 255;
        }
        S.held += (int64_t) rset.size();
        S.held_bytes += (int64_t) ram_b;
        std::fprintf(stderr, "ds41_helper: RAM tier: %zu experts, %.2f GiB (%s), profile from %.1f GiB, read in %.1f s%s\n", rall.size(),
                     ram_b / 1073741824.0, ram_pinned ? "pinned" : "pageable", (seen - ram_b - ram_skipped) / 1073741824.0 + a.ram_skip_gib,
                     (now_ms() - tr) / 1000.0, a.vram_cache ? " (holds the card's set too: VRAM is an LRU cache)" : "");
    }
    // --vram-cache: the card's slots by format class, with the expert each holds and when it was last used
    struct VSlot { unsigned long long d; int32_t at; int64_t used; };   // at = layer * n_experts + expert
    std::vector<std::vector<VSlot>> vslots((size_t) NCL);
    std::vector<int32_t> vslot_of((size_t) (G.n_layers * G.n_experts), -1);
    const bool lru = a.vram_cache && !rset.empty();
    if (lru)
        for (const auto& le : set) {
            const int c = G.cls(le.first, le.second);
            const int32_t at = (int32_t) (le.first * G.n_experts + le.second);
            vslot_of[(size_t) at] = (int32_t) vslots[(size_t) c].size();
            vslots[(size_t) c].push_back({dptr[(size_t) at], at, 0});
        }
    int64_t nlru = 0;
    // staging slots on the card for the RAM tier's blobs of one request
    size_t max_blob = 0;
    for (int64_t c = 0; c < NCL; ++c) max_blob = std::max(max_blob, (size_t) cbl[(size_t) c]);
    max_blob = (max_blob + 4095) & ~(size_t) 4095;
    const int kStage = rset.empty() ? 0 : 48;
    uint8_t* d_stage = nullptr;
    if (kStage > 0 && !ck(cudaMalloc(&d_stage, (size_t) kStage * max_blob), "staging")) return fail("cannot allocate the staging slots");
    std::vector<int32_t> stg_e;
    std::vector<unsigned long long> stg_p;
    // ---- the CPU share of the RAM tier (--cpu-threads): the engine's own CPU expert kernels, per format class ----
    std::vector<cpu::NativeFmt> fl((size_t) NCL);
    std::vector<char> fl_ok((size_t) NCL, 0);
    std::unique_ptr<cpu::ExpertPool> cpool;
    if (a.cpu_threads > 0 && !rset.empty()) {
        for (int64_t c = 0; c < NCL; ++c) {
            if (!G.per_expert() && !G.routed(c)) continue;
            std::string e3;
            fl_ok[(size_t) c] = ds41_moe_class_layout(G, c, fl[(size_t) c], e3) ? 1 : 0;
            if (!fl_ok[(size_t) c]) std::fprintf(stderr, "ds41_helper: class %lld has no CPU layout (%s) - its blobs stay on the card\n", (long long) c, e3.c_str());
        }
        cpool.reset(new cpu::ExpertPool(a.cpu_threads));
        std::fprintf(stderr, "ds41_helper: CPU share: %d threads, %.2f of a request's RAM-tier experts\n", a.cpu_threads, a.cpu_share);
    }
    std::vector<cpu::ExpertJobMulti> cjobs((size_t) Ds41RemoteShm::kMaxEnt);
    std::vector<uint8_t> cnact((size_t) Ds41RemoteShm::kMaxEnt * cpu::kNativeActBytes);
    std::vector<cpu::ActQ> cactq((size_t) Ds41RemoteShm::kMaxEnt);
    std::vector<float> cparts((size_t) Ds41RemoteShm::kMaxEnt * (size_t) G.n_embd);
    std::vector<int32_t> ram_e, cpu_rows;
    int64_t ncpu_blobs = 0;

    DBG("slab");
    // request buffers
    const int64_t H = G.n_embd;
    const int kMaxEnt = Ds41RemoteShm::kMaxEnt;
    float* d_x = nullptr; void* d_xq = nullptr; float* d_out = nullptr; void* d_scratch = nullptr; uint8_t* d_meta = nullptr;
    const size_t meta_bytes = 64 * 1024;
    bool ok = ck(cudaMalloc(&d_x, (size_t) Ds41RemoteShm::kMaxTok * H * 4), "d_x") &&
              // one column's q8_1 bytes x the tokens (native_q8_1_bytes takes at most 8 columns; the tier sizes it so)
              ck(cudaMalloc(&d_xq, (size_t) Ds41RemoteShm::kMaxTok * sk::native_q8_1_bytes((int) H, 1)), "d_xq") &&
              ck(cudaMalloc(&d_out, (size_t) kMaxEnt * H * 4), "d_out") &&
              ck(cudaMalloc(&d_scratch, sk::native_expert_scratch_bytes(kMaxEnt, G.n_ff)), "d_scratch") &&
              ck(cudaMalloc(&d_meta, meta_bytes), "d_meta");
    // pinned: the metadata copy is a plain DMA (the request ends with a stream sync, so the next one may rewrite it)
    uint8_t* h_meta = nullptr;
    ok = ok && ck(cudaMallocHost((void**) &h_meta, meta_bytes), "h_meta");
    // per-request scratch, kept across requests
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
    // the segment pinned for this process: the copies DMA straight from x / into out
    ok = ok && ck(cudaHostRegister(&S, sizeof(Ds41RemoteShm), cudaHostRegisterDefault), "register the segment");
    cudaStream_t st = nullptr;
    ok = ok && ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    if (!ok) return fail("buffers");

    // ---- prompt chunks: the `<name>_chunk` segment and its buffers.  The chunk's rows go up once (fp32, as the
    // engine's c_x); the entries (sorted by class, then expert, by the engine = the slab's slot order) run in sub-runs
    // of at most kMr rows: on MMQ (ds41_mmq_plan.hpp: three products over a stretch of a group's slots, PTQ1_0
    // transcoded to Q2_0 first, as run_chunk) or, for a class MMQ does not cover, on native_expert_grouped.  Kernels
    // run on cs[0]; each sub-run's rows go down on cs[1] while the next one computes.  Any failure here leaves
    // prompts to the engine alone (it finds no segment). ----
    const int64_t kMr = 4096, kTc = 16;
    const int64_t FF = G.n_ff;
    Ds41RemoteMap cmap;
    Ds41RemoteChunk* C = nullptr;
    float* d_cx = nullptr; void* d_cxq = nullptr; void* d_cscr = nullptr; float* d_cout[2] = {}; uint8_t* d_cmeta[2] = {};
    uint8_t* h_cmeta[2] = {};
    int32_t* d_ctok = nullptr;
    const size_t cmeta_bytes = 256 * 1024;
    cudaStream_t cs[2] = {};
    cudaEvent_t ev_meta[2] = {}, ev_kern[2] = {}, ev_piece[2] = {};
    bool meta_used[2] = {};
    bool need_mmvq = true;   // a class of the set without an MMQ plan (or no MMQ at all)
#if defined(DS41_HELPER_MMQ)
    namespace mmq = strata::prefill::mmq;
    void *m_gu = nullptr, *m_h = nullptr, *m_xg = nullptr, *m_xu = nullptr, *m_xd = nullptr, *m_tc = nullptr;
    int32_t* m_iota = nullptr;
    mmq::Context* m_ctx = nullptr;
    size_t tc_slot = 0;
    {
        need_mmvq = false;
        for (const Grp& g : grps) {
            if (!mplan[(size_t) g.c].ok) need_mmvq = true;
            tc_slot = std::max(tc_slot, mplan[(size_t) g.c].SL2);
        }
        if (!rset.empty()) need_mmvq = true;   // the RAM tier's experts run on native_expert_grouped from staging slots
    }
    // staging slots for the RAM tier's blobs during a prompt chunk (a sub-run stages at most this many distinct experts)
    uint8_t* d_cstage = nullptr;
    int cst_slots = 0;
    int64_t ncstaged = 0;
    size_t cst_stride = max_blob;   // a staging slot: the largest class slot (MMQ stride), 4 KiB aligned
    for (int64_t c = 0; c < NCL; ++c) cst_stride = std::max(cst_stride, (size_t) cstride[(size_t) c]);
    cst_stride = (cst_stride + 4095) & ~(size_t) 4095;
    if (a.chunk_tok > 0 && !rset.empty()) {
        cst_slots = (int) ((size_t) (a.chunk_stage_gib * 1073741824.0) / cst_stride);
        if (cst_slots < (int) kTc) cst_slots = (int) kTc;   // an MMQ sub-run stages up to kTc experts
        if (!ck(cudaMalloc(&d_cstage, (size_t) cst_slots * cst_stride + ((size_t) 1 << 20)), "chunk staging")) {
            std::fprintf(stderr, "ds41_helper: no VRAM for the chunk staging (%.1f GiB) - prompt chunks off\n", a.chunk_stage_gib);
            a.chunk_tok = 0;
            cst_slots = 0;
        } else {
            ck(cudaMemset(d_cstage, 0, (size_t) cst_slots * cst_stride + ((size_t) 1 << 20)), "chunk staging zero");   // MMQ reads a little past a blob
            std::fprintf(stderr, "ds41_helper: chunk staging: %d slots of %zu KB (%.1f GiB)\n", cst_slots, cst_stride >> 10, (double) cst_slots * cst_stride / 1073741824.0);
        }
    }
#endif
    if (a.chunk_tok > 0) {
        const int64_t cap_ent = a.chunk_tok * G.top_k / 2;   // the helper's share of a chunk's entries, at most
        const size_t bytes = Ds41RemoteChunk::size_for(a.chunk_tok, cap_ent, H);
        std::string e2;
        bool cok = cmap.open(a.name + "_chunk", true, e2, bytes);
        if (cok) {
            C = (Ds41RemoteChunk*) cmap.data();
            C->magic = 0;
            C->bytes = (int64_t) bytes;
            C->cap_tok = a.chunk_tok;
            C->cap_ent = cap_ent;
            C->n_embd = H;
            C->req.store(0);
            C->done.store(0);
        }
        cok = cok && ck(cudaHostRegister(C, bytes, cudaHostRegisterDefault), "register the chunk segment") &&
              ck(cudaMalloc(&d_cx, (size_t) a.chunk_tok * H * 4), "chunk x") &&
              ck(cudaMalloc(&d_ctok, sizeof(int32_t) * (size_t) cap_ent), "chunk tok");
        if (cok && need_mmvq)
            cok = ck(cudaMalloc(&d_cxq, (size_t) a.chunk_tok * sk::native_q8_1_bytes((int) H, 1)), "chunk xq") &&
                  ck(cudaMalloc(&d_cscr, sk::native_expert_scratch_bytes(kMr, FF)), "chunk scratch");
        for (int b = 0; b < 2 && cok; ++b)
            cok = ck(cudaMalloc(&d_cout[b], (size_t) kMr * H * 4), "chunk out") &&
                  ck(cudaMalloc(&d_cmeta[b], cmeta_bytes), "chunk meta") &&
                  ck(cudaMallocHost((void**) &h_cmeta[b], cmeta_bytes), "chunk meta host") &&
                  ck(cudaStreamCreateWithFlags(&cs[b], cudaStreamNonBlocking), "chunk stream") &&
                  ck(cudaEventCreateWithFlags(&ev_meta[b], cudaEventDisableTiming), "chunk event") &&
                  ck(cudaEventCreateWithFlags(&ev_kern[b], cudaEventDisableTiming), "chunk event") &&
                  ck(cudaEventCreateWithFlags(&ev_piece[b], cudaEventDisableTiming), "chunk event");
#if defined(DS41_HELPER_MMQ)
        if (cok && use_mmq) {
            cok = ck(cudaMalloc(&m_gu, (size_t) kMr * (size_t) (2 * FF) * 4), "m_gu") &&
                  ck(cudaMalloc(&m_h, (size_t) kMr * (size_t) FF * 4), "m_h") &&
                  ck(cudaMalloc(&m_xg, mmq::q8_bytes(kMr, H)), "m_xg") && ck(cudaMalloc(&m_xu, mmq::q8_bytes(kMr, H)), "m_xu") &&
                  ck(cudaMalloc(&m_xd, mmq::q8_bytes(kMr, FF)), "m_xd") &&
                  ck(cudaMalloc(&m_iota, sizeof(int32_t) * (size_t) (kMr + kMmqIdSlack)), "m_iota") &&   // MMQ id over-read
                  (tc_slot == 0 || ck(cudaMalloc(&m_tc, (size_t) kTc * tc_slot), "m_tc"));
            if (cok) {
                mmq::iota(m_iota, kMr + kMmqIdSlack, cs[0]);
                m_ctx = new mmq::Context();
            }
        }
#endif
        if (cok) {
            C->magic = Ds41RemoteChunk::kMagic;   // last: the engine checks it
            std::fprintf(stderr, "ds41_helper: prompt chunks up to %lld rows / %lld entries (%.2f GB segment)%s\n",
                         (long long) a.chunk_tok, (long long) cap_ent, bytes / 1e9, need_mmvq ? "" : ", every class on MMQ");
        } else {
            std::fprintf(stderr, "ds41_helper: prompt chunks off (%s)\n", e2.empty() ? "buffers" : e2.c_str());
            cudaGetLastError();
            if (C) cudaHostUnregister(C);
            C = nullptr;
            cmap.close();
        }
    }
    int64_t clast = 0, cserved = 0;
    double cbusy = 0, cents = 0;
    // one prompt-chunk request: status 0 = ok
    auto serve_chunk = [&]() -> int {
        const int64_t n = C->n, ne = C->n_ent;
        const int layer = C->layer;
        if (n <= 0 || n > C->cap_tok || ne <= 0 || ne > C->cap_ent || layer < 0 || layer >= G.n_layers) return 1;
        const int32_t* ee = C->ent_expert();
        const int32_t* et = C->ent_tok();
        for (int64_t i = 0; i < ne; ++i) {
            if (ee[i] < 0 || ee[i] >= G.n_experts || et[i] < 0 || et[i] >= n) return 2;
            if (!dptr[(size_t) (layer * G.n_experts + ee[i])] && !(cst_slots > 0 && hptr[(size_t) (layer * G.n_experts + ee[i])])) return 2;
            if (i > 0) {   // sorted by (class, expert): the sub-runs below rely on it
                const int c0 = G.cls(layer, ee[i - 1]), c1 = G.cls(layer, ee[i]);
                if (c1 < c0 || (c1 == c0 && ee[i] < ee[i - 1])) return 5;
            }
        }
        const double tc0 = now_ms();
        if (!ck(cudaMemcpyAsync(d_cx, C->x(), (size_t) n * H * 4, cudaMemcpyHostToDevice, cs[0]), "chunk x h2d") ||
            !ck(cudaMemcpyAsync(d_ctok, et, sizeof(int32_t) * (size_t) ne, cudaMemcpyHostToDevice, cs[0]), "chunk tok h2d")) return 4;
        bool xq_done = false;   // the MMVQ classes' q8_1 rows (made on first need)
        static const bool cprof = std::getenv("DS41_HELPER_PROF") != nullptr;
        if (cprof) cudaStreamSynchronize(cs[0]);
        const double tc1 = now_ms();
        int run = 0;
        // a sub-run done: rows [r0, r0 + R) of d_cout[b] down to their entries' rows of the segment
        auto rows_down = [&](int b, int64_t r0, int64_t R) -> bool {
            cudaEventRecord(ev_kern[b], cs[0]);
            cudaStreamWaitEvent(cs[1], ev_kern[b], 0);
            if (!ck(cudaMemcpyAsync(C->out() + (size_t) r0 * H, d_cout[b], (size_t) R * H * 4, cudaMemcpyDeviceToHost, cs[1]), "chunk out d2h")) return false;
            cudaEventRecord(ev_piece[b], cs[1]);
            return true;
        };
        // the next sub-run's buffers: its metadata slot free on the host, its out buffer read by the last copy
        auto take = [&]() -> int {
            const int b = run++ & 1;
            if (meta_used[b]) {
                cudaEventSynchronize(ev_meta[b]);
                cudaStreamWaitEvent(cs[0], ev_piece[b], 0);
            }
            meta_used[b] = true;
            return b;
        };
        auto al16 = [](size_t x) { return (x + 15) & ~(size_t) 15; };
        for (int64_t i0 = 0; i0 < ne;) {
            const int c = G.cls(layer, ee[i0]);
            int64_t i1 = i0;
            while (i1 < ne && G.cls(layer, ee[i1]) == c) ++i1;
#if defined(DS41_HELPER_MMQ)
            const MmqPlan& MP = mplan[(size_t) c];
            bool has_ram = false;   // a class run with RAM-tier experts
            for (int64_t i = i0; i < i1 && !has_ram; ++i) has_ram = !dptr[(size_t) (layer * G.n_experts + ee[i])];
            bool all_in_ram = has_ram && cst_slots > 0;   // ... and every expert of the run has a RAM copy (--vram-cache 1)
            for (int64_t i = i0; i < i1 && all_in_ram; ++i) all_in_ram = hptr[(size_t) (layer * G.n_experts + ee[i])] != nullptr;
            if (use_mmq && MP.ok && all_in_ram) {
                // sub-runs of at most kTc experts (dense slots 0..nq-1 in the staging area) and kMr rows; each expert's blob
                // goes up from the RAM tier on the kernel stream right before the products (the previous sub-run's kernels
                // are done with the slots by then); the products read the staging area at the class stride
                const size_t SL = (size_t) cstride[(size_t) c];
                const sk::NativeExpertLayout& GL = gll[(size_t) c];
                std::vector<int32_t> bnd;
                int64_t i = i0;
                while (i < i1) {
                    const int64_t r0 = i;
                    bnd.assign(1, 0);
                    int64_t maxr = 0;
                    int nq = 0;
                    const int b = take();
                    while (i < i1 && nq < (int) kTc) {
                        const int32_t e = ee[i];
                        int64_t k = 0;
                        while (i + k < i1 && ee[i + k] == e) ++k;
                        if (nq > 0 && (i + k - r0) > kMr) break;
                        if (nq == 0 && k > kMr) k = kMr;
                        const size_t at = (size_t) (layer * G.n_experts + e), eb = src.expert_bytes(layer, e);
                        uint8_t* slot = d_cstage + (size_t) nq * SL;   // dense at the class stride (MMQ counts it in blocks)
                        if (!ck(cudaMemcpyAsync(slot, hptr[at], eb, cudaMemcpyHostToDevice, cs[0]), "chunk stage h2d")) return 4;
                        if (SL > eb && !ck(cudaMemsetAsync(slot + eb, 0, SL - eb, cs[0]), "chunk stage pad")) return 4;   // the slab's zero padding
                        ++ncstaged;
                        bnd.push_back(bnd.back() + (int32_t) k);
                        maxr = std::max(maxr, k);
                        ++nq;
                        i += k;
                        if (nq == 1 && i < i1 && ee[i] == e) break;   // a split expert's remainder starts the next sub-run
                    }
                    const int64_t R = i - r0;
                    std::memcpy(h_cmeta[b], bnd.data(), sizeof(int32_t) * bnd.size());
                    if (!ck(cudaMemcpyAsync(d_cmeta[b], h_cmeta[b], sizeof(int32_t) * bnd.size(), cudaMemcpyHostToDevice, cs[0]), "bounds")) return 4;
                    cudaEventRecord(ev_meta[b], cs[0]);
                    const int32_t* tokr = d_ctok + r0;
                    mmq::quantize(d_cx, tokr, m_xg, MP.gt, H, H, R, cs[0]);
                    if (MP.ut != MP.gt) mmq::quantize(d_cx, tokr, m_xu, MP.ut, H, H, R, cs[0]);
                    const uint8_t* base = d_cstage;
                    uint8_t* tc = (uint8_t*) m_tc;
                    if (MP.tg) ds41_ptq1_to_q2_0_strided(base, SL, tc + MP.g2, MP.SL2, MP.g_bytes, nq, cs[0]);
                    if (MP.tu) ds41_ptq1_to_q2_0_strided(base + GL.up_off, SL, tc + MP.u2, MP.SL2, MP.u_bytes, nq, cs[0]);
                    if (MP.td) ds41_ptq1_to_q2_0_strided(base + GL.down_off, SL, tc + MP.d2, MP.SL2, MP.d_bytes, nq, cs[0]);
                    mmq::Product p;
                    p.n = nq;
                    p.bounds = (const int32_t*) d_cmeta[b];
                    p.total_rows = R;
                    p.max_rows = maxr;
                    p.w = MP.tg ? tc + MP.g2 : base; p.expert_bytes = MP.tg ? MP.SL2 : SL;
                    p.type = MP.gt; p.w_rows = FF; p.w_cols = H; p.xq = m_xg;
                    p.ids = m_iota; p.dst = (float*) m_gu; p.ld_dst = 2 * FF;
                    m_ctx->run(p, cs[0]);
                    p.w = MP.tu ? tc + MP.u2 : base + GL.up_off; p.expert_bytes = MP.tu ? MP.SL2 : SL;
                    p.type = MP.ut; p.xq = MP.ut != MP.gt ? m_xu : m_xg; p.dst = (float*) m_gu + FF;
                    m_ctx->run(p, cs[0]);
                    if (std::isfinite(GL.swiglu_limit))
                        mmq::swiglu_clamp((const float*) m_gu, (float*) m_h, R, FF, false, GL.swiglu_limit, cs[0]);
                    else
                        mmq::swiglu((const float*) m_gu, (float*) m_h, R, FF, false, cs[0]);
                    mmq::quantize((const float*) m_h, nullptr, m_xd, MP.dt, FF, FF, R, cs[0]);
                    p.w = MP.td ? tc + MP.d2 : base + GL.down_off; p.expert_bytes = MP.td ? MP.SL2 : SL;
                    p.type = MP.dt; p.w_rows = H; p.w_cols = FF; p.xq = m_xd;
                    p.ids = m_iota; p.dst = d_cout[b]; p.ld_dst = H;
                    m_ctx->run(p, cs[0]);
                    if (!rows_down(b, r0, R)) return 4;
                }
                i0 = i1;
                continue;
            }
            if (use_mmq && MP.ok && !has_ram) {
                const Grp& g = grps[(size_t) grp_of[(size_t) (layer * NCL + c)]];
                const size_t SL = (size_t) cstride[(size_t) c];
                const sk::NativeExpertLayout& GL = gll[(size_t) c];
                // sub-runs: consecutive slots (at most kTc of them, empty ones included) and at most kMr rows; an
                // expert with more rows than that runs alone, kMr at a time
                std::vector<int32_t> bnd;
                int64_t i = i0;
                while (i < i1) {
                    const int64_t r0 = i;
                    const int32_t q0 = slot_of[(size_t) (layer * G.n_experts + ee[i])];
                    int32_t qlast = q0;
                    bnd.assign(1, 0);
                    int64_t maxr = 0;
                    while (i < i1) {
                        const int32_t e = ee[i];
                        const int32_t q = slot_of[(size_t) (layer * G.n_experts + e)];
                        int64_t k = 0;
                        while (i + k < i1 && ee[i + k] == e) ++k;
                        const bool first = i == r0;
                        if (!first && (q - q0 + 1 > kTc || (i + k - r0) > kMr)) break;
                        if (first && k > kMr) k = kMr;   // a big expert: kMr rows now, the rest in the next sub-run
                        for (int32_t qq = qlast + (first ? 0 : 1); qq < q; ++qq) bnd.push_back(bnd.back());   // empty slots
                        bnd.push_back(bnd.back() + (int32_t) k);
                        maxr = std::max(maxr, k);
                        qlast = q;
                        i += k;
                        if (first && i < i1 && ee[i] == e) break;   // the split expert's remainder starts the next sub-run
                    }
                    const int64_t R = i - r0;
                    const int nq = qlast - q0 + 1;
                    const int b = take();
                    std::memcpy(h_cmeta[b], bnd.data(), sizeof(int32_t) * bnd.size());
                    if (!ck(cudaMemcpyAsync(d_cmeta[b], h_cmeta[b], sizeof(int32_t) * bnd.size(), cudaMemcpyHostToDevice, cs[0]), "bounds")) return 4;
                    cudaEventRecord(ev_meta[b], cs[0]);
                    const int32_t* tokr = d_ctok + r0;
                    mmq::quantize(d_cx, tokr, m_xg, MP.gt, H, H, R, cs[0]);
                    if (MP.ut != MP.gt) mmq::quantize(d_cx, tokr, m_xu, MP.ut, H, H, R, cs[0]);
                    const uint8_t* base = slab + g.off + (size_t) q0 * SL;
                    uint8_t* tc = (uint8_t*) m_tc;
                    if (MP.tg) ds41_ptq1_to_q2_0_strided(base, SL, tc + MP.g2, MP.SL2, MP.g_bytes, nq, cs[0]);
                    if (MP.tu) ds41_ptq1_to_q2_0_strided(base + GL.up_off, SL, tc + MP.u2, MP.SL2, MP.u_bytes, nq, cs[0]);
                    if (MP.td) ds41_ptq1_to_q2_0_strided(base + GL.down_off, SL, tc + MP.d2, MP.SL2, MP.d_bytes, nq, cs[0]);
                    mmq::Product p;
                    p.n = nq;
                    p.bounds = (const int32_t*) d_cmeta[b];
                    p.total_rows = R;
                    p.max_rows = maxr;
                    p.w = MP.tg ? tc + MP.g2 : base; p.expert_bytes = MP.tg ? MP.SL2 : SL;
                    p.type = MP.gt; p.w_rows = FF; p.w_cols = H; p.xq = m_xg;
                    p.ids = m_iota; p.dst = (float*) m_gu; p.ld_dst = 2 * FF;
                    m_ctx->run(p, cs[0]);
                    p.w = MP.tu ? tc + MP.u2 : base + GL.up_off; p.expert_bytes = MP.tu ? MP.SL2 : SL;
                    p.type = MP.ut; p.xq = MP.ut != MP.gt ? m_xu : m_xg; p.dst = (float*) m_gu + FF;
                    m_ctx->run(p, cs[0]);
                    if (std::isfinite(GL.swiglu_limit))
                        mmq::swiglu_clamp((const float*) m_gu, (float*) m_h, R, FF, false, GL.swiglu_limit, cs[0]);
                    else
                        mmq::swiglu((const float*) m_gu, (float*) m_h, R, FF, false, cs[0]);
                    mmq::quantize((const float*) m_h, nullptr, m_xd, MP.dt, FF, FF, R, cs[0]);
                    p.w = MP.td ? tc + MP.d2 : base + GL.down_off; p.expert_bytes = MP.td ? MP.SL2 : SL;
                    p.type = MP.dt; p.w_rows = H; p.w_cols = FF; p.xq = m_xd;
                    p.ids = m_iota; p.dst = d_cout[b]; p.ld_dst = H;
                    m_ctx->run(p, cs[0]);
                    if (!rows_down(b, r0, R)) return 4;
                }
                i0 = i1;
                continue;
            }
#endif
            // native_expert_grouped over the class's entries, kMr at a time; groups = runs of one expert
            if (!xq_done) {
                const size_t col = sk::native_q8_1_bytes((int) H, 1);
                for (int64_t c0 = 0; c0 < n; c0 += 8)
                    sk::native_quantize_q8_1(d_cx + (size_t) c0 * H, (uint8_t*) d_cxq + (size_t) c0 * col, (int) H,
                                             (int) std::min<int64_t>(8, n - c0), cs[0]);
                xq_done = true;
            }
            std::vector<unsigned long long> pp;
            std::vector<int32_t> pst, pdst, ptok;
            for (int64_t p0 = i0; p0 < i1;) {
                // the sub-run [p0, p1): at most kMr rows and at most cst_slots RAM-tier experts (staged below)
                int64_t p1 = p0;
                int nst = 0;
                while (p1 < i1 && p1 - p0 < kMr) {
                    if ((p1 == p0 || ee[p1] != ee[p1 - 1]) && !dptr[(size_t) (layer * G.n_experts + ee[p1])]) {
                        if (nst >= cst_slots) break;
                        ++nst;
                    }
                    ++p1;
                }
                if (p1 == p0) return 6;
                const int b = take();
                pp.clear(); pst.clear(); pdst.clear(); ptok.clear();
                int kst = 0;
                for (int64_t i = p0; i < p1; ++i) {
                    if (i == p0 || ee[i] != ee[i - 1]) {
                        const size_t at = (size_t) (layer * G.n_experts + ee[i]);
                        unsigned long long d = dptr[at];
                        if (!d) {   // a RAM-tier expert: into a staging slot, on the kernel stream (the previous sub-run's kernel is done with the slot)
                            uint8_t* ds = d_cstage + (size_t) kst++ * max_blob;
                            if (!ck(cudaMemcpyAsync(ds, hptr[at], src.expert_bytes(layer, ee[i]), cudaMemcpyHostToDevice, cs[0]), "chunk stage h2d")) return 4;
                            d = (unsigned long long) ds;
                            ++ncstaged;
                        }
                        pp.push_back(d);
                        pst.push_back((int32_t) pdst.size());
                    }
                    pdst.push_back((int32_t) (i - p0));
                    ptok.push_back(et[i]);
                }
                const int ng = (int) pp.size();
                const int64_t pne = (int64_t) pdst.size();
                pst.push_back((int32_t) pne);
                const size_t o_st = al16(8 * (size_t) ng), o_ng = al16(o_st + 4 * (size_t) (ng + 1)), o_dst = al16(o_ng + 4),
                             o_tok = al16(o_dst + 4 * (size_t) pne), at = al16(o_tok + 4 * (size_t) pne);
                if (at > cmeta_bytes) return 3;
                std::memcpy(h_cmeta[b], pp.data(), 8 * (size_t) ng);
                std::memcpy(h_cmeta[b] + o_st, pst.data(), 4 * (size_t) (ng + 1));
                std::memcpy(h_cmeta[b] + o_ng, &ng, 4);
                std::memcpy(h_cmeta[b] + o_dst, pdst.data(), 4 * (size_t) pne);
                std::memcpy(h_cmeta[b] + o_tok, ptok.data(), 4 * (size_t) pne);
                if (!ck(cudaMemcpyAsync(d_cmeta[b], h_cmeta[b], at, cudaMemcpyHostToDevice, cs[0]), "chunk meta")) return 4;
                cudaEventRecord(ev_meta[b], cs[0]);
                sk::native_expert_grouped(gll[(size_t) c], (const unsigned long long*) d_cmeta[b], (const int32_t*) (d_cmeta[b] + o_st),
                                          (const int32_t*) (d_cmeta[b] + o_ng), (const int32_t*) (d_cmeta[b] + o_dst),
                                          (const int32_t*) (d_cmeta[b] + o_tok), ng, pne, d_cxq, d_cscr, d_cout[b], cs[0], ng);
                if (!rows_down(b, p0, p1 - p0)) return 4;
                p0 = p1;
            }
            i0 = i1;
        }
        if (!ck(cudaStreamSynchronize(cs[0]), "chunk") || !ck(cudaStreamSynchronize(cs[1]), "chunk")) return 4;
        cents += (double) ne;
        if (cprof && n >= 1024)
            std::fprintf(stderr, "ds41_helper: chunk layer %d: %lld rows, %lld entries: rows up %.1f ms, sub-runs %.1f ms, %lld RAM-tier blobs staged\n", layer,
                         (long long) n, (long long) ne, tc1 - tc0, now_ms() - tc1, (long long) ncstaged);
        ncstaged = 0;
        return 0;
    };

    DBG("buffers");
    // warm-up: one launch per class the set has, so the module loading of each kernel (seconds on the first call)
    // is paid here and not by the engine's first request
    {
        const double tw = now_ms();
        std::vector<char> warm((size_t) NCL, 0);
        int nwarm = 0;
        std::vector<float> zero((size_t) H, 0.0f);
        ok = ck(cudaMemcpy(d_x, zero.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "warm x");
        sk::native_quantize_q8_1(d_x, d_xq, (int) H, 1, st);
        for (const auto& le : set) {
            const int c = G.cls(le.first, le.second);
            if (warm[(size_t) c] || !ok) continue;
            warm[(size_t) c] = 1;
            const unsigned long long p = dptr[(size_t) (le.first * G.n_experts + le.second)];
            const int32_t meta[6] = {0, 1, 1, 0, 0, 0};   // start {0, 1}, n_groups 1, dst 0, tok 0
            ok = ck(cudaMemcpy(d_meta, &p, 8, cudaMemcpyHostToDevice), "warm ptr") &&
                 ck(cudaMemcpy(d_meta + 16, meta, sizeof(meta), cudaMemcpyHostToDevice), "warm meta");
            sk::native_expert_grouped(gll[(size_t) c], (const unsigned long long*) d_meta, (const int32_t*) (d_meta + 16),
                                      (const int32_t*) (d_meta + 24), (const int32_t*) (d_meta + 28),
                                      (const int32_t*) (d_meta + 32), 1, 1, d_xq, d_scratch, d_out, st, 1);
            ++nwarm;
        }
        if (!ok || !ck(cudaStreamSynchronize(st), "warm-up")) return fail("warm-up");
        DBG("warm");
        std::fprintf(stderr, "ds41_helper: warm-up of %d classes in %.1f s\n", nwarm, (now_ms() - tw) / 1000.0);
    }
    std::fprintf(stderr, "ds41_helper: %s (CUDA%d): %zu experts, %.2f GiB (profile from %.1f GiB), loaded in %.1f s - serving '%s'\n",
                 prop.name, a.device, set.size(), held_b / 1073741824.0, a.skip_gib, (now_ms() - t0) / 1000.0, a.name.c_str());
    S.ready.store(1, std::memory_order_release);

    int64_t last = 0, served = 0;
    double busy = 0, busy_max = 0;   // since the last print
    // DS41_HELPER_PROF=1 (diagnostic, slower): a sync after each stage, wall ms per stage summed over the requests
    static const bool sprof = std::getenv("DS41_HELPER_PROF") != nullptr;
    double sp[5] = {};
    double ts = 0;
    auto stage = [&](int i) {
        if (!sprof) return;
        cudaStreamSynchronize(st);
        const double t = now_ms();
        sp[i] += t - ts;
        ts = t;
    };
    // Keep the card awake: between requests (the engine sends one every few layer calls, often 5-20 ms apart) the
    // card drops to an idle clock and a request then took 1-2 ms instead of ~0.14 ms (remote_bench with idle gaps on
    // the RTX 4080: 0 / 1 / 5 / 20 ms gap -> 135 / 152 / 953 / 1931 us median).  While idle, one expert of the set is
    // run every DS41_HELPER_WARM_US (default 100; 0 = off) on a stream and buffers of its own, so a request never waits
    // for it.  Measured (remote_bench, 0 / 5 / 20 ms gaps, median): a one-column quantize every 500 us kept the card at
    // 285 MHz (- / 576 / 1157 us); an expert every 300 us ~1 GHz (- / 584 / 663 us); every 150 us 1.7 GHz, 60 W
    // (165 / 177 / 249 us, p90 802 at 20 ms); every 100 us 2.6 GHz, ~100 W while serving (137 / 142 / 151 us).
    const double warm_us = std::getenv("DS41_HELPER_WARM_US") ? std::atof(std::getenv("DS41_HELPER_WARM_US")) : 100.0;
    // ... but only until no request came for DS41_HELPER_WARM_IDLE_S (default 10; 0 = always): then the card may idle
    // (the keep-warm holds it at ~83 W, fan on, with no one using the model); the first request after that wakes it
    const double warm_idle_ms = (std::getenv("DS41_HELPER_WARM_IDLE_S") ? std::atof(std::getenv("DS41_HELPER_WARM_IDLE_S")) : 10.0) * 1000.0;
    uint8_t* d_wmeta = nullptr;
    void *d_wxq = nullptr, *d_wscratch = nullptr;
    float* d_wout = nullptr;
    cudaStream_t wst = nullptr;
    const int wc = G.cls(set.front().first, set.front().second);
    if (warm_us > 0) {
        const unsigned long long p = dptr[(size_t) (set.front().first * G.n_experts + set.front().second)];
        const int32_t meta[6] = {0, 1, 1, 0, 0, 0};   // start {0, 1}, n_groups 1, dst 0, tok 0
        if (!ck(cudaMalloc(&d_wmeta, 64), "keep-warm meta") || !ck(cudaMemcpy(d_wmeta, &p, 8, cudaMemcpyHostToDevice), "keep-warm ptr") ||
            !ck(cudaMemcpy(d_wmeta + 16, meta, sizeof(meta), cudaMemcpyHostToDevice), "keep-warm meta") ||
            !ck(cudaMalloc(&d_wxq, sk::native_q8_1_bytes((int) H, 1)), "keep-warm xq") ||
            !ck(cudaMemcpy(d_wxq, d_xq, sk::native_q8_1_bytes((int) H, 1), cudaMemcpyDeviceToDevice), "keep-warm xq copy") ||
            !ck(cudaMalloc(&d_wscratch, sk::native_expert_scratch_bytes(1, G.n_ff)), "keep-warm scratch") ||
            !ck(cudaMalloc(&d_wout, (size_t) H * 4), "keep-warm out") ||
            !ck(cudaStreamCreateWithFlags(&wst, cudaStreamNonBlocking), "keep-warm stream"))
            return fail("keep-warm");
    }
    double last_busy = now_ms(), last_req = last_busy;
    int64_t nstaged = 0;
#ifdef _WIN32
    // the serving thread spins between requests; keep the scheduler from parking it behind the pool's workers
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    bool warm_on = true;
    for (;;) {
        int64_t r;
        while ((r = S.req.load(std::memory_order_acquire)) == last) {
            if (S.stop.load(std::memory_order_relaxed)) goto out;
            if (C) {
                const int64_t cr = C->req.load(std::memory_order_acquire);
                if (cr != clast) {
                    const double tc = now_ms();
                    const int cst = serve_chunk();
                    C->status.store(cst, std::memory_order_relaxed);
                    C->done.store(cr, std::memory_order_release);
                    clast = cr;
                    last_busy = last_req = now_ms();
                    cbusy += last_busy - tc;
                    if (++cserved % 400 == 0) {
                        std::fprintf(stderr, "ds41_helper: %lld prompt chunks: %.1f ms busy, %.0f entries each\n", (long long) cserved,
                                     cbusy / 400.0, cents / 400.0);
                        cbusy = 0;
                        cents = 0;
                    }
                    continue;
                }
            }
            if (warm_us > 0 && warm_idle_ms > 0 && warm_on != (now_ms() - last_req < warm_idle_ms)) {
                warm_on = !warm_on;
                if (std::getenv("DS41_HELPER_PROF"))
                    std::fprintf(stderr, "ds41_helper: keep-warm %s\n", warm_on ? "on" : "off (idle)");
            }
            if (warm_us > 0 && warm_on && (now_ms() - last_busy) * 1000.0 >= warm_us) {
                if (cudaStreamQuery(wst) != cudaErrorNotReady)   // never queue more than one
                    sk::native_expert_grouped(gll[(size_t) wc], (const unsigned long long*) d_wmeta, (const int32_t*) (d_wmeta + 16),
                                              (const int32_t*) (d_wmeta + 24), (const int32_t*) (d_wmeta + 28),
                                              (const int32_t*) (d_wmeta + 32), 1, 1, d_wxq, d_wscratch, d_wout, wst, 1);
                last_busy = now_ms();
            }
            ds41_remote_pause();
        }
        const double tq = now_ms();
        ts = tq;
        const int nt = S.nt, ne = S.n_ent, layer = S.layer;
        int status = 0;
        if (nt <= 0 || nt > Ds41RemoteShm::kMaxTok || ne <= 0 || ne > kMaxEnt || layer < 0 || layer >= G.n_layers) status = 1;
        // every entry's blob must be here (the engine only sends what `holds` says); the RAM tier's go up into a
        // staging slot first (H2D on the request stream, so the kernels below see them)
        stg_e.clear();
        stg_p.clear();
        ram_e.clear();
        cpu_rows.clear();
        bool cpu_ent[Ds41RemoteShm::kMaxEnt] = {};
        int ncj = 0;
        for (int i = 0; i < ne && status == 0; ++i) {   // the distinct RAM-tier experts of this request, in first-seen order
            const int32_t e = S.ent_expert[i];
            const size_t at = (size_t) (layer * G.n_experts + e);
            if (dptr[at]) continue;
            if (!hptr[at]) { status = 2; break; }
            bool have = false;
            for (int32_t q : ram_e) if (q == e) { have = true; break; }
            if (!have) ram_e.push_back(e);
        }
        if (lru)   // touch the slots this request uses FIRST, so no victim below is one of them (a NULL blob in the launch)
            for (int i = 0; i < ne; ++i) {
                const int32_t vi = vslot_of[(size_t) (layer * G.n_experts + S.ent_expert[i])];
                if (vi >= 0) vslots[(size_t) G.cls(layer, S.ent_expert[i])][(size_t) vi].used = served;
            }
        // the last k of them go to the CPU pool (classes with a CPU layout only); the rest are staged into the card
        int kc = cpool ? (int) std::lround(a.cpu_share * (double) ram_e.size()) : 0;
        for (size_t q = 0; q < ram_e.size() && status == 0; ++q) {
            const int32_t e = ram_e[q];
            const size_t at = (size_t) (layer * G.n_experts + e);
            const int c = G.cls(layer, e);
            const bool to_cpu = kc > 0 && q >= ram_e.size() - (size_t) kc && fl_ok[(size_t) c];
            if (to_cpu) {
                const cpu::NativeFmt& fc = fl[(size_t) c];
                cpu::ExpertJobMulti* jb = nullptr;
                for (int j = 0; j < ne; ++j) {
                    if (S.ent_expert[j] != e) continue;
                    if (!jb || jb->nt == cpu::MAXT) {
                        jb = &cjobs[(size_t) ncj++];
                        jb->blob = hptr[at];
                        jb->nt = 0;
                        jb->fmt = &fc;
                        for (int t = 0; t < cpu::MAXT; ++t) { jb->nact[t] = nullptr; jb->out[t] = nullptr; jb->act[t] = nullptr; }
                    }
                    const float* xr = S.x + (size_t) S.ent_tok[j] * H;
                    if (cpu::q2_native_kernels(fc.gu_type)) {
                        cpu::act_quant_any(xr, (int) H, cactq[(size_t) j]);
                        jb->act[jb->nt] = &cactq[(size_t) j];
                    } else {
                        uint8_t* slot = cnact.data() + (size_t) j * cpu::kNativeActBytes;
                        cpu::native_quant_act(fc, xr, slot);
                        jb->nact[jb->nt] = slot;
                    }
                    jb->out[jb->nt] = cparts.data() + (size_t) j * (size_t) H;
                    ++jb->nt;
                    cpu_ent[j] = true;
                    cpu_rows.push_back(j);
                }
                ++ncpu_blobs;
                continue;
            }
            if (lru) {   // an old slot of this class (unused for vram_cache_min_age requests): the blob moves in for good
                std::vector<VSlot>& vs = vslots[(size_t) c];
                int best = -1;
                for (size_t k = 0; k < vs.size(); ++k)
                    if (vs[k].used != served && served - vs[k].used >= (int64_t) a.vram_cache_min_age && (best < 0 || vs[k].used < vs[(size_t) best].used)) best = (int) k;
                if (best >= 0) {
                    VSlot& v = vs[(size_t) best];
                    if (v.at >= 0) { dptr[(size_t) v.at] = 0; vslot_of[(size_t) v.at] = -1; }   // the victim is served from RAM from now on
                    ck(cudaMemcpyAsync((void*) v.d, hptr[at], src.expert_bytes(layer, e), cudaMemcpyHostToDevice, st), "cache h2d");
                    v.at = (int32_t) at;
                    v.used = served;
                    dptr[at] = v.d;
                    vslot_of[at] = best;
                    ++nlru;
                    continue;
                }
            }
            if ((int) stg_e.size() >= kStage) { status = 5; break; }
            stg_e.push_back(e);
            stg_p.push_back(0);   // staged below, once the CPU jobs are off
        }
        // the CPU jobs run on the pool while this thread issues the card's copies (a WDDM H2D from pinned memory holds
        // the issuing thread for about its transfer time, so issuing first would serialize the two)
        std::future<void> cpu_done;
        if (ncj > 0 && status == 0) cpu_done = std::async(std::launch::async, [&]() { cpool->run_split_multi_native(*cjobs[0].fmt, cjobs.data(), ncj); });
        for (size_t q = 0; q < stg_e.size() && status == 0; ++q) {
            uint8_t* d = d_stage + q * max_blob;
            ck(cudaMemcpyAsync(d, hptr[(size_t) (layer * G.n_experts + stg_e[q])], src.expert_bytes(layer, stg_e[q]), cudaMemcpyHostToDevice, st), "stage h2d");
            stg_p[q] = (unsigned long long) d;
        }
        auto blob_of = [&](int32_t e) -> unsigned long long {
            const unsigned long long d = dptr[(size_t) (layer * G.n_experts + e)];
            if (d) return d;
            for (size_t q = 0; q < stg_e.size(); ++q) if (stg_e[q] == e) return stg_p[q];
            return 0;
        };
        nstaged += (int64_t) stg_e.size();
        if (status == 0) {
            ck(cudaMemcpyAsync(d_x, S.x, (size_t) nt * H * 4, cudaMemcpyHostToDevice, st), "x h2d");
            sk::native_quantize_q8_1(d_x, d_xq, (int) H, std::min(nt, 8), st);
            if (nt > 8) {   // as the tier's quantize_rows_q8_1: 8 columns per call
                const size_t col = sk::native_q8_1_bytes((int) H, 1);
                for (int c0 = 8; c0 < nt; c0 += 8)
                    sk::native_quantize_q8_1(d_x + (size_t) c0 * H, (uint8_t*) d_xq + (size_t) c0 * col, (int) H, std::min(8, nt - c0), st);
            }
            stage(0);   // x up + quantize
            // one grouped launch per format class: groups = distinct experts in first-seen order, entries = the
            // requests' entries (output row i = entry i); the metadata of every class goes up in one copy
            struct Launch { int c, ng; size_t ne, o_ptr, o_st, o_ng, o_dst, o_tok; };
            static std::vector<Launch> launches;
            launches.clear();
            auto al16 = [](size_t x) { return (x + 15) & ~(size_t) 15; };
            bool taken[Ds41RemoteShm::kMaxEnt] = {};
            for (int i = 0; i < ne; ++i) taken[i] = cpu_ent[i];
            size_t meta_at = 0;
            for (int i0 = 0; i0 < ne && status == 0; ++i0) {
                if (taken[i0]) continue;
                const int c = G.cls(layer, S.ent_expert[i0]);
                ptr.clear(); start.clear(); dst.clear(); tok.clear();
                for (int i = i0; i < ne; ++i) {   // i: the first entry of a not yet grouped expert of class c
                    if (taken[i] || G.cls(layer, S.ent_expert[i]) != c) continue;
                    const int32_t e = S.ent_expert[i];
                    ptr.push_back(blob_of(e));
                    start.push_back((int32_t) dst.size());
                    for (int j = i; j < ne; ++j)
                        if (!taken[j] && S.ent_expert[j] == e) { dst.push_back(j); tok.push_back(S.ent_tok[j]); taken[j] = true; }
                }
                Launch L;
                L.c = c;
                L.ng = (int) ptr.size();
                L.ne = dst.size();
                start.push_back((int32_t) L.ne);
                L.o_ptr = meta_at;
                L.o_st = al16(L.o_ptr + 8 * (size_t) L.ng);
                L.o_ng = al16(L.o_st + 4 * (size_t) (L.ng + 1));
                L.o_dst = al16(L.o_ng + 4);
                L.o_tok = al16(L.o_dst + 4 * L.ne);
                meta_at = al16(L.o_tok + 4 * L.ne);
                if (meta_at > meta_bytes) { status = 3; break; }
                std::memcpy(h_meta + L.o_ptr, ptr.data(), 8 * (size_t) L.ng);
                std::memcpy(h_meta + L.o_st, start.data(), 4 * (size_t) (L.ng + 1));
                std::memcpy(h_meta + L.o_ng, &L.ng, 4);
                std::memcpy(h_meta + L.o_dst, dst.data(), 4 * L.ne);
                std::memcpy(h_meta + L.o_tok, tok.data(), 4 * L.ne);
                launches.push_back(L);
            }
            stage(1);   // the groups on the host
            if (status == 0) ck(cudaMemcpyAsync(d_meta, h_meta, meta_at, cudaMemcpyHostToDevice, st), "meta");
            stage(2);   // metadata up
            for (const Launch& L : launches) {
                if (status != 0) break;
                sk::native_expert_grouped(gll[(size_t) L.c], (const unsigned long long*) (d_meta + L.o_ptr),
                                          (const int32_t*) (d_meta + L.o_st), (const int32_t*) (d_meta + L.o_ng),
                                          (const int32_t*) (d_meta + L.o_dst), (const int32_t*) (d_meta + L.o_tok), L.ng,
                                          (int64_t) L.ne, d_xq, d_scratch, d_out, st, L.ng);
            }
            stage(3);   // the grouped launches
            if (status == 0) ck(cudaMemcpyAsync(S.out, d_out, (size_t) ne * H * 4, cudaMemcpyDeviceToHost, st), "out d2h");
            if (!ck(cudaStreamSynchronize(st), "request")) status = 4;
            if (cpu_done.valid()) cpu_done.wait();
            for (int32_t j : cpu_rows) std::memcpy(S.out + (size_t) j * H, cparts.data() + (size_t) j * (size_t) H, (size_t) H * 4);
            stage(4);   // rows down
        }
        if (cpu_done.valid()) cpu_done.wait();
        S.status.store(status, std::memory_order_relaxed);
        S.done.store(r, std::memory_order_release);
        last = r;
        last_busy = last_req = now_ms();
        ++served;
        const double bt = now_ms() - tq;
        busy += bt;
        busy_max = std::max(busy_max, bt);
        if (served % 20000 == 0) {
            std::fprintf(stderr, "ds41_helper: %lld requests: last 20000 %.1f us busy each (max %.1f), %lld RAM-tier blobs staged, %lld on the CPU, %lld moved into VRAM\n", (long long) served,
                         1000.0 * busy / 20000.0, 1000.0 * busy_max, (long long) nstaged, (long long) ncpu_blobs, (long long) nlru);
            nstaged = 0;
            ncpu_blobs = 0;
            nlru = 0;
            if (sprof) {
                std::fprintf(stderr, "ds41_helper: us/request: x+quant %.1f, groups %.1f, meta %.1f, kernels %.1f, rows %.1f\n",
                             sp[0] / 20.0, sp[1] / 20.0, sp[2] / 20.0, sp[3] / 20.0, sp[4] / 20.0);
                for (double& v : sp) v = 0;
            }
            busy = 0;
            busy_max = 0;
        }
    }
out:
    std::fprintf(stderr, "ds41_helper: stop after %lld requests\n", (long long) served);
    if (C) cudaHostUnregister(C);
    cudaHostUnregister(&S);
    return 0;
}
