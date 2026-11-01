// ds41_remote_tcp.cpp - carries Xolotl's STRATA_DS41_REMOTE helper protocol (ds41_remote.hpp, shared memory) over TCP,
// so ds41_helper can run on another machine (two-box experiment: engine on llm-server, helper on the gaming PC's RTX 5080).
//
//   helper side (next to ds41_helper, Windows or Linux):
//       ds41_remote_tcp --helper-side [--name ds41_remote] (--listen 5041 | --connect HOST:5041)
//     opens the helper's segment like the engine would, waits for ready, then serves one engine at a time.
//   engine side (next to ds41_generate):
//       ds41_remote_tcp --engine-side [--name ds41_remote] (--connect 10.0.0.2:5041 | --listen 5041)
//     creates the segment the engine opens (STRATA_DS41_REMOTE=<name>), fills it from the helper's hello, serves ready=1,
//     and forwards each request: header + entries + x rows up, status + out rows down.
//
// Wire format (little-endian, host order on x86):
//   hello  (helper -> engine): Hello, then holds[n_layers * n_experts] bytes
//   request(engine -> helper): Req, ent_expert[n_ent] i32, ent_tok[n_ent] i32, x[nt * n_embd] f32 (or f16 with --f16)
//   reply  (helper -> engine): Rep, out[n_ent * n_embd] f32 (or f16) when status == 0
// No prompt-chunk segment (<name>_chunk): the engine then runs prompts without the helper (its message says so); use
// STRATA_DS41_REMOTE_DUP=1 so the RAM arena keeps the helper's experts for prompts.
//
// Build:  Linux   g++ -O2 -std=c++17 -I<xolotl>/tools/v41/engine ds41_remote_tcp.cpp -o ds41_remote_tcp -lpthread -lrt
//         Windows cl /O2 /std:c++17 /EHsc /I<xolotl>\tools\v41\engine ds41_remote_tcp.cpp ws2_32.lib
#ifdef _WIN32
// winsock2 must come before windows.h (which ds41_remote.hpp includes)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif
#include "ds41_remote.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
typedef SOCKET sock_t;
#define SOCK_BAD INVALID_SOCKET
static void sock_close(sock_t s) { closesocket(s); }
static int sock_err() { return WSAGetLastError(); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
typedef int sock_t;
#define SOCK_BAD (-1)
static void sock_close(sock_t s) { ::close(s); }
static int sock_err() { return errno; }
#endif

using namespace strata::v41;

namespace {

struct Hello {
    uint32_t magic = 0x44533454;   // "DS4T"
    uint32_t version = 2;
    int64_t n_layers = 0, n_experts = 0, n_embd = 0, held = 0, held_bytes = 0;
    int32_t f16 = 0, pad = 0;
    int64_t cap_tok = 0, cap_ent = 0;   // the helper's prompt-chunk segment (0 = none)
};
struct Req { int32_t kind, layer, nt, n_ent, f16, pad; };    // kind 1 = a layer call (nt rows), 2 = a prompt chunk (nt = rows)
struct Rep { int32_t status, n_ent, f16, pad; };

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool send_all(sock_t s, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
#ifdef _WIN32
        const int k = send(s, c, (int) std::min<size_t>(n, 1 << 30), 0);
#else
        const ssize_t k = send(s, c, n, MSG_NOSIGNAL);
#endif
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}
bool recv_all(sock_t s, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
#ifdef _WIN32
        const int k = recv(s, c, (int) std::min<size_t>(n, 1 << 30), 0);
#else
        const ssize_t k = recv(s, c, n, MSG_WAITALL);
#endif
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}
void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof(one));
    int buf = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof(buf));
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof(buf));
}

// f16 <-> f32 (round to nearest even), for --f16
uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t) (sign | h);
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, x;
    if (exp == 0) {
        if (man == 0) x = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --exp; } man &= 0x3ffu; x = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) x = sign | 0x7f800000u | (man << 13);
    else x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &x, 4);
    return f;
}

void to_f16(const float* f, uint16_t* h, size_t n) {
    const int nth = n > (1u << 20) ? 4 : 1;
    std::vector<std::thread> th;
    for (int q = 1; q < nth; ++q) th.emplace_back([=]() { for (size_t i = n * q / nth; i < n * (q + 1) / nth; ++i) h[i] = f32_to_f16(f[i]); });
    for (size_t i = 0; i < n / nth; ++i) h[i] = f32_to_f16(f[i]);
    for (auto& t : th) t.join();
}
void to_f32(const uint16_t* h, float* f, size_t n) {
    const int nth = n > (1u << 20) ? 4 : 1;
    std::vector<std::thread> th;
    for (int q = 1; q < nth; ++q) th.emplace_back([=]() { for (size_t i = n * q / nth; i < n * (q + 1) / nth; ++i) f[i] = f16_to_f32(h[i]); });
    for (size_t i = 0; i < n / nth; ++i) f[i] = f16_to_f32(h[i]);
    for (auto& t : th) t.join();
}

struct Args {
    bool helper_side = false, engine_side = false, f16 = false, chunk_f32 = false, no_chunk = false;
    std::string name = "ds41_remote", connect;
    int listen = 0;
};
bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "--helper-side") a.helper_side = true;
        else if (k == "--engine-side") a.engine_side = true;
        else if (k == "--name") a.name = next();
        else if (k == "--listen") a.listen = std::atoi(next().c_str());
        else if (k == "--connect") a.connect = next();
        else if (k == "--f16") a.f16 = true;
        else if (k == "--chunk-f32") a.chunk_f32 = true;   // prompt-chunk rows as f32 (default f16: half the bytes)
        else if (k == "--no-chunk") a.no_chunk = true;     // do not carry the prompt-chunk segment
        else { std::fprintf(stderr, "ds41_remote_tcp: unknown argument %s\n", k.c_str()); return false; }
    }
    return a.helper_side != a.engine_side && (a.listen > 0) != !a.connect.empty();
}

// spin on an atomic until it differs from `last`, or stop is set; tight for kSpinMs after the last event, then short naps
// (a decode's requests come 2-5 ms apart; on Windows a nap is a scheduler tick unless timeBeginPeriod(1), so the tight
// window must cover the gap: a 15 ms nap in the middle of a decode was the 17-19 ms round-trip spikes)
static double kSpinMs = 50.0;
template <class F>
void wait_change(F changed, const std::atomic<int32_t>* stop, double& last_event_ms) {
    for (;;) {
        if (changed()) { last_event_ms = now_ms(); return; }
        if (stop && stop->load(std::memory_order_relaxed)) return;
        if (now_ms() - last_event_ms > kSpinMs) std::this_thread::sleep_for(std::chrono::microseconds(200));
        else ds41_remote_pause();
    }
}

// one TCP link: --listen PORT accepts the first peer, --connect HOST:PORT retries every 5 s until the peer is up
sock_t open_link(const Args& a) {
    if (a.listen > 0) {
        static sock_t ls = SOCK_BAD;
        if (ls == SOCK_BAD) {
            ls = socket(AF_INET, SOCK_STREAM, 0);
            int one = 1;
            setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof(one));
            sockaddr_in sa{};
            sa.sin_family = AF_INET;
            sa.sin_addr.s_addr = htonl(INADDR_ANY);
            sa.sin_port = htons((uint16_t) a.listen);
            if (ls == SOCK_BAD) { std::fprintf(stderr, "ds41_remote_tcp: socket: %d\n", sock_err()); return SOCK_BAD; }
            int tries = 0;
            while (bind(ls, (sockaddr*) &sa, sizeof(sa)) != 0) {   // the previous listener may still be going away
                if (++tries > 30) { std::fprintf(stderr, "ds41_remote_tcp: bind %d: %d\n", a.listen, sock_err()); return SOCK_BAD; }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            if (listen(ls, 1) != 0) { std::fprintf(stderr, "ds41_remote_tcp: listen %d: %d\n", a.listen, sock_err()); return SOCK_BAD; }
            std::fprintf(stderr, "ds41_remote_tcp: listening on port %d\n", a.listen);
        }
        for (;;) {
            sockaddr_in ca{};
#ifdef _WIN32
            int cl = sizeof(ca);
#else
            socklen_t cl = sizeof(ca);
#endif
            sock_t c = accept(ls, (sockaddr*) &ca, &cl);
            if (c == SOCK_BAD) { std::fprintf(stderr, "ds41_remote_tcp: accept: %d\n", sock_err()); continue; }
            char ip[64];
            inet_ntop(AF_INET, &ca.sin_addr, ip, sizeof(ip));
            std::fprintf(stderr, "ds41_remote_tcp: peer connected from %s\n", ip);
            tune(c);
            return c;
        }
    }
    const size_t colon = a.connect.rfind(':');
    if (colon == std::string::npos) { std::fprintf(stderr, "ds41_remote_tcp: --connect HOST:PORT\n"); return SOCK_BAD; }
    const std::string host = a.connect.substr(0, colon), port = a.connect.substr(colon + 1);
    for (;;) {
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) {
            std::fprintf(stderr, "ds41_remote_tcp: cannot resolve %s\n", host.c_str());
            return SOCK_BAD;
        }
        sock_t c = socket(AF_INET, SOCK_STREAM, 0);
        const bool ok = c != SOCK_BAD && connect(c, res->ai_addr, (int) res->ai_addrlen) == 0;
        freeaddrinfo(res);
        if (ok) { std::fprintf(stderr, "ds41_remote_tcp: connected to %s:%s\n", host.c_str(), port.c_str()); tune(c); return c; }
        if (c != SOCK_BAD) sock_close(c);
        std::fprintf(stderr, "ds41_remote_tcp: connect %s:%s failed (%d) - retrying in 5 s\n", host.c_str(), port.c_str(), sock_err());
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

// ---- helper side -----------------------------------------------------------------------------------------------
int helper_side(const Args& a) {
    std::string err;
    Ds41RemoteMap map;
    // the helper creates the segment; wait for it (it may still be loading) then for ready
    for (;;) {
        if (map.open(a.name, false, err, sizeof(Ds41RemoteShm))) break;
        std::fprintf(stderr, "ds41_remote_tcp: %s - waiting for ds41_helper\n", err.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    Ds41RemoteShm& S = *map.shm();
    while (S.ready.load(std::memory_order_acquire) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (S.ready.load() != 1) { std::fprintf(stderr, "ds41_remote_tcp: the helper failed (ready=%d)\n", (int) S.ready.load()); return 1; }
    const int64_t H = S.n_embd, NP = S.n_layers * S.n_experts;
    std::fprintf(stderr, "ds41_remote_tcp: helper holds %lld experts (%.2f GiB), %lld layers x %lld experts, n_embd %lld\n",
                 (long long) S.held, S.held_bytes / 1073741824.0, (long long) S.n_layers, (long long) S.n_experts, (long long) H);
    // the helper's prompt-chunk segment (its --chunk-tok > 0)
    Ds41RemoteMap cmap;
    Ds41RemoteChunk* C = nullptr;
    if (!a.no_chunk && cmap.open(a.name + "_chunk", false, err, 0)) {
        C = (Ds41RemoteChunk*) cmap.data();
        if (C->magic != Ds41RemoteChunk::kMagic || C->n_embd != H) { std::fprintf(stderr, "ds41_remote_tcp: the chunk segment does not match - not carried\n"); C = nullptr; }
        else std::fprintf(stderr, "ds41_remote_tcp: prompt chunks too (up to %lld rows, %lld entries)\n", (long long) C->cap_tok, (long long) C->cap_ent);
    }
    std::vector<uint16_t> ch16;
    std::vector<uint8_t> creq;

    std::vector<uint16_t> h16((size_t) std::max(Ds41RemoteShm::kMaxTok, Ds41RemoteShm::kMaxEnt) * (size_t) H);
    std::vector<uint8_t> reqbuf;
    int64_t last = S.done.load();
    for (;;) {
        sock_t c = open_link(a);
        if (c == SOCK_BAD) return 1;
        Hello hl;
        hl.n_layers = S.n_layers; hl.n_experts = S.n_experts; hl.n_embd = H; hl.held = S.held; hl.held_bytes = S.held_bytes;
        hl.f16 = a.f16 ? 1 : 0;
        hl.cap_tok = C ? C->cap_tok : 0;
        hl.cap_ent = C ? C->cap_ent : 0;
        if (!send_all(c, &hl, sizeof(hl)) || !send_all(c, S.holds, (size_t) NP)) { sock_close(c); continue; }
        int64_t served = 0;
        double busy = 0, t0 = now_ms();
        for (;;) {
            Req rq;
            if (!recv_all(c, &rq, sizeof(rq))) break;
            if (rq.layer < 0) break;   // engine says goodbye
            if (rq.kind == 2) {   // ---- a prompt chunk: into the helper's chunk segment, wait, rows back ----
                if (!C || rq.nt <= 0 || rq.nt > C->cap_tok || rq.n_ent <= 0 || rq.n_ent > C->cap_ent) {
                    std::fprintf(stderr, "ds41_remote_tcp: bad chunk request (rows %d entries %d)\n", rq.nt, rq.n_ent);
                    break;
                }
                const size_t xb = rq.f16 ? (size_t) rq.nt * H * 2 : (size_t) rq.nt * H * 4;
                creq.resize((size_t) rq.n_ent * 8 + xb);
                if (!recv_all(c, creq.data(), creq.size())) break;
                const double ta = now_ms();
                std::memcpy(C->ent_expert(), creq.data(), (size_t) rq.n_ent * 4);
                std::memcpy(C->ent_tok(), creq.data() + (size_t) rq.n_ent * 4, (size_t) rq.n_ent * 4);
                if (rq.f16) to_f32((const uint16_t*) (creq.data() + (size_t) rq.n_ent * 8), C->x(), (size_t) rq.nt * H);
                else std::memcpy(C->x(), creq.data() + (size_t) rq.n_ent * 8, xb);
                C->layer = rq.layer; C->n = rq.nt; C->n_ent = rq.n_ent;
                const int64_t cr = C->req.load() + 1;
                C->req.store(cr, std::memory_order_release);
                double ev = now_ms();
                wait_change([&] { return C->done.load(std::memory_order_acquire) == cr; }, nullptr, ev);
                Rep rp{C->status.load(std::memory_order_relaxed), rq.n_ent, rq.f16, 0};
                const double tb = now_ms();
                if (!send_all(c, &rp, sizeof(rp))) break;
                if (rp.status == 0) {
                    const size_t ne = (size_t) rq.n_ent * H;
                    if (rq.f16) { ch16.resize(ne); to_f16(C->out(), ch16.data(), ne); if (!send_all(c, ch16.data(), ne * 2)) break; }
                    else if (!send_all(c, C->out(), ne * 4)) break;
                }
                if (rq.nt >= 1024)
                    std::fprintf(stderr, "ds41_remote_tcp: chunk layer %d: %d rows, %d entries: helper %.0f ms, rows out %.0f ms\n", rq.layer, rq.nt,
                                 rq.n_ent, tb - ta, now_ms() - tb);
                continue;
            }
            if (rq.nt <= 0 || rq.nt > Ds41RemoteShm::kMaxTok || rq.n_ent <= 0 || rq.n_ent > Ds41RemoteShm::kMaxEnt) {
                std::fprintf(stderr, "ds41_remote_tcp: bad request (layer %d nt %d n_ent %d)\n", rq.layer, rq.nt, rq.n_ent);
                break;
            }
            const size_t xb = rq.f16 ? (size_t) rq.nt * H * 2 : (size_t) rq.nt * H * 4;
            reqbuf.resize((size_t) rq.n_ent * 8 + xb);
            if (!recv_all(c, reqbuf.data(), reqbuf.size())) break;
            const double ta = now_ms();
            std::memcpy(S.ent_expert, reqbuf.data(), (size_t) rq.n_ent * 4);
            std::memcpy(S.ent_tok, reqbuf.data() + (size_t) rq.n_ent * 4, (size_t) rq.n_ent * 4);
            if (rq.f16) {
                const uint16_t* h = (const uint16_t*) (reqbuf.data() + (size_t) rq.n_ent * 8);
                for (size_t i = 0; i < (size_t) rq.nt * H; ++i) S.x[i] = f16_to_f32(h[i]);
            } else std::memcpy(S.x, reqbuf.data() + (size_t) rq.n_ent * 8, xb);
            S.layer = rq.layer; S.nt = rq.nt; S.n_ent = rq.n_ent;
            const int64_t r = S.req.load() + 1;
            S.req.store(r, std::memory_order_release);
            double ev = now_ms();
            wait_change([&] { return S.done.load(std::memory_order_acquire) == r; }, nullptr, ev);
            last = r;
            Rep rp{S.status.load(std::memory_order_relaxed), rq.n_ent, rq.f16, 0};
            const size_t ob = rp.status == 0 ? (rq.f16 ? (size_t) rq.n_ent * H * 2 : (size_t) rq.n_ent * H * 4) : 0;
            busy += now_ms() - ta;
            if (!send_all(c, &rp, sizeof(rp))) break;
            if (ob) {
                if (rq.f16) {
                    for (size_t i = 0; i < (size_t) rq.n_ent * H; ++i) h16[i] = f32_to_f16(S.out[i]);
                    if (!send_all(c, h16.data(), ob)) break;
                } else if (!send_all(c, S.out, ob)) break;
            }
            if (++served % 20000 == 0) {
                std::fprintf(stderr, "ds41_remote_tcp: %lld requests, helper %.1f us each, %.1f req/s\n", (long long) served,
                             1000.0 * busy / 20000.0, 20000.0 * 1000.0 / (now_ms() - t0));
                busy = 0; t0 = now_ms();
            }
        }
        sock_close(c);
        std::fprintf(stderr, "ds41_remote_tcp: engine disconnected after %lld requests\n", (long long) served);
    }
    (void) last;
}

// ---- engine side -----------------------------------------------------------------------------------------------
int engine_side(const Args& a) {
    std::string err;
    Ds41RemoteMap map;
    if (!map.open(a.name, true, err, sizeof(Ds41RemoteShm))) { std::fprintf(stderr, "ds41_remote_tcp: %s\n", err.c_str()); return 1; }
    Ds41RemoteShm& S = *map.shm();
    std::memset(&S, 0, sizeof(Ds41RemoteShm));   // ready 0: the engine waits
    S.magic = Ds41RemoteShm::kMagic;
    S.version = Ds41RemoteShm::kVersion;

    sock_t c = open_link(a);
    if (c == SOCK_BAD) return 1;
    Hello hl;
    if (!recv_all(c, &hl, sizeof(hl)) || hl.magic != 0x44533454 || hl.version != 2) {
        std::fprintf(stderr, "ds41_remote_tcp: bad hello from the helper side\n");
        return 1;
    }
    if (hl.n_embd > Ds41RemoteShm::kMaxEmbd || hl.n_layers * hl.n_experts > Ds41RemoteShm::kMaxPairs) {
        std::fprintf(stderr, "ds41_remote_tcp: the helper's model exceeds the segment's limits\n");
        return 1;
    }
    const int64_t H = hl.n_embd, NP = hl.n_layers * hl.n_experts;
    if (!recv_all(c, S.holds, (size_t) NP)) { std::fprintf(stderr, "ds41_remote_tcp: hello holds[] cut short\n"); return 1; }
    const bool f16 = a.f16 || hl.f16;
    S.n_layers = hl.n_layers; S.n_experts = hl.n_experts; S.n_embd = H; S.held = hl.held; S.held_bytes = hl.held_bytes;
    // the prompt-chunk segment the engine opens, sized like the helper's
    Ds41RemoteMap cmap;
    Ds41RemoteChunk* C = nullptr;
    const bool cf16 = !a.chunk_f32;
    if (!a.no_chunk && hl.cap_tok > 0 && hl.cap_ent > 0) {
        const size_t cb = Ds41RemoteChunk::size_for(hl.cap_tok, hl.cap_ent, H);
        if (cmap.open(a.name + "_chunk", true, err, cb)) {
            C = (Ds41RemoteChunk*) cmap.data();
            std::memset(C, 0, Ds41RemoteChunk::kHead);
            C->bytes = (int64_t) cb; C->cap_tok = hl.cap_tok; C->cap_ent = hl.cap_ent; C->n_embd = H;
            C->req.store(0); C->done.store(0); C->status.store(0);
            C->magic = Ds41RemoteChunk::kMagic;
            std::fprintf(stderr, "ds41_remote_tcp: prompt-chunk segment %s_chunk: %lld rows, %lld entries, %.0f MB (%s rows on the wire)\n",
                         a.name.c_str(), (long long) hl.cap_tok, (long long) hl.cap_ent, cb / 1e6, cf16 ? "f16" : "f32");
        } else std::fprintf(stderr, "ds41_remote_tcp: %s - prompts run without the helper\n", err.c_str());
    }
    std::vector<uint8_t> cbuf;
    std::vector<uint16_t> ch16;
    int64_t clast = C ? C->req.load() : 0;
    S.ready.store(1, std::memory_order_release);
    std::fprintf(stderr, "ds41_remote_tcp: helper holds %lld experts (%.2f GiB); segment %s ready (%s rows)\n",
                 (long long) hl.held, hl.held_bytes / 1073741824.0, a.name.c_str(), f16 ? "f16" : "f32");

    std::vector<uint8_t> buf;
    std::vector<uint16_t> h16((size_t) Ds41RemoteShm::kMaxEnt * (size_t) H);
    int64_t last = S.req.load(), served = 0;
    double ev = now_ms(), rtt = 0, rtt_max = 0, t0 = now_ms(), rtt_all = 0, ent_all = 0, bytes_all = 0;
    int64_t n_all = 0;
    for (;;) {
        int64_t r;
        bool chunk = false;
        wait_change([&] { chunk = C && C->req.load(std::memory_order_acquire) != clast; return chunk || (r = S.req.load(std::memory_order_acquire)) != last; }, &S.stop, ev);
        if (chunk) {   // ---- a prompt chunk: header + entries + rows out, rows back ----
            const int64_t cr = C->req.load(std::memory_order_acquire);
            const int n = (int) C->n, ne = C->n_ent, layer = C->layer;
            int status = 1;
            const double ta = now_ms();
            if (n > 0 && n <= C->cap_tok && ne > 0 && ne <= C->cap_ent && layer >= 0 && layer < hl.n_layers) {
                const size_t xb = cf16 ? (size_t) n * H * 2 : (size_t) n * H * 4;
                cbuf.resize(sizeof(Req) + (size_t) ne * 8 + xb);
                Req rq{2, layer, n, ne, cf16 ? 1 : 0, 0};
                std::memcpy(cbuf.data(), &rq, sizeof(rq));
                std::memcpy(cbuf.data() + sizeof(Req), C->ent_expert(), (size_t) ne * 4);
                std::memcpy(cbuf.data() + sizeof(Req) + (size_t) ne * 4, C->ent_tok(), (size_t) ne * 4);
                if (cf16) to_f16(C->x(), (uint16_t*) (cbuf.data() + sizeof(Req) + (size_t) ne * 8), (size_t) n * H);
                else std::memcpy(cbuf.data() + sizeof(Req) + (size_t) ne * 8, C->x(), xb);
                Rep rp;
                if (!send_all(c, cbuf.data(), cbuf.size()) || !recv_all(c, &rp, sizeof(rp))) { std::fprintf(stderr, "ds41_remote_tcp: connection lost\n"); S.ready.store(-1); return 1; }
                status = rp.status;
                if (status == 0) {
                    const size_t no = (size_t) ne * H;
                    if (rp.f16) {
                        ch16.resize(no);
                        if (!recv_all(c, ch16.data(), no * 2)) { std::fprintf(stderr, "ds41_remote_tcp: connection lost\n"); S.ready.store(-1); return 1; }
                        to_f32(ch16.data(), C->out(), no);
                    } else if (!recv_all(c, C->out(), no * 4)) { std::fprintf(stderr, "ds41_remote_tcp: connection lost\n"); S.ready.store(-1); return 1; }
                }
            }
            C->status.store(status, std::memory_order_relaxed);
            C->done.store(cr, std::memory_order_release);
            clast = cr;
            if (n >= 1024) std::fprintf(stderr, "ds41_remote_tcp: chunk layer %d: %d rows, %d entries, status %d: %.0f ms round trip\n", layer, n, ne, status, now_ms() - ta);
            ev = now_ms();
            continue;
        }
        if (S.stop.load(std::memory_order_relaxed)) {
            // the engine exits; keep serving the next engine (the real helper does the same)
            std::fprintf(stderr, "ds41_remote_tcp: engine stopped after %lld requests: %.1f entries/request, round trip %.0f us avg, %.1f MB moved - waiting for the next\n",
                         (long long) served, n_all ? ent_all / n_all : 0.0, n_all ? 1000.0 * rtt_all / n_all : 0.0, bytes_all / 1e6);
            rtt_all = ent_all = bytes_all = 0; n_all = 0; served = 0;
            S.stop.store(0);
            ev = now_ms();
            continue;
        }
        const double ta = now_ms();
        const int nt = S.nt, ne = S.n_ent, layer = S.layer;
        int status = 1;
        if (nt > 0 && nt <= Ds41RemoteShm::kMaxTok && ne > 0 && ne <= Ds41RemoteShm::kMaxEnt && layer >= 0 && layer < hl.n_layers) {
            const size_t xb = f16 ? (size_t) nt * H * 2 : (size_t) nt * H * 4;
            buf.resize(sizeof(Req) + (size_t) ne * 8 + xb);
            Req rq{1, layer, nt, ne, f16 ? 1 : 0, 0};
            std::memcpy(buf.data(), &rq, sizeof(rq));
            std::memcpy(buf.data() + sizeof(Req), S.ent_expert, (size_t) ne * 4);
            std::memcpy(buf.data() + sizeof(Req) + (size_t) ne * 4, S.ent_tok, (size_t) ne * 4);
            if (f16) {
                uint16_t* h = (uint16_t*) (buf.data() + sizeof(Req) + (size_t) ne * 8);
                for (size_t i = 0; i < (size_t) nt * H; ++i) h[i] = f32_to_f16(S.x[i]);
            } else std::memcpy(buf.data() + sizeof(Req) + (size_t) ne * 8, S.x, xb);
            Rep rp;
            if (!send_all(c, buf.data(), buf.size()) || !recv_all(c, &rp, sizeof(rp))) {
                std::fprintf(stderr, "ds41_remote_tcp: connection lost\n");
                S.ready.store(-1);
                return 1;
            }
            status = rp.status;
            if (status == 0) {
                if (rp.f16) {
                    if (!recv_all(c, h16.data(), (size_t) ne * H * 2)) { std::fprintf(stderr, "ds41_remote_tcp: connection lost\n"); S.ready.store(-1); return 1; }
                    for (size_t i = 0; i < (size_t) ne * H; ++i) S.out[i] = f16_to_f32(h16[i]);
                } else if (!recv_all(c, S.out, (size_t) ne * H * 4)) { std::fprintf(stderr, "ds41_remote_tcp: connection lost\n"); S.ready.store(-1); return 1; }
            }
        }
        S.status.store(status, std::memory_order_relaxed);
        S.done.store(r, std::memory_order_release);
        last = r;
        const double d = now_ms() - ta;
        rtt += d;
        rtt_max = std::max(rtt_max, d);
        rtt_all += d; ++n_all; ent_all += ne; bytes_all += (double) buf.size() + 16.0 + (f16 ? 2.0 : 4.0) * (double) ne * (double) H;
        if (++served % 500 == 0) {
            std::fprintf(stderr, "ds41_remote_tcp: %lld requests: round trip %.0f us avg, %.0f us max; %.0f req/s\n", (long long) served,
                         1000.0 * rtt / 500.0, 1000.0 * rtt_max, 500.0 * 1000.0 / (now_ms() - t0));
            rtt = rtt_max = 0;
            t0 = now_ms();
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: ds41_remote_tcp --helper-side (--listen PORT | --connect HOST:PORT) [--name ds41_remote] [--f16]\n"
                             "       ds41_remote_tcp --engine-side (--connect HOST:PORT | --listen PORT) [--name ds41_remote] [--f16]\n");
        return 2;
    }
#ifdef _WIN32
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
    timeBeginPeriod(1);   // 1 ms scheduler tick for this process's sleeps
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    if (const char* sp = std::getenv("DS41_TCP_SPIN_MS")) kSpinMs = std::atof(sp);
    return a.helper_side ? helper_side(a) : engine_side(a);
}
