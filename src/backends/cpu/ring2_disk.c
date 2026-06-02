/* ring2_disk.c — C2.1 Step 2b physical Optane-backed Ring-2 (see ring2_disk.h).
 * Win32 NO_BUFFERING|OVERLAPPED (hits the device, bypasses the page cache);
 * POSIX O_DIRECT + pread/pwrite fallback. v0 = synchronous blocking reads. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/ring2_disk.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <fcntl.h>
#  include <unistd.h>
#  include <time.h>
#endif

struct ring2_disk {
    size_t block;
    volatile long long n_reads;
    volatile long long read_ns;
#ifdef _WIN32
    HANDLE hK, hV;
    HANDLE wevent;          /* event for the single-writer spill path */
    void  *wbounce;         /* aligned bounce buffer for NO_BUFFERING writes */
    CRITICAL_SECTION wlock; /* serialize the shared bounce buffer */
    LARGE_INTEGER qpf;      /* QPC frequency */
#else
    int fdK, fdV;
#endif
};

struct ring2_scratch {
    ring2_disk *r;
    void *buf;
#ifdef _WIN32
    HANDLE ev;
#endif
};

size_t ring2_disk_block_bytes(const ring2_disk *r) { return r ? r->block : 0; }

void ring2_disk_stats(const ring2_disk *r, unsigned long long *n_reads, double *read_seconds) {
    if (!r) { if (n_reads) *n_reads = 0; if (read_seconds) *read_seconds = 0; return; }
    if (n_reads)     *n_reads = (unsigned long long)r->n_reads;
    if (read_seconds) *read_seconds = (double)r->read_ns * 1e-9;
}

#ifdef _WIN32
static HANDLE open_store(const char *dir, const char *name, size_t bytes) {
    char path[1024];
    snprintf(path, sizeof path, "%s%s", dir, name);
    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "    [ring2-disk] CreateFile FAIL %s (err %lu)\n", path, (unsigned long)GetLastError());
        return NULL;
    }
    LARGE_INTEGER li; li.QuadPart = (LONGLONG)bytes;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN) || !SetEndOfFile(h)) {
        fprintf(stderr, "    [ring2-disk] presize FAIL %s\n", path); CloseHandle(h); return NULL;
    }
    return h;
}

ring2_disk *ring2_disk_open(const char *dir, size_t bytes_per_file, size_t block_bytes) {
    if (!dir || block_bytes == 0) return NULL;
    ring2_disk *r = (ring2_disk *)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->block = block_bytes;
    QueryPerformanceFrequency(&r->qpf);
    InitializeCriticalSection(&r->wlock);
    r->wevent = CreateEvent(NULL, TRUE, FALSE, NULL);
    r->wbounce = _aligned_malloc(block_bytes, 4096);
    r->hK = open_store(dir, "sp_ring2_k.bin", bytes_per_file);
    r->hV = open_store(dir, "sp_ring2_v.bin", bytes_per_file);
    if (!r->wevent || !r->wbounce || !r->hK || !r->hV) { ring2_disk_close(r); return NULL; }
    fprintf(stderr, "    [ring2-disk] Optane store @ %s (NO_BUFFERING, %zu B/block, %.2f GB/file)\n",
            dir, block_bytes, (double)bytes_per_file / 1e9);
    return r;
}

int ring2_disk_write(ring2_disk *r, int which, size_t off, const void *src) {
    if (!r || !src) return 1;
    EnterCriticalSection(&r->wlock);
    memcpy(r->wbounce, src, r->block);
    OVERLAPPED ov; memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(off & 0xFFFFFFFFu); ov.OffsetHigh = (DWORD)(off >> 32); ov.hEvent = r->wevent;
    HANDLE h = which ? r->hV : r->hK;
    DWORD got = 0; BOOL ok = WriteFile(h, r->wbounce, (DWORD)r->block, NULL, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(h, &ov, &got, TRUE);
    else if (ok) got = (DWORD)r->block;
    LeaveCriticalSection(&r->wlock);
    return (got == (DWORD)r->block) ? 0 : 1;
}

const void *ring2_disk_read(ring2_disk *r, int which, size_t off, ring2_scratch *sc) {
    if (!r || !sc) return NULL;
    OVERLAPPED ov; memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(off & 0xFFFFFFFFu); ov.OffsetHigh = (DWORD)(off >> 32); ov.hEvent = sc->ev;
    HANDLE h = which ? r->hV : r->hK;
    LARGE_INTEGER t0, t1; QueryPerformanceCounter(&t0);
    DWORD got = 0; BOOL ok = ReadFile(h, sc->buf, (DWORD)r->block, NULL, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(h, &ov, &got, TRUE);
    else if (ok) got = (DWORD)r->block;
    QueryPerformanceCounter(&t1);
    InterlockedIncrement64(&r->n_reads);
    InterlockedExchangeAdd64(&r->read_ns,
        (LONGLONG)((double)(t1.QuadPart - t0.QuadPart) * 1e9 / (double)r->qpf.QuadPart));
    return (got == (DWORD)r->block) ? sc->buf : NULL;
}

ring2_scratch *ring2_disk_scratch_new(ring2_disk *r) {
    if (!r) return NULL;
    ring2_scratch *sc = (ring2_scratch *)calloc(1, sizeof *sc);
    if (!sc) return NULL;
    sc->r = r;
    sc->buf = _aligned_malloc(r->block, 4096);
    sc->ev = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!sc->buf || !sc->ev) { ring2_disk_scratch_free(sc); return NULL; }
    return sc;
}
void ring2_disk_scratch_free(ring2_scratch *sc) {
    if (!sc) return;
    if (sc->buf) _aligned_free(sc->buf);
    if (sc->ev)  CloseHandle(sc->ev);
    free(sc);
}
void ring2_disk_close(ring2_disk *r) {
    if (!r) return;
    if (r->hK) CloseHandle(r->hK);
    if (r->hV) CloseHandle(r->hV);
    if (r->wevent) CloseHandle(r->wevent);
    if (r->wbounce) _aligned_free(r->wbounce);
    DeleteCriticalSection(&r->wlock);
    free(r);
}

#else  /* POSIX: O_DIRECT + pread/pwrite (thread-safe, explicit offset) */
static int open_store(const char *dir, const char *name, size_t bytes) {
    char path[1024]; snprintf(path, sizeof path, "%s%s", dir, name);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC
#ifdef O_DIRECT
                  | O_DIRECT
#endif
                  , 0644);
    if (fd < 0) { fprintf(stderr, "    [ring2-disk] open FAIL %s\n", path); return -1; }
    if (ftruncate(fd, (off_t)bytes) != 0) { fprintf(stderr, "    [ring2-disk] ftruncate FAIL\n"); close(fd); return -1; }
    return fd;
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec*1e9 + (double)t.tv_nsec; }

ring2_disk *ring2_disk_open(const char *dir, size_t bytes_per_file, size_t block_bytes) {
    if (!dir || block_bytes == 0) return NULL;
    ring2_disk *r = (ring2_disk *)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->block = block_bytes;
    r->fdK = open_store(dir, "sp_ring2_k.bin", bytes_per_file);
    r->fdV = open_store(dir, "sp_ring2_v.bin", bytes_per_file);
    if (r->fdK < 0 || r->fdV < 0) { ring2_disk_close(r); return NULL; }
    fprintf(stderr, "    [ring2-disk] store @ %s (%zu B/block, %.2f GB/file)\n",
            dir, block_bytes, (double)bytes_per_file / 1e9);
    return r;
}
int ring2_disk_write(ring2_disk *r, int which, size_t off, const void *src) {
    if (!r || !src) return 1;
    int fd = which ? r->fdV : r->fdK;
    ssize_t w = pwrite(fd, src, r->block, (off_t)off);
    return (w == (ssize_t)r->block) ? 0 : 1;
}
const void *ring2_disk_read(ring2_disk *r, int which, size_t off, ring2_scratch *sc) {
    if (!r || !sc) return NULL;
    int fd = which ? r->fdV : r->fdK;
    double t0 = now_ns();
    ssize_t got = pread(fd, sc->buf, r->block, (off_t)off);
    double dt = now_ns() - t0;
    __sync_fetch_and_add(&r->n_reads, 1);
    __sync_fetch_and_add(&r->read_ns, (long long)dt);
    return (got == (ssize_t)r->block) ? sc->buf : NULL;
}
ring2_scratch *ring2_disk_scratch_new(ring2_disk *r) {
    if (!r) return NULL;
    ring2_scratch *sc = (ring2_scratch *)calloc(1, sizeof *sc);
    if (!sc) return NULL;
    sc->r = r;
    if (posix_memalign(&sc->buf, 4096, r->block) != 0) { free(sc); return NULL; }
    return sc;
}
void ring2_disk_scratch_free(ring2_scratch *sc) { if (!sc) return; free(sc->buf); free(sc); }
void ring2_disk_close(ring2_disk *r) {
    if (!r) return;
    if (r->fdK >= 0) close(r->fdK);
    if (r->fdV >= 0) close(r->fdV);
    free(r);
}
#endif
