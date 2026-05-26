#include "sp_engine/avx512.h"
void sp_avx512_vnni_matvec(const int8_t *w, const uint8_t *a, const float *s,
                            const int32_t *b, int rows, int cols, float *out) {
    (void)w; (void)a; (void)s; (void)b; (void)rows; (void)cols; (void)out;
}
