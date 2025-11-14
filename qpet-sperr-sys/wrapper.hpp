#include <stdlib.h>

#include "SPERR_C_API.h"

void sperr_parse_header(
    const void *src,
    size_t *dimx,
    size_t *dimy,
    size_t *dimz,
    int *is_float)
{
    return C_API::sperr_parse_header(src, dimx, dimy, dimz, is_float);
}

int qpet_sperr_comp_3d(
    const void *src,
    int is_float,
    size_t dimx,
    size_t dimy,
    size_t dimz,
    size_t chunk_x,
    size_t chunk_y,
    size_t chunk_z,
    double data_pwe,
    size_t nthreads,
    void **dst,
    size_t *dst_len,
    const char* qoi,
    double qoi_pw,
    size_t qoi_bs_x,
    size_t qoi_bs_y,
    size_t qoi_bs_z,
    double qoi_k,
    bool high_prec)
{
    return C_API::qpet_sperr_comp_3d(src, is_float, dimx, dimy, dimz, chunk_x, chunk_y, chunk_z, data_pwe, nthreads, dst, dst_len, qoi, qoi_pw, qoi_bs_x, qoi_bs_y, qoi_bs_z, qoi_k, high_prec);
}

int sperr_decomp_3d(
    const void *src,
    size_t src_len,
    int output_float,
    size_t nthreads,
    size_t *dimx,
    size_t *dimy,
    size_t *dimz,
    void **dst)
{
    return C_API::sperr_decomp_3d(src, src_len, output_float, nthreads, dimx, dimy, dimz, dst);
}

void free_dst(void *dst)
{
    free(dst);
}
