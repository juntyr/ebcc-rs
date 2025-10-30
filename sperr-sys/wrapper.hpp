#include <stdlib.h>

#include "SPERR_C_API.h"

int sperr_comp_2d(
    const void *src,
    int is_float,
    size_t dimx,
    size_t dimy,
    int mode,
    double quality,
    int out_inc_header,
    void **dst,
    size_t *dst_len)
{
    return C_API::sperr_comp_2d(src, is_float, dimx, dimy, mode, quality, out_inc_header, dst, dst_len);
}

int sperr_decomp_2d(
    const void *src,
    size_t src_len,
    int output_float,
    size_t dimx,
    size_t dimy,
    void **dst)
{
    return C_API::sperr_decomp_2d(src, src_len, output_float, dimx, dimy, dst);
}

void sperr_parse_header(
    const void *src,
    size_t *dimx,
    size_t *dimy,
    size_t *dimz,
    int *is_float)
{
    return C_API::sperr_parse_header(src, dimx, dimy, dimz, is_float);
}

int sperr_comp_3d(
    const void *src,
    int is_float,
    size_t dimx,
    size_t dimy,
    size_t dimz,
    size_t chunk_x,
    size_t chunk_y,
    size_t chunk_z,
    int mode,
    double quality,
    size_t nthreads,
    void **dst,
    size_t *dst_len,
    const char* qoi,
    bool high_prec)
{
    return C_API::sperr_comp_3d(src, is_float, dimx, dimy, dimz, chunk_x, chunk_y, chunk_z, mode, quality, nthreads, dst, dst_len, qoi, high_prec);
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
