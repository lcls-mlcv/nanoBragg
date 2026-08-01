/*
 * nanoBraggCUDA.cu -- CUDA kernel computing nanoBragg diffraction
 * images. The precision-sensitive chain -- the detector position, the diffracted and
 * scattering vectors, the Miller-index projection, the water background and the photon
 * scale -- is written against a precision-selectable working type Real; the correction
 * factors (airpath, solid angle, parallax, polarization) stay float. The same source
 * compiles at two precisions -- single and df64 -- each renaming its public entry
 * point so both objects link into one binary (see the NB_PRECISION selector below).
 *
 * Entry point: extern "C" nanoBraggSpotsCUDA(...), renamed per precision to
 * nanoBraggSpotsCUDA_single / nanoBraggSpotsCUDA_double, which marshals the host
 * (double) parameters to the device, launches nanoBraggSpotsCUDAKernel, and
 * reduces the result image.
 *
 * Three structural facts a reader needs:
 *   - The per-(phi, mosaic) rotated unit-cell vectors are precomputed on the
 *     host (in double, matching the nanoBraggCPU.c reference) and read from the
 *     phi_mos_* tables in the hot loop, so the kernel does no in-loop rotation.
 *   - The lattice shape transform uses a delta-reduced sincg: the fractional
 *     Miller index is reduced to |delta| <= 0.5 before the trig call.
 *   - Working-precision (float2) values are block-scoped: each is built and consumed
 *     inside its loop body, and only plain float/short representatives escape. At
 *     df64 the compensated pairs carry near-double accuracy on the single-precision
 *     pipe without the scarce double-precision units.
 */

/* Configuration and types */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>   /* usleep for the host poll cadence */
#include <driver_types.h>
#include "nanotypes.h"
#include "nanoBraggCUDA.h"

/* Precision selector. Real is the working type of the precision-sensitive chain: the
   detector position, the diffracted and scattering vectors, the Miller indices and the
   rotated cell vectors that feed them, the water background and the photon scale. At
   single precision Real is float and the chain is ordinary single-precision arithmetic;
   at df64 Real is a float2 carrying a value as an unevaluated (hi, lo) pair, so
   compensated arithmetic reaches near-double accuracy on the single-precision pipe
   without the scarce double-precision units. Overload resolution on Real selects every
   precision-sensitive operation whose parameters carry the working type. make_real and
   real_product take the same float (or float, float) input at both precisions and only
   their result differs, so overload resolution cannot pick between them; this selector
   aliases each to its precision-specific name instead, and both names are always defined
   below regardless of which one a given compile actually calls. Each compile also renames
   the public entry point so both precisions can link into one binary. */
#define NB_PREC_SINGLE 1
#define NB_PREC_DF64   2
#ifndef NB_PRECISION
#define NB_PRECISION NB_PREC_DF64
#endif
#if   NB_PRECISION == NB_PREC_SINGLE
typedef float  Real;
#define nanoBraggSpotsCUDA nanoBraggSpotsCUDA_single
#define make_real          make_real_float
#define real_product       real_product_float
#elif NB_PRECISION == NB_PREC_DF64
typedef float2 Real;
#define nanoBraggSpotsCUDA nanoBraggSpotsCUDA_double
#define make_real          make_real_double
#define real_product       real_product_double
#else
#error "NB_PRECISION must be NB_PREC_SINGLE or NB_PREC_DF64"
#endif

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define VECTOR_SIZE 4

struct structureFactorParams {
    short hkls;
    short h_min;
    short h_max;
    short h_range;
    short k_min;
    short k_max;
    short k_range;
    short l_min;
    short l_max;
    short l_range;
};

struct constParams {
    Real Avogadro;
    Real r_e_sqr;
    /* r_e_sqr * fluence, folded once on the host in double ahead of the per-precision
       split (see water_bg / photon_scale for how each precision uses it). */
    Real r_e_sqr_fluence;
};

struct beamSource {
    Real neg_unit_source_vector[VECTOR_SIZE];
    float intensity;
    Real lambda;
};

struct beamParams {
    float beam_vector[VECTOR_SIZE];
    Real fluence;
    bool calc_polar;
    float polarization;
    float polar_vector[VECTOR_SIZE];
    short sources;
};

struct detectorParams {
    short spixels;
    short fpixels;
    short roi_xmin;
    short roi_xmax;
    short roi_ymin;
    short roi_ymax;
    short oversample;
    bool point_pixel;
    float pixel_size;
    Real subpixel_size;
    long steps;
    float detector_thickstep;
    short detector_thicksteps;
    float detector_thick;
    float detector_mu;
    bool curved_detector;
    Real sdet_vector[VECTOR_SIZE];
    Real fdet_vector[VECTOR_SIZE];
    Real odet_vector[VECTOR_SIZE];
    Real pix0_vector[VECTOR_SIZE];
};

struct sampleParams {
    float distance;
    float close_distance;
    float water_size;
    Real water_F;
    Real water_MW;
};

struct unitCell {
    Real a0[VECTOR_SIZE];
    Real b0[VECTOR_SIZE];
    Real c0[VECTOR_SIZE];
    float V_cell;
};

struct crystalParams {
    unitCell uc;
    float Na;
    float Nb;
    float Nc;
    float default_F;
    float dmin;
    shapetype xtal_shape;
    int mosaic_domains;
    structureFactorParams fhklParams;
};

struct goniometerParams {
    float phi0;
    float phistep;
    int phisteps;
    float spindle_vector[VECTOR_SIZE];
};

/* Forward declarations */

static __global__ void nanoBraggSpotsCUDAKernel(const detectorParams * __restrict__ detectorPtr, const beamParams * __restrict__ beamPtr, const goniometerParams * __restrict__ goniometerPtr, const sampleParams * __restrict__ samplePtr,
        const crystalParams * crystalPtr, const constParams * __restrict__ constantsPtr, const beamSource * __restrict__ beam_sources, const float * __restrict__ Fhkl,
        const Real * __restrict__ phi_mos_a, const Real * __restrict__ phi_mos_b, const Real * __restrict__ phi_mos_c, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/, float * max_I_y_reduction /*out*/, bool * rangemap, unsigned int * progress,
        unsigned int * progress_pub, int progress_meter);

/* vector cross product where vector magnitude is 0th element */
__device__ static float *cross_product(const float * x, const float * y, float * z);
__device__ __inline__ float norm3d_fma_rn(float v1, float v2, float v3);
/* rotate a 3-vector about a unit vector axis */
__device__ static float *rotate_axis(const float * __restrict__ v, float *newv, const float * __restrict__ axis, const float phi);
/* compensated-pair rotate: the angle is carried as a (hi, lo) pair and its sin/cos are
   rebuilt in df arithmetic (see df_sincos), keeping both above single precision */
__device__ __inline__ static void rotate_axis(const float2 * __restrict__ v, float2 * newv, const float2 * __restrict__ axis, const float2 phi);
/* sin and cos of a compensated-pair angle, returned as compensated pairs, float-only */
__device__ __inline__ static void df_sincos(float2 phi, float2 * sinphi, float2 * cosphi);
__device__ __inline__ static long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range);
__device__ __inline__ float quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const float * __restrict__ Fhkl);
/* load the fully phi+mosaic-rotated cell (a/b/c[4], element[0]=0) from the host
   phi_mos_* table via __ldg; arguments are (table, out, index). */
__device__ __forceinline__ void load_rotated_cell_ldg(const Real * __restrict__ tbl, Real out[4], int idx_base);
/* delta-reduced sincg: N-slit interference function evaluated from the already-reduced
   delta = h - rint(h); pi is applied inside */
__device__ __inline__ static float sincg_delta(float delta, float N);
/* Fourier transform of a sphere */
__device__ static float sinc3(float x);
/* polarization factor from vectors */
__device__ static float polarization_factor(float kahn_factor, const float * __restrict__ unitIncident, float *unitDiffracted,
        const float * __restrict__ unitAxis);

/* compensated (hi, lo) primitives: error-free transforms (df_two_sum/df_two_prod/
   df_quick_two_sum) and the double-float add/sub/mul/sqrt/div composed from them. They
   back the df64 overloads of the geometry and scale operations below. */
__device__ __inline__ static float2 df_two_sum(float a, float b);
__device__ __inline__ static float2 df_two_prod(float a, float b);
__device__ __inline__ static float2 df_quick_two_sum(float a, float b);
__device__ __inline__ static float2 df_add(float2 a, float2 b);
__device__ __inline__ static float2 df_add_f(float2 a, float b);
__device__ __inline__ static float2 df_mul(float2 a, float2 b);
__device__ __inline__ static float2 df_mul_f(float2 a, float b);
__device__ __inline__ static float2 df_sub(float2 a, float2 b);
__device__ __inline__ static float2 df_sqrt(float2 a);
__device__ __inline__ static float2 df_div(float2 a, float2 b);

/* Precision-selectable geometry and scale operations, one float form (base kernel
   expressions) and one float2 form (compensated df64 expressions) each, adjacent in
   pairs. Overload resolution on the working type keeps the kernel body free of
   precision branches, except for make_real and real_product (see the selector above). */
/* vector inner product where vector magnitude is 0th element */
__device__ __inline__ static float dot_product(const float * x, const float * y);
__device__ __inline__ static float2 dot_product(const float2 * x, const float2 * y);
/* fractional: distance from the nearest Bragg peak, reduced to |delta| <= 0.5 */
__device__ __forceinline__ static float fractional(float h);
__device__ __forceinline__ static float fractional(float2 h);
/* nearest_hkl: integer Miller index of the nearest reciprocal-lattice point
   (the Fhkl structure-factor lookup index) */
__device__ __forceinline__ static short nearest_hkl(float h);
__device__ __forceinline__ static short nearest_hkl(float2 h);
/* widen a single-precision value into the working precision: identity for float,
   zero low word for the compensated pair */
__device__ __forceinline__ static float make_real_float(float v);
__device__ __forceinline__ static float2 make_real_double(float v);
/* the single-precision representative of a working-precision value: identity for
   float, high word of the compensated pair */
__device__ __forceinline__ static float real_to_float(float v);
__device__ __forceinline__ static float real_to_float(float2 v);
/* product of two single-precision values into the working precision (exact df pair) */
__device__ __inline__ static float real_product_float(float a, float b);
__device__ __inline__ static float2 real_product_double(float a, float b);
/* sub-pixel detector coordinate: subpix * count + subpix/2 */
__device__ __inline__ static float subpixel_coord(float subpix, float count);
__device__ __inline__ static float2 subpixel_coord(float2 subpix, float count);
/* one detector-basis position component: Fdet*fv + Sdet*sv + Odet*ov + pv */
__device__ __inline__ static float detector_position(float Fdet, float Sdet, float Odet, float fv, float sv, float ov, float pv);
__device__ __inline__ static float2 detector_position(float2 Fdet, float2 Sdet, float2 Odet, float2 fv, float2 sv, float2 ov, float2 pv);
/* working-precision diffracted ray: the single form reuses the already-computed float
   unit vector; the df64 form normalizes the df pixel position (df sqrt + df divide) */
__device__ __inline__ static void diffracted_ray(const float * diffracted_f, const float * pixel_pos, float * diffracted);
__device__ __inline__ static void diffracted_ray(const float * diffracted_f, const float2 * pixel_pos, float2 * diffracted);
/* working-precision scattering vector sc = (diffracted - source)/lambda; the third
   overload is the float pre-filter at the dmin site, narrowing the Real source and
   lambda to feed the cheap resolution cutoff before the working-precision vector below */
__device__ __inline__ static void scattering_vector(const float * diffracted, const float * neg_source, float lambda, float * sc);
__device__ __inline__ static void scattering_vector(const float2 * diffracted, const float2 * neg_source, float2 lambda, float2 * sc);
__device__ __inline__ static void scattering_vector(const float * diffracted_f, const float2 * neg_src, float2 lambda, float * scattering);
/* inverse of effective detector-thickness increase: dot product of odet_vector and
   the float representative of the diffracted ray */
__device__ __inline__ static float parallax(const float * __restrict__ odet, const float * diffracted_f);
__device__ __inline__ static float parallax(const float2 * __restrict__ odet, const float * diffracted_f);
/* curved-detector pixel rotation: rotate pixel_pos about sdet_vector then fdet_vector
   so it is always "distance" from the sample */
__device__ __inline__ static void curved_position(const float * sdet_vector, const float * fdet_vector, float distance,
        const float * dbvector, float * pixel_pos);
__device__ __inline__ static void curved_position(const float2 * sdet_vector, const float2 * fdet_vector, float distance,
        const float * dbvector, float2 * pixel_pos);
/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
__device__ static float unitize(float * vector, float *new_unit_vector);
__device__ static float unitize(float2 * vector, float *new_unit_vector);
/* amorphous water background intensity per sub-pixel (single precision result) */
__device__ __inline__ static float water_bg(float water_F, float r_e_sqr, float fluence, float r_e_sqr_fluence,
        float water_size, float Avogadro, float water_MW);
__device__ __inline__ static float water_bg(float2 water_F, float2 r_e_sqr, float2 fluence, float2 r_e_sqr_fluence,
        float water_size, float2 Avogadro, float2 water_MW);
/* photons-per-pixel scale: r_e_sqr * fluence * polar * I / steps (single precision result) */
__device__ __inline__ static float photon_scale(float r_e_sqr, float fluence, float r_e_sqr_fluence, float polar, float I, long steps);
__device__ __inline__ static float photon_scale(float2 r_e_sqr, float2 fluence, float2 r_e_sqr_fluence, float polar, float I, long steps);

/* Host code */

/* Check the return value of the CUDA runtime API call and exit
   the application if the call has failed. */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

static void doubleVectorToFloatVector(float * dest, double * src, size_t vector_items) {
    for (size_t i = 0; i < vector_items; i++) {
        dest[i] = src[i];
    }
}

/* narrow a double to a plain float: the correctly rounded conversion */
static inline float make_real_float(double v) {
    return (float) v;
}
/* split a double into a compensated (hi, lo) float pair: hi is the correctly
   rounded float, lo is the residual v - (double)hi the float drops, itself kept as a
   float. Reconstructing hi + lo recovers the double to near-full precision on a
   device that carries only float. */
static inline float2 make_real_double(double v) {
    float hi = (float) v;
    float lo = (float) (v - (double) hi);
    return make_float2(hi, lo);
}

/* Store a double vector into a working-precision vector, component by component,
   through make_real (the Real analogue of doubleVectorToFloatVector). */
static inline void make_real_vector(Real * dest, const double * src, size_t vector_items) {
    for (size_t i = 0; i < vector_items; i++) {
        dest[i] = make_real(src[i]);
    }
}

/* Reconstruct a double from a stored working-precision value for host-side diagnostics:
   the plain float cast at single precision, hi + lo at df64. */
static inline double real_to_double(float v) {
    return (double) v;
}
static inline double real_to_double(float2 v) {
    return (double) v.x + (double) v.y;
}

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
static double unitizeCPU(double * vector, double * new_unit_vector) {

    double v1 = vector[1];
    double v2 = vector[2];
    double v3 = vector[3];

    double mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

    if (mag != 0.0) {
        /* normalize it */
        new_unit_vector[0] = mag;
        new_unit_vector[1] = v1 / mag;
        new_unit_vector[2] = v2 / mag;
        new_unit_vector[3] = v3 / mag;
    } else {
        /* can't normalize, report zero vector */
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;
    }
    return mag;
}

extern "C" void nanoBraggSpotsCUDA(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
        double pixel_size, double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu,
        double sdet_vector[4], double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance,
        double beam_vector[4], double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources,
        double *source_X, double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4],
        shapetype xtal_shape, double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell,
        double water_size, double water_F, double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F,
        int interpolate, double *** Fhkl, int h_min, int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls,
        int nopolar, double polar_vector[4], double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/,
        double * omega_sum/*out*/, int * sumn /*out*/, double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/,
        double * max_I_y /*out*/, int progress_meter) {

    int total_pixels = spixels * fpixels;

    /* Enable host-mapped (zero-copy) allocations for the progress scalar. Must be the first CUDA
       runtime call here, before any context-creating alloc, or cudaHostAlloc(...Mapped) later fails
       with cudaErrorSetOnActiveProcess. */
    CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));

    bool * rangemap = (bool*) calloc(total_pixels, sizeof(bool));
    float * omega_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_x_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_y_reduction = (float*) calloc(total_pixels, sizeof(float));

    memset(floatimage, 0, sizeof(typeof(*floatimage)) * total_pixels);

    detectorParams detector;
    detector.spixels = spixels;
    detector.fpixels = fpixels;
    detector.roi_xmin = roi_xmin;
    detector.roi_xmax = roi_xmax;
    detector.roi_ymin = roi_ymin;
    detector.roi_ymax = roi_ymax;
    detector.oversample = oversample;
    detector.point_pixel = point_pixel;
    detector.pixel_size = pixel_size;
    detector.subpixel_size = make_real(subpixel_size);
    detector.steps = steps;
    detector.detector_thickstep = detector_thickstep;
    detector.detector_thicksteps = detector_thicksteps;
    detector.detector_thick = detector_thick;
    detector.detector_mu = detector_mu;
    detector.curved_detector = curved_detector;
    make_real_vector(detector.sdet_vector, sdet_vector, VECTOR_SIZE);
    make_real_vector(detector.fdet_vector, fdet_vector, VECTOR_SIZE);
    make_real_vector(detector.odet_vector, odet_vector, VECTOR_SIZE);
    make_real_vector(detector.pix0_vector, pix0_vector, VECTOR_SIZE);

    if (getenv("NB_DUMP_PIX0")) {
        fprintf(stderr, "NB_DUMP_PIX0 subpixel_size=%.17g pix0_vector=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) subpixel_size, (double) pix0_vector[0], (double) pix0_vector[1],
                (double) pix0_vector[2], (double) pix0_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 sdet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                sdet_vector[0], sdet_vector[1], sdet_vector[2], sdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 fdet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                fdet_vector[0], fdet_vector[1], fdet_vector[2], fdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 odet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                odet_vector[0], odet_vector[1], odet_vector[2], odet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 detector.sdet_vector(device real)=[%.17g, %.17g, %.17g, %.17g]\n",
                real_to_double(detector.sdet_vector[0]), real_to_double(detector.sdet_vector[1]), real_to_double(detector.sdet_vector[2]), real_to_double(detector.sdet_vector[3]));
        fprintf(stderr, "NB_DUMP_PIX0 detector.fdet_vector(device real)=[%.17g, %.17g, %.17g, %.17g]\n",
                real_to_double(detector.fdet_vector[0]), real_to_double(detector.fdet_vector[1]), real_to_double(detector.fdet_vector[2]), real_to_double(detector.fdet_vector[3]));
        fprintf(stderr, "NB_DUMP_PIX0 detector.odet_vector(device real)=[%.17g, %.17g, %.17g, %.17g]\n",
                real_to_double(detector.odet_vector[0]), real_to_double(detector.odet_vector[1]), real_to_double(detector.odet_vector[2]), real_to_double(detector.odet_vector[3]));
        fprintf(stderr, "NB_DUMP_PIX0 detector.pix0_vector(device real)=[%.17g, %.17g, %.17g, %.17g]\n",
                real_to_double(detector.pix0_vector[0]), real_to_double(detector.pix0_vector[1]), real_to_double(detector.pix0_vector[2]), real_to_double(detector.pix0_vector[3]));
    }

    beamParams beam;
    doubleVectorToFloatVector(beam.beam_vector, beam_vector, VECTOR_SIZE);
    beam.fluence = make_real(fluence);
    beam.calc_polar = !nopolar;
    beam.polarization = polarization;
    /* Unitize the polar vector once here on the host instead of once per pixel on the GPU. */
    double unit_polar_vector[VECTOR_SIZE];
    unitizeCPU(polar_vector, unit_polar_vector);
    doubleVectorToFloatVector(beam.polar_vector, unit_polar_vector, VECTOR_SIZE);
    beam.sources = sources;

    goniometerParams goniometer;
    goniometer.phi0 = phi0;
    goniometer.phistep = phistep;
    goniometer.phisteps = phisteps;
    doubleVectorToFloatVector(goniometer.spindle_vector, spindle_vector, VECTOR_SIZE);

    /* The unit-cell vectors fed to the h,k,l dot product are
         a/b/c = rotate_umat( rotate_axis(a0/b0/c0, spindle, phi), mosaic_umat[mos] )
       and depend only on (phi_tic, mos_tic); the spindle axis, base cell, phi steps and
       mosaic umats are all launch constants. Precompute the fully phi+mosaic-rotated cell
       once here on the host in double, using the same rotate_axis + rotate_umat algebra as
       the nanoBraggCPU.c reference, then cast each component to float. This keeps the
       rotation (and its sin/cos and matrix multiply) out of the per-subpixel/per-source hot
       loop, and matching the reference's double rotation improves parity. At phi==0 with no
       mosaic umat the rotation is an exact identity, so phi_mos_a[.] reproduces (float)a0[i]
       to the bit. Layout: phi_mos_a[(phi_tic*nmos + mos_tic)*3 + {0,1,2}] = rotated a's
       components {1,2,3}; same for b and c. */
    int nphi = phisteps > 0 ? phisteps : 1;
    int nmos = mosaic_domains > 0 ? mosaic_domains : 1;
    int pm_count = 3 * nphi * nmos;
    Real * phi_mos_a_host = new Real[pm_count];
    Real * phi_mos_b_host = new Real[pm_count];
    Real * phi_mos_c_host = new Real[pm_count];
    bool do_umat = (mosaic_spread > 0.0);
    for (int pt = 0; pt < nphi; pt++) {
        double phi_i = phistep * (double) pt + phi0;
        double sinphi = sin(phi_i);
        double cosphi = cos(phi_i);
        const double * srcv[3] = { a0, b0, c0 };
        double rot[3][4];   /* [vec][1..3] phi-rotated cell vector components, double */
        for (int _c = 0; _c < 3; ++_c) {
            const double * v = srcv[_c];
            /* rotate_axis algebra matching the nanoBraggCPU.c reference, in double */
            double dot = (spindle_vector[1] * v[1] + spindle_vector[2] * v[2] + spindle_vector[3] * v[3]) * (1.0 - cosphi);
            rot[_c][1] = spindle_vector[1] * dot + v[1] * cosphi + (-spindle_vector[3] * v[2] + spindle_vector[2] * v[3]) * sinphi;
            rot[_c][2] = spindle_vector[2] * dot + v[2] * cosphi + (+spindle_vector[3] * v[1] - spindle_vector[1] * v[3]) * sinphi;
            rot[_c][3] = spindle_vector[3] * dot + v[3] * cosphi + (-spindle_vector[2] * v[1] + spindle_vector[1] * v[2]) * sinphi;
        }
        Real * dstv[3] = { phi_mos_a_host, phi_mos_b_host, phi_mos_c_host };
        for (int mt = 0; mt < nmos; mt++) {
            const double * um = mosaic_umats + (long) mt * 9; /* row-major double[9] */
            int base = (pt * nmos + mt) * 3;
            for (int _c = 0; _c < 3; ++_c) {
                double newv[4];
                if (do_umat) {
                    /* rotate_umat algebra matching the nanoBraggCPU.c reference, in double */
                    newv[1] = um[0] * rot[_c][1] + um[1] * rot[_c][2] + um[2] * rot[_c][3];
                    newv[2] = um[3] * rot[_c][1] + um[4] * rot[_c][2] + um[5] * rot[_c][3];
                    newv[3] = um[6] * rot[_c][1] + um[7] * rot[_c][2] + um[8] * rot[_c][3];
                } else {
                    newv[1] = rot[_c][1]; newv[2] = rot[_c][2]; newv[3] = rot[_c][3];
                }
                dstv[_c][base + 0] = make_real(newv[1]);
                dstv[_c][base + 1] = make_real(newv[2]);
                dstv[_c][base + 2] = make_real(newv[3]);
            }
        }
    }
    Real * cu_phi_mos_a = NULL; Real * cu_phi_mos_b = NULL; Real * cu_phi_mos_c = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_a, sizeof(*cu_phi_mos_a) * pm_count));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_b, sizeof(*cu_phi_mos_b) * pm_count));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_c, sizeof(*cu_phi_mos_c) * pm_count));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_a, phi_mos_a_host, sizeof(*cu_phi_mos_a) * pm_count, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_b, phi_mos_b_host, sizeof(*cu_phi_mos_b) * pm_count, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_c, phi_mos_c_host, sizeof(*cu_phi_mos_c) * pm_count, cudaMemcpyHostToDevice));

    sampleParams sample;
    sample.distance = distance;
    sample.close_distance = close_distance;
    sample.water_F = make_real(water_F);
    sample.water_size = water_size;
    sample.water_MW = make_real(water_MW);

    crystalParams crystal;
    crystal.default_F = default_F;
    crystal.dmin = dmin;
    crystal.Na = Na;
    crystal.Nb = Nb;
    crystal.Nc = Nc;
    crystal.uc.V_cell = V_cell;
    make_real_vector(crystal.uc.a0, a0, VECTOR_SIZE);
    make_real_vector(crystal.uc.b0, b0, VECTOR_SIZE);
    make_real_vector(crystal.uc.c0, c0, VECTOR_SIZE);
    crystal.xtal_shape = xtal_shape;
    crystal.fhklParams.hkls = hkls;
    crystal.fhklParams.h_max = h_max;
    crystal.fhklParams.h_min = h_min;
    crystal.fhklParams.h_range = h_range;
    crystal.fhklParams.k_max = k_max;
    crystal.fhklParams.k_min = k_min;
    crystal.fhklParams.k_range = k_range;
    crystal.fhklParams.l_max = l_max;
    crystal.fhklParams.l_min = l_min;
    crystal.fhklParams.l_range = l_range;
    crystal.mosaic_domains = mosaic_domains;

    /* Pad hkl with default_F value; */
    int h_min_pad = h_min - 1, h_max_pad = h_max + 1, h_range_pad = h_range + 2;
    int k_min_pad = k_min - 1, k_max_pad = k_max + 1, k_range_pad = k_range + 2;
    int l_min_pad = l_min - 1, l_max_pad = l_max + 1, l_range_pad = l_range + 2;
    int hklsize_pad = h_range_pad * k_range_pad * l_range_pad;
    float * FhklLinearPad = (float*) calloc(hklsize_pad, sizeof(*FhklLinearPad));
    for (int h = 0; h < h_range_pad; h++) {
        for (int k = 0; k < k_range_pad; k++) {
            for (int l = 0; l < l_range_pad; l++) {
                /* convert Fhkl double to float */
                if (h > 0 && h < h_range_pad - 1 && k > 0 && k < k_range_pad - 1 && l > 0 && l < l_range_pad - 1) {
                    FhklLinearPad[h * k_range_pad * l_range_pad + k * l_range_pad + l] = Fhkl[h - 1][k - 1][l - 1];
                } else {
                    FhklLinearPad[h * k_range_pad * l_range_pad + k * l_range_pad + l] = crystal.default_F;
                }
            }
        }
    }
    crystal.fhklParams.hkls = hkls;
    crystal.fhklParams.h_max = h_max_pad;
    crystal.fhklParams.h_min = h_min_pad;
    crystal.fhklParams.h_range = h_range_pad;
    crystal.fhklParams.k_max = k_max_pad;
    crystal.fhklParams.k_min = k_min_pad;
    crystal.fhklParams.k_range = k_range_pad;
    crystal.fhklParams.l_max = l_max_pad;
    crystal.fhklParams.l_min = l_min_pad;
    crystal.fhklParams.l_range = l_range_pad;

    constParams constants;
    constants.Avogadro = make_real(Avogadro);
    constants.r_e_sqr = make_real(r_e_sqr);
    /* fold r_e_sqr * fluence once, in double, ahead of the per-precision split
       (see water_bg / photon_scale). */
    constants.r_e_sqr_fluence = make_real(r_e_sqr * fluence);

    detectorParams * cu_detector;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_detector, sizeof(*cu_detector)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_detector, &detector, sizeof(*cu_detector), cudaMemcpyHostToDevice));

    beamParams * cu_beam;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam, sizeof(*cu_beam)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_beam, &beam, sizeof(*cu_beam), cudaMemcpyHostToDevice));

    goniometerParams * cu_goniometer;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_goniometer, sizeof(*cu_goniometer)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_goniometer, &goniometer, sizeof(*cu_goniometer), cudaMemcpyHostToDevice));

    sampleParams * cu_sample;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_sample, sizeof(*cu_sample)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_sample, &sample, sizeof(*cu_sample), cudaMemcpyHostToDevice));

    crystalParams * cu_crystal;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_crystal, sizeof(*cu_crystal)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_crystal, &crystal, sizeof(*cu_crystal), cudaMemcpyHostToDevice));

    constParams * cu_constants;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_constants, sizeof(*cu_constants)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_constants, &constants, sizeof(*cu_constants), cudaMemcpyHostToDevice));

    /* Repackage beam sources. Unitize source vectors and pack them together contiguously. */
    beamSource * beam_sources = new beamSource[beam.sources];
    for (int i = 0; i < beam.sources; i++) {
        double unitSource[VECTOR_SIZE] = { 0.0, -source_X[i], -source_Y[i], -source_Z[i] };
        unitizeCPU(unitSource, unitSource);
        make_real_vector(beam_sources[i].neg_unit_source_vector, unitSource, VECTOR_SIZE);
        beam_sources[i].lambda = make_real(source_lambda[i]);
        beam_sources[i].intensity = source_I[i];
    }
    beamSource * cu_beam_sources = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam_sources, sizeof(*cu_beam_sources) * beam.sources));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_beam_sources, beam_sources, sizeof(*cu_beam_sources) * beam.sources, cudaMemcpyHostToDevice));

    float * cu_Fhkl = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_Fhkl, sizeof(*cu_Fhkl) * hklsize_pad));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_Fhkl, FhklLinearPad, sizeof(*cu_Fhkl) * hklsize_pad, cudaMemcpyHostToDevice));

    int unsigned short * cu_maskimage = NULL;
    if (maskimage != NULL) {
        CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_maskimage, sizeof(*cu_maskimage) * total_pixels));
        CUDA_CHECK_RETURN(cudaMemcpy(cu_maskimage, maskimage, sizeof(*cu_maskimage) * total_pixels, cudaMemcpyHostToDevice));
    }

    float * cu_floatimage = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_floatimage, sizeof(*cu_floatimage) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_floatimage, floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_omega_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_omega_reduction, omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_max_I_x_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_x_reduction, max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_max_I_y_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_y_reduction, max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyHostToDevice));

    bool * cu_rangemap = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_rangemap, sizeof(*cu_rangemap) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_rangemap, rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyHostToDevice));

    int deviceId = 0;
    CUDA_CHECK_RETURN(cudaGetDevice(&deviceId));
    int smCount = 0;
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks(smCount * 32, 1);

    unsigned int * cu_progress = NULL;                                      /* device-global fine counter */
    CUDA_CHECK_RETURN(cudaMalloc((void **)&cu_progress, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemset(cu_progress, 0, sizeof(unsigned int)));     /* per-launch reset */

    unsigned int * h_progress = NULL;                                       /* host-mapped pinned scalar */
    unsigned int * d_progress = NULL;                                       /* its device-side pointer */
    CUDA_CHECK_RETURN(cudaHostAlloc((void **)&h_progress, sizeof(unsigned int), cudaHostAllocMapped));
    *h_progress = 0;                                                        /* host-init: first read is a clean 0 */
    CUDA_CHECK_RETURN(cudaHostGetDevicePointer((void **)&d_progress, h_progress, 0));

    cudaStream_t kernelStream;
    CUDA_CHECK_RETURN(cudaStreamCreate(&kernelStream));                     /* blocking; no poll stream needed */

    /* Time the kernel launch and print "KERNEL_MS <ms>" to stderr as a stable,
       machine-parseable token (host-side timing only, does not affect the image). */
    cudaEvent_t nb_kern_start, nb_kern_stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&nb_kern_start));
    CUDA_CHECK_RETURN(cudaEventCreate(&nb_kern_stop));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());   /* MANDATORY: drain all H2D copies + the counter memset */
    CUDA_CHECK_RETURN(cudaEventRecord(nb_kern_start, kernelStream));

    nanoBraggSpotsCUDAKernel<<<numBlocks, threadsPerBlock, 0, kernelStream>>>(cu_detector, cu_beam, cu_goniometer, cu_sample, cu_crystal, cu_constants, cu_beam_sources, cu_Fhkl,
            cu_phi_mos_a, cu_phi_mos_b, cu_phi_mos_c, cu_maskimage, cu_floatimage /*out*/, cu_omega_reduction/*out*/, cu_max_I_x_reduction/*out*/, cu_max_I_y_reduction /*out*/, cu_rangemap /*out*/, cu_progress, d_progress, progress_meter);

    CUDA_CHECK_RETURN(cudaEventRecord(nb_kern_stop, kernelStream));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    if (!progress_meter) {
        /* -noprogress: baseline path -- the kernel issued no atomics/stores; just join, no prints
           (no 0%, no 1-99%, no 100%). */
        CUDA_CHECK_RETURN(cudaStreamSynchronize(kernelStream));
    } else {
        /* Parent-owned bookend: print 0% once, right before entering the poll loop. */
        printf("%lu%% done\n", 0UL);
        fflush(stdout);

        unsigned int  running_max = 0;   /* host-side monotone max over torn/laggy mapped reads */
        unsigned long last_pct    = 0;   /* highest integer percent already printed */
        /* cudaStreamQuery is used RAW here -- NEVER wrapped in CUDA_CHECK_RETURN, which would exit(1)
           on the normal, expected cudaErrorNotReady. It is NON-BLOCKING and touches no copy queue. */
        while (cudaStreamQuery(kernelStream) == cudaErrorNotReady) {
            /* ZERO-COPY read of the host-mapped scalar through its HOST pointer -- the kernel's
               __threadfence_system() store is directly visible. HARD RULE: no cudaMemcpy /
               cudaMemcpyAsync / cudaStreamSynchronize in this loop. On WSL2 there is no copy/execute
               overlap, so any blocking copy/sync here re-serializes the whole loop behind the kernel
               and the meter degrades to one poll at the end. */
            unsigned int v = *(volatile unsigned int *)h_progress;
            if (v > running_max) running_max = v;
            unsigned long pct = (total_pixels > 0)                  /* guard total_pixels == 0 */
                    ? (unsigned long)running_max * 100UL / (unsigned long)total_pixels
                    : 0UL;
            if (pct > 99UL) pct = 99UL;         /* the loop owns only 1-99%; 0% and 100% are the parent's */
            /* CPU-reference cadence (nanoBraggCPU meter): emit every crossed print-worthy percent,
               not just the newest, so a poll that jumps several percent stays faithful. Print-worthy
               = a multiple of 5, or inside the first 10% (p < 10) or last 10% (p > 90): 1% steps at
               the ends, 5% steps through the middle -> 0,1..9,10,15..90,91..99,100. */
            for (unsigned long p = last_pct + 1UL; p <= pct; ++p) {
                if (p % 5UL == 0UL || p < 10UL || p > 90UL) {
                    printf("%lu%% done\n", p);
                    fflush(stdout);                     /* stream live even when stdout is a pipe */
                }
            }
            last_pct = pct;
            usleep(50000);   /* ~50 ms cadence */
        }
        CUDA_CHECK_RETURN(cudaStreamSynchronize(kernelStream));   /* final join before any result output */
        printf("%lu%% done\n", 100UL);                            /* parent-owned bookend: meter ends at 100% */
        fflush(stdout);   /* emitted immediately after the join, before KERNEL_MS and all other output */
    }

    {
        float nb_kernel_ms = 0.0f;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&nb_kernel_ms, nb_kern_start, nb_kern_stop));
        fprintf(stderr, "KERNEL_MS %f\n", nb_kernel_ms);
    }

    CUDA_CHECK_RETURN(cudaEventDestroy(nb_kern_start));
    CUDA_CHECK_RETURN(cudaEventDestroy(nb_kern_stop));

    CUDA_CHECK_RETURN(cudaMemcpy(floatimage, cu_floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(omega_reduction, cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(max_I_x_reduction, cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(max_I_y_reduction, cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(rangemap, cu_rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(cu_detector));
    CUDA_CHECK_RETURN(cudaFree(cu_beam));
    CUDA_CHECK_RETURN(cudaFree(cu_goniometer));
    CUDA_CHECK_RETURN(cudaFree(cu_sample));
    CUDA_CHECK_RETURN(cudaFree(cu_crystal));
    CUDA_CHECK_RETURN(cudaFree(cu_constants));
    CUDA_CHECK_RETURN(cudaFree(cu_beam_sources));
    CUDA_CHECK_RETURN(cudaFree(cu_Fhkl));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_a));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_b));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_c));
    CUDA_CHECK_RETURN(cudaFree(cu_floatimage));
    CUDA_CHECK_RETURN(cudaFree(cu_omega_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_x_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_y_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_maskimage));
    CUDA_CHECK_RETURN(cudaFree(cu_rangemap));
    CUDA_CHECK_RETURN(cudaFree(cu_progress));           /* device-global counter */
    CUDA_CHECK_RETURN(cudaFreeHost(h_progress));        /* host-mapped pinned scalar (NOT cudaFree) */
    CUDA_CHECK_RETURN(cudaStreamDestroy(kernelStream));

    *max_I = 0;
    *max_I_x = 0;
    *max_I_y = 0;
    *sum = 0.0;
    *sumsqr = 0.0;
    *sumn = 0;
    *omega_sum = 0.0;

    for (int i = 0; i < total_pixels; i++) {
        if (!rangemap[i]) {
            continue;
        }
        float pixel = floatimage[i];
        if (pixel > (double) *max_I) {
            *max_I = pixel;
            *max_I_x = max_I_x_reduction[i];
            *max_I_y = max_I_y_reduction[i];
        }
        *sum += pixel;
        *sumsqr += pixel * pixel;
        ++(*sumn);
        *omega_sum += omega_reduction[i];
    }

    delete[] beam_sources;
    delete[] phi_mos_a_host;
    delete[] phi_mos_b_host;
    delete[] phi_mos_c_host;
    free(FhklLinearPad);
    free(rangemap);
    free(omega_reduction);
    free(max_I_x_reduction);
    free(max_I_y_reduction);
}

/* Device code */

/* crystal is const (the kernel never writes it) but deliberately NOT __restrict__: measured on
   sm_120, __restrict__ here costs +1 register for no speed change (it only re-routes ~14 loads
   onto the read-only cache). The register budget takes priority. */
static __global__ void nanoBraggSpotsCUDAKernel(const detectorParams * __restrict__ detector, const beamParams * __restrict__ beam, const goniometerParams * __restrict__ goniometer, const sampleParams * __restrict__ sample,
        const crystalParams * crystal, const constParams * __restrict__ constants, const beamSource * __restrict__ beam_sources, const float * __restrict__ Fhkl,
        const Real * __restrict__ phi_mos_a, const Real * __restrict__ phi_mos_b, const Real * __restrict__ phi_mos_c, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
        float * max_I_y_reduction /*out*/, bool * rangemap, unsigned int * progress,
        unsigned int * progress_pub, int progress_meter) {

    __shared__ float s_Na, s_Nb, s_Nc;

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        s_Na = crystal->Na;
        s_Nb = crystal->Nb;
        s_Nc = crystal->Nc;

    }
    __syncthreads();

    const long total_pixels = detector->spixels * detector->fpixels;
    const long fstride = gridDim.x * blockDim.x;
    const long sstride = gridDim.y * blockDim.y;
    const long stride = fstride * sstride;

    /* background intensity from amorphous water; see water_bg's overloads below for
       the per-precision expressions. */
    const float I_bg = water_bg(sample->water_F, constants->r_e_sqr, beam->fluence, constants->r_e_sqr_fluence, sample->water_size, constants->Avogadro, sample->water_MW);

    for (long pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
        /* progress: threadIdx.x == 0 of each row adds this wave's assigned pixels to the device-global
           fine counter (before the ROI skip, so it counts assigned pixels), then PUBLISHES the running
           total to the host-mapped scalar so the host can read it copy-free. __threadfence_system()
           pushes the mapped store out to host visibility; a plain store + this fence is sufficient
           (atomicAdd_system not required). progress_meter is warp-uniform (a scalar kernel arg), so
           with -noprogress neither the atomic nor the publish is issued -- the path is bit-for-bit
           baseline. The count stays correct for any block shape: the blockDim.y threads with
           threadIdx.x == 0 each add blockDim.x, summing to the block's per-wave pixel count. The
           atomicAdd and the mapped store + fence both run every wave so the published counter stays
           exact and current. */
        if (progress_meter && threadIdx.x == 0) {
            unsigned int published = atomicAdd(progress, blockDim.x) + blockDim.x;
            *(volatile unsigned int *)progress_pub = published;
            __threadfence_system();
        }
        const short fpixel = pixIdx % detector->fpixels;
        const short spixel = pixIdx / detector->fpixels;

        /* allow for just one part of detector to be rendered */
        if (fpixel < detector->roi_xmin || fpixel > detector->roi_xmax || spixel < detector->roi_ymin || spixel > detector->roi_ymax) { /* ROI region of interest */
            continue;
        }

        /* position in pixel array */
        const long j = pixIdx;

        /* allow for the use of a mask */
        if (maskimage != NULL) {
            /* skip any flagged pixels in the mask */
            if (maskimage[j] == 0) {
                continue;
            }
        }

        /* photon accumulator. Water background (I_bg) is not seeded here; it is added
           at the first sub-pixel below, scaled by that pixel's solid angle, so it
           tracks the CPU reference. Folding it into I keeps the kernel register-neutral. */
        float I = 0.0;
        float omega_sub_reduction = 0.0;
        float max_I_x_sub_reduction = 0.0;
        float max_I_y_sub_reduction = 0.0;
        float polar = 1.0;
        bool polar_computed = false;

        /* loop over sub-pixels */
        for (short subS = 0; subS < detector->oversample; ++subS) { /* Y voxel */
            for (short subF = 0; subF < detector->oversample; ++subF) { /* X voxel */
                /* absolute mm position of this sub-pixel on the detector (relative to its
                   origin). subpixel_size is the working-precision detector input; this is
                   the start of the pixel_pos -> diffracted -> scattering -> h,k,l -> sincg
                   geometry chain, carried at the working precision throughout. Only the
                   float representative escapes to the reduction outputs. */
                const Real subpix = detector->subpixel_size;
                const float subF_count = (float) (fpixel * detector->oversample + subF);
                const float subS_count = (float) (spixel * detector->oversample + subS);
                Real Fdet = subpixel_coord(subpix, subF_count); /* X voxel */
                Real Sdet = subpixel_coord(subpix, subS_count); /* Y voxel */

                max_I_x_sub_reduction = real_to_float(Fdet);
                max_I_y_sub_reduction = real_to_float(Sdet);

                for (short thick_tic = 0; thick_tic < detector->detector_thicksteps; ++thick_tic) {
                    /* assume "distance" is to the front of the detector sensor layer */
                    Real Odet = real_product((float) thick_tic, detector->detector_thickstep); /* Z Orthogonal voxel. */

                    /* construct detector subpixel position in 3D space: the pix0 origin
                       plus Fdet/Sdet/Odet along the detector basis vectors, at the working
                       precision. */
                    Real pixel_pos[4];
                    pixel_pos[1] = detector_position(Fdet, Sdet, Odet, detector->fdet_vector[1], detector->sdet_vector[1], detector->odet_vector[1], detector->pix0_vector[1]); /* X */
                    pixel_pos[2] = detector_position(Fdet, Sdet, Odet, detector->fdet_vector[2], detector->sdet_vector[2], detector->odet_vector[2], detector->pix0_vector[2]); /* Y */
                    pixel_pos[3] = detector_position(Fdet, Sdet, Odet, detector->fdet_vector[3], detector->sdet_vector[3], detector->odet_vector[3], detector->pix0_vector[3]); /* Z */
                    pixel_pos[0] = make_real(0.0f);

                    if (detector->curved_detector) {
                        /* construct detector pixel that is always "distance" from the sample. */
                        float dbvector[4];
                        dbvector[1] = sample->distance * beam->beam_vector[1];
                        dbvector[2] = sample->distance * beam->beam_vector[2];
                        dbvector[3] = sample->distance * beam->beam_vector[3];
                        curved_position(detector->sdet_vector, detector->fdet_vector, sample->distance, dbvector, pixel_pos);
                    }

                    /* Working-precision diffracted ray for the scattering chain, plus a float
                       representative (airpath + diffracted_f) that feeds the correction factors
                       below; see diffracted_ray's overloads for the per-precision derivation. */
                    float diffracted_f[4];
                    float airpath = unitize(pixel_pos, diffracted_f);
                    Real diffracted[4];
                    diffracted_ray(diffracted_f, pixel_pos, diffracted);

                    /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                    float omega_pixel = detector->pixel_size * detector->pixel_size / airpath / airpath * sample->close_distance / airpath;
                    /* option to turn off obliquity effect, inverse-square-law only.
                       1.0f (not 1.0): a bare 1.0 is a double literal in C++, and
                       "1.0 / airpath" would silently promote airpath to double for
                       the division -- exactly the double promotion this file must not
                       have. */
                    if (detector->point_pixel) {
                        omega_pixel = 1.0f / airpath / airpath;
                    }

                    /* now calculate detector thickness effects */
                    float capture_fraction = 1.0;
                    if (detector->detector_thick > 0.0f) {
                        /* inverse of effective thickness increase: dot product of odet_vector
                           and the float representative of the diffracted ray, via parallax().
                           odet_vector is read via __ldg as a caching hint inside parallax(); it
                           lives in a read-only const-restrict struct, so this changes no results
                           but lets ptxas schedule this cold branch's loads independently of the
                           hot pixel_pos/geometry loads, relieving register pressure. */
                        float parallax_dot = parallax(detector->odet_vector, diffracted_f);
                        capture_fraction = exp(-thick_tic * detector->detector_thickstep * detector->detector_mu / parallax_dot)
                                - exp(-(thick_tic + 1) * detector->detector_thickstep * detector->detector_mu / parallax_dot);
                    }

                    /* Add the water background once, scaled by this sub-pixel's solid
                       angle and capture fraction, so it tracks the CPU reference (which
                       scales the whole pixel intensity by omega, water included). A no-op
                       when there is no water (I_bg == 0). */
                    if (subS == 0 && subF == 0 && thick_tic == 0) {
                        I += capture_fraction * omega_pixel * I_bg;
                    }

                    /* loop over sources now */
                    for (short source = 0; source < beam->sources; ++source) {

                        /* read this source's wavelength and unit vector via __ldg: beam_sources
                           is constant for the whole launch and every thread reads the same
                           entries, so routing the loads through the read-only data cache serves
                           them from cache instead of issuing regular global loads. Both are
                           carried at the working precision; float representatives feed the
                           resolution heuristic and the polarization incident vector. */
                        Real lambda = __ldg(&beam_sources[source].lambda);
                        Real neg_src[VECTOR_SIZE];
                        neg_src[1] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                        neg_src[2] = __ldg(&beam_sources[source].neg_unit_source_vector[2]);
                        neg_src[3] = __ldg(&beam_sources[source].neg_unit_source_vector[3]);

                        /* float scattering vector from the float representative of the diffracted
                           ray, for the dmin/stol resolution cutoff only. */
                        float scattering[4];
                        scattering_vector(diffracted_f, neg_src, lambda, scattering);

                        /* sin(theta)/lambda is half the scattering vector length */
                        float stol = (float)0.5 * norm3d_fma_rn(scattering[1], scattering[2], scattering[3]);

                        /* rough cut to speed things up when we aren't using whole detector */
                        if (crystal->dmin > 0.0f && stol > 0.0f) {
                            if (crystal->dmin > 0.5f / stol) {
                                continue;
                            }
                        }

                        /* working-precision scattering vector (diffracted - source)/lambda,
                           feeding the h,k,l projection; block-scoped, consumed below. */
                        Real sc[VECTOR_SIZE];
                        scattering_vector(diffracted, neg_src, lambda, sc);

                        /* Compute the polarization factor once per pixel (from the first
                           source) and reuse it, matching the CPU reference; the boolean flag
                           keeps the guard uniform across the warp (no divergence). */
                        if (beam->calc_polar && !polar_computed) {
                            /* need to compute polarization factor */
                            float incident[4];
                            incident[1] = real_to_float(neg_src[1]);
                            incident[2] = real_to_float(neg_src[2]);
                            incident[3] = real_to_float(neg_src[3]);
                            polar = polarization_factor(beam->polarization, incident, diffracted_f, beam->polar_vector);
                            polar_computed = true;
                        }

                        /* sweep over phi angles */
                        for (int phi_tic = 0; phi_tic < goniometer->phisteps; ++phi_tic) {

                            /* enumerate mosaic domains */
                            for (int mos_tic = 0; mos_tic < crystal->mosaic_domains; ++mos_tic) {

                                /* Outputs of the lattice-shape block below, all single precision:
                                   the Miller indices h,k,l (the round crystal shapes need their
                                   radial distance from the nearest reflection), the nearest
                                   reflection h0,k0,l0 (the structure-factor lookup index), and the
                                   reduced fractional offset dh,dk,dl (the square-crystal shape
                                   transform). Only these plain values cross out; no working-precision
                                   value leaves the block. */
                                float h, k, l;
                                short h0, k0, l0;
                                float dh, dk, dl;
                                {
                                    /* The fully phi+mosaic-rotated cell a/b/c was precomputed on the
                                       host (see the phi_mos_* precompute above) and is looked up here,
                                       so there is no in-kernel rotate_axis or rotate_umat, no sin/cos
                                       and no umat matrix multiply in the hot loop. phi_mos_{a,b,c}
                                       [(phi_tic*mosaic_domains + mos_tic)*3 + {0,1,2}] carry the rotated
                                       a/b/c components {1,2,3}. At phi==0 with no mosaic umat the
                                       rotation is an exact identity, so these equal the base cell
                                       a0/b0/c0 to the bit. */
                                    const int pm = (phi_tic * crystal->mosaic_domains + mos_tic) * 3;
                                    Real a[VECTOR_SIZE]; load_rotated_cell_ldg(phi_mos_a, a, pm);
                                    Real b[VECTOR_SIZE]; load_rotated_cell_ldg(phi_mos_b, b, pm);
                                    Real c[VECTOR_SIZE]; load_rotated_cell_ldg(phi_mos_c, c, pm);

                                    /* construct Miller indices: project the rotated cell onto the
                                       working-precision scattering vector sc built above */
                                    Real hh = dot_product(a, sc);
                                    Real kk = dot_product(b, sc);
                                    Real ll = dot_product(c, sc);

                                    /* single-precision Miller index (the round-shape radial distance) */
                                    h = real_to_float(hh);
                                    k = real_to_float(kk);
                                    l = real_to_float(ll);

                                    /* integer Miller indices of the nearest reflection */
                                    h0 = nearest_hkl(hh);
                                    k0 = nearest_hkl(kk);
                                    l0 = nearest_hkl(ll);

                                    /* reduce to fractional Miller indices */
                                    dh = fractional(hh);
                                    dk = fractional(kk);
                                    dl = fractional(ll);
                                }

                                /* structure factor of the lattice (parallelepiped crystal)
                                 F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
                                 */
                                float F_latt = 1.0; /* Shape transform for the crystal. */
                                if (crystal->xtal_shape == SQUARE) {
                                    /* xtal is a parallelepiped */
                                    /* delta-reduced sincg: reduce h,k,l with fractional above BEFORE the trig,
                                       then evaluate sinf(PIf*N*delta)/sinf(PIf*delta) */
                                    if (crystal->Na > 1) F_latt *= sincg_delta(dh, (float) crystal->Na);
                                    if (crystal->Nb > 1) F_latt *= sincg_delta(dk, (float) crystal->Nb);
                                    if (crystal->Nc > 1) F_latt *= sincg_delta(dl, (float) crystal->Nc);
                                } else if (crystal->xtal_shape == ROUND) {
                                    /* use sinc3 for elliptical xtal shape,
                                     correcting for sqrt of volume ratio between cube and sphere */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * 0.723601254558268f * sinc3((float) M_PI * sqrt(hrad_sqr * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == GAUSS) {
                                    /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * exp(-(hrad_sqr / 0.63f * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == TOPHAT) {
                                    /* make a flat-top spot of same height and volume as square_xtal spots */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * (hrad_sqr * 1.0f /*fudge*/ < 0.3969f);
                                }
                                /* structure factor of the unit cell */
                                float F_cell =
                                        crystal->fhklParams.hkls ?
                                                 quickFcell_ldg(crystal->fhklParams.hkls,
                                                    h0,
                                                    crystal->fhklParams.h_max,
                                                    crystal->fhklParams.h_min,
                                                    k0,
                                                    crystal->fhklParams.k_max,
                                                    crystal->fhklParams.k_min,
                                                    l0,
                                                    crystal->fhklParams.l_max,
                                                    crystal->fhklParams.l_min,
                                                    crystal->fhklParams.h_range,
                                                    crystal->fhklParams.k_range,
                                                    crystal->fhklParams.l_range, Fhkl) :
                                                 crystal->default_F;

                                /* now we have the structure factor for this pixel */

                                /* convert amplitudes into intensity (photons per steradian) */
                                I += F_cell * F_cell * F_latt * F_latt * capture_fraction * omega_pixel;
                                omega_sub_reduction += omega_pixel;
                            }
                            /* end of mosaic loop */
                        }
                        /* end of phi loop */
                    }
                    /* end of source loop */
                }
                /* end of detector thickness loop */
            }
            /* end of sub-pixel y loop */
        }
        /* end of sub-pixel x loop */
        /* I holds the Bragg sum plus the solid-angle-scaled water background; apply
           polarization and normalize by the sub-step count. The df64 form carries the
           folded r_e_sqr*fluence scale through in compensated pairs, popping to a single
           float here; the single form is the base scale expression. */
        const float photons = photon_scale(constants->r_e_sqr, beam->fluence, constants->r_e_sqr_fluence, polar, I, detector->steps);
        floatimage[j] = photons;
        omega_reduction[j] = omega_sub_reduction;
        max_I_x_reduction[j] = max_I_x_sub_reduction;
        max_I_y_reduction[j] = max_I_y_sub_reduction;
        rangemap[j] = true;
    }
}

/* vector cross product where vector magnitude is 0th element */
__device__ static float *cross_product(const float * x, const float * y, float * z) {
    z[1] = x[2] * y[3] - x[3] * y[2];
    z[2] = x[3] * y[1] - x[1] * y[3];
    z[3] = x[1] * y[2] - x[2] * y[1];
    z[0] = 0.0;

    return z;
}

__device__ __inline__ float norm3d_fma_rn(float v1, float v2, float v3) {
    float q_sqr = v3 * v3;
    q_sqr = __fmaf_rn(v2, v2, q_sqr);
    q_sqr = __fmaf_rn(v1, v1, q_sqr);
    return sqrt(q_sqr);
}

/* rotate a 3-vector about a unit vector axis */
__device__ static float *rotate_axis(const float * __restrict__ v, float * newv, const float * __restrict__ axis, const float phi) {

    const float sinphi = sin(phi);
    const float cosphi = cos(phi);
    const float a1 = axis[1];
    const float a2 = axis[2];
    const float a3 = axis[3];
    const float v1 = v[1];
    const float v2 = v[2];
    const float v3 = v[3];
    const float dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0f - cosphi);

    newv[1] = __fmaf_rn(a1, dot, v1 * cosphi) + __fmaf_rn(-a3, v2, a2 * v3) * sinphi;
    newv[2] = __fmaf_rn(a2, dot, v2 * cosphi) + __fmaf_rn(+a3, v1, -a1 * v3) * sinphi;
    newv[3] = __fmaf_rn(a3, dot, v3 * cosphi) + __fmaf_rn(-a2, v1, a1 * v2) * sinphi;

    return newv;
}

__device__ __inline__ static long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range) {
    return x * y_range * z_range + y * z_range + z;
}

__device__ __inline__ float quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const float * __restrict__ Fhkl) {
    short h = min(max(h0 - h_min, 0), h_range - 1);
    short k = min(max(k0 - k_min, 0), k_range - 1);
    short l = min(max(l0 - l_min, 0), l_range - 1);
    return __ldg(&Fhkl[flatten3dindex(h, k, l, h_range, k_range, l_range)]);
}

/* Copy one rotated cell vector (three components) from the host-precomputed table into
   out[]. Vectors in this codebase use a 4-element convention: element 0 holds the
   magnitude and elements 1..3 hold the x,y,z components. The magnitude is not needed
   here (the dot product reads only elements 1..3), so out[0] is left untouched and
   out[1], out[2], out[3] receive the three components, read through the __ldg read-only
   cache. idx_base selects the table entry: (phi_tic * mosaic_domains + mos_tic) * 3. */
__device__ __forceinline__ void load_rotated_cell_ldg(const Real * __restrict__ tbl, Real out[4], int idx_base) {
    out[1] = __ldg(&tbl[idx_base + 0]);
    out[2] = __ldg(&tbl[idx_base + 1]);
    out[3] = __ldg(&tbl[idx_base + 2]);
}

/* Delta-reduced sincg: evaluates the N-slit interference function
   sin(pi*N*delta)/sin(pi*delta) from an already-reduced argument
   delta = h - rint(h), so |delta| <= 0.5. Reducing before the trig calls keeps
   the argument small and well-conditioned. F_latt is squared into intensity
   downstream, so the sign difference vs evaluating at the unreduced argument
   is irrelevant. */
__device__ __inline__ static float sincg_delta(float delta, float N) {
    /* removable singularity: as delta->0, sin(pi*N*delta)/sin(pi*delta) -> N.
       Guard delta==0 (Bragg peak / integer h) to avoid 0/0 = NaN. */
    const float PIf = 3.14159265358979323846f;
    if (delta == 0.0f)
        return N;
    return sinf(PIf * N * delta) / sinf(PIf * delta);
}

/* Fourier transform of a sphere */
__device__ static float sinc3(float x) {
    if (x != 0.0f)
        return 3.0f * (sin(x) / x - cos(x)) / (x * x);

    return 1.0;

}

/* polarization factor */
__device__ static float polarization_factor(float kahn_factor, const float * __restrict__ unitIncident, float *unitDiffracted,
        const float * __restrict__ unitAxis) {
    float cos2theta, cos2theta_sqr, sin2theta_sqr;
    float psi = 0.0f;
    float E_in[4], B_in[4], E_out[4], B_out[4];

    /* component of diffracted unit vector along incident beam unit vector */
    cos2theta = dot_product(unitIncident, unitDiffracted);
    cos2theta_sqr = cos2theta * cos2theta;
    sin2theta_sqr = 1.0f - cos2theta_sqr;

    if (kahn_factor != 0.0f) {
        /* tricky bit here is deciding which direction the E-vector lies in for each source
         here we assume it is closest to the "axis" defined above */

        /* cross product to get "vertical" axis that is orthogonal to the canonical "polarization" */
        cross_product(unitAxis, unitIncident, B_in);
        /* make it a unit vector */
        unitize(B_in, B_in);

        /* cross product with incident beam to get E-vector direction */
        cross_product(unitIncident, B_in, E_in);
        /* make it a unit vector */
        unitize(E_in, E_in);

        /* get components of diffracted ray projected onto the E-B plane */
        E_out[0] = dot_product(unitDiffracted, E_in);
        B_out[0] = dot_product(unitDiffracted, B_in);

        /* compute the angle of the diffracted ray projected onto the incident E-B plane */
        psi = -atan2(B_out[0], E_out[0]);
    }

    /* correction for polarized incident beam */
    return 0.5f * (1.0f + cos2theta_sqr - kahn_factor * cos(2.0f * psi) * sin2theta_sqr);
}

/* --- df64 arithmetic and precision-selectable helpers ---------------------------
   The compensated (hi, lo) primitives -- the error-free transforms df_two_sum /
   df_two_prod / df_quick_two_sum and the double-float add/sub/mul/sqrt/div composed
   from them -- and the Real overloads they back. These are selected only when Real is
   float2; at single precision the float forms above are chosen and none of this is
   reached. */

/* error-free transform: the returned pair sums to a + b exactly */
__device__ __inline__ static float2 df_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return make_float2(s, err);
}

/* error-free transform: the returned pair equals a * b exactly, one fused
   multiply-add capturing the rounding remainder in the low word */
__device__ __inline__ static float2 df_two_prod(float a, float b) {
    float p = a * b;
    float e = __fmaf_rn(a, b, -p);
    return make_float2(p, e);
}

/* renormalizing sum valid when |a| >= |b| */
__device__ __inline__ static float2 df_quick_two_sum(float a, float b) {
    float s = a + b;
    float err = b - (s - a);
    return make_float2(s, err);
}

/* (hi, lo) + (hi, lo) */
__device__ __inline__ static float2 df_add(float2 a, float2 b) {
    float2 s = df_two_sum(a.x, b.x);
    float2 t = df_two_sum(a.y, b.y);
    s.y += t.x;
    float2 r = df_quick_two_sum(s.x, s.y);
    r.y += t.y;
    r = df_quick_two_sum(r.x, r.y);
    return r;
}

/* (hi, lo) * (hi, lo) */
__device__ __inline__ static float2 df_mul(float2 a, float2 b) {
    float2 p = df_two_prod(a.x, b.x);
    float e = __fmaf_rn(a.x, b.y, __fmaf_rn(a.y, b.x, p.y));
    return df_quick_two_sum(p.x, e);
}

/* (hi, lo) + float */
__device__ __inline__ static float2 df_add_f(float2 a, float b) {
    float2 s = df_two_sum(a.x, b);
    s.y += a.y;
    return df_quick_two_sum(s.x, s.y);
}

/* (hi, lo) * float */
__device__ __inline__ static float2 df_mul_f(float2 a, float b) {
    float2 p = df_two_prod(a.x, b);
    float e = __fmaf_rn(a.y, b, p.y);
    return df_quick_two_sum(p.x, e);
}

/* (hi, lo) - (hi, lo) */
__device__ __inline__ static float2 df_sub(float2 a, float2 b) {
    return df_add(a, make_float2(-b.x, -b.y));
}

/* (hi, lo) sqrt: single Newton correction from a float sqrt seed */
__device__ __inline__ static float2 df_sqrt(float2 a) {
    if (a.x == 0.0f) return make_float2(0.0f, 0.0f);
    const float xn = rsqrtf(a.x);            /* approx 1/sqrt(a) */
    const float yn = a.x * xn;               /* approx sqrt(a) */
    const float2 yn_sqr = df_two_prod(yn, yn);
    const float2 resid = df_sub(a, yn_sqr);  /* a - yn^2, as df */
    return df_add_f(make_float2(yn, 0.0f), resid.x * (xn * 0.5f));
}

/* (hi, lo) / (hi, lo): one Newton correction from a float quotient seed */
__device__ __inline__ static float2 df_div(float2 a, float2 b) {
    const float xn = a.x / b.x;              /* approx quotient */
    const float2 prod = df_mul_f(b, xn);     /* b*xn as df */
    const float2 resid = df_sub(a, prod);    /* a - b*xn */
    const float corr = resid.x / b.x;
    return df_add_f(make_float2(xn, 0.0f), corr);
}

/* vector inner product where vector magnitude is 0th element */
__device__ __inline__ static float dot_product(const float * x, const float * y) {
    return x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

/* Compensated dot product of two 3-vectors whose components are (hi, lo) float
   pairs -- forms one Miller index (x . y) to near-double accuracy using only
   float arithmetic. 4-element convention, element 0 unused. */
__device__ __inline__ static float2 dot_product(const float2 * x, const float2 * y) {
    /* The obvious way to write this dot product -- read it to see WHAT the body below
       computes; the body computes the identical result, just faster:

           float2 sum = make_float2(0.0f, 0.0f);
           sum = df_add(sum, df_mul(x[1], y[1]));   // multiply a pair, add it to the total
           sum = df_add(sum, df_mul(x[2], y[2]));
           sum = df_add(sum, df_mul(x[3], y[3]));
           return sum;                              // ~81 float ops over the 3 terms

       df_mul and df_add each finish by renormalizing their (hi, lo) result -- cleaning it
       into a tidy non-overlapping pair for the next op. Between terms that cleanup is wasted
       work: the very next add disturbs the low word again. The body below skips it -- it keeps
       the running total's high word in sum.x, lets every leftover bit pile up untidied in
       sum.y, and renormalizes ONCE at the end (df_quick_two_sum). Same answer in ~39 float ops
       instead of ~81, a bit under half the work. Per term: p is the exact product as
       (high, error); lo folds in the two cross terms hi*lo + lo*hi that the pairs carry (one
       FMA); s two-sums the high word into the running total, and its remainder plus lo drop
       into sum.y. Hand-unrolled over the three components to match the rest of the file. */
    float2 sum = make_float2(0.0f, 0.0f);

    float2 p1  = df_two_prod(x[1].x, y[1].x);
    float  lo1 = __fmaf_rn(x[1].x, y[1].y, __fmaf_rn(x[1].y, y[1].x, p1.y));
    float2 s1  = df_two_sum(sum.x, p1.x);
    sum.x = s1.x;
    sum.y += s1.y + lo1;

    float2 p2  = df_two_prod(x[2].x, y[2].x);
    float  lo2 = __fmaf_rn(x[2].x, y[2].y, __fmaf_rn(x[2].y, y[2].x, p2.y));
    float2 s2  = df_two_sum(sum.x, p2.x);
    sum.x = s2.x;
    sum.y += s2.y + lo2;

    float2 p3  = df_two_prod(x[3].x, y[3].x);
    float  lo3 = __fmaf_rn(x[3].x, y[3].y, __fmaf_rn(x[3].y, y[3].x, p3.y));
    float2 s3  = df_two_sum(sum.x, p3.x);
    sum.x = s3.x;
    sum.y += s3.y + lo3;

    return df_quick_two_sum(sum.x, sum.y);
}

/* fractional: distance from the nearest Bragg peak, reduced to |delta| <= 0.5 */
__device__ __forceinline__ static float fractional(float h) {
    return h - rintf(h);
}

/* fractional offset of a carried value; the low word folds into the reduced offset,
   which is small enough to hold in single precision */
__device__ __forceinline__ static float fractional(float2 h) {
    return (h.x - rintf(h.x)) + h.y;
}

/* nearest_hkl: integer Miller index of the nearest reciprocal-lattice point
   (the Fhkl structure-factor lookup index) */
__device__ __forceinline__ static short nearest_hkl(float h) {
    return (short) ceilf(h - 0.5f);
}

/* nearest reciprocal-lattice point of a carried value; the nearest point is fixed by
   the high word */
__device__ __forceinline__ static short nearest_hkl(float2 h) {
    return (short) ceilf(h.x - 0.5f);
}

/* widen a single-precision value into the working precision: identity for float,
   zero low word for the compensated pair. NB_PRECISION aliases make_real to whichever
   of these two names actually gets called, since both take this same float input and
   differ only in their return type (see the selector above). */
__device__ __forceinline__ static float make_real_float(float v) {
    return v;
}
__device__ __forceinline__ static float2 make_real_double(float v) {
    return make_float2(v, 0.0f);
}

/* the single-precision representative of a working-precision value: identity for
   float, high word of the compensated pair */
__device__ __forceinline__ static float real_to_float(float v) {
    return v;
}
__device__ __forceinline__ static float real_to_float(float2 v) {
    return v.x;
}

/* product of two single-precision values into the working precision (exact df pair).
   NB_PRECISION aliases real_product the same way it aliases make_real, above, since
   both forms share these float inputs and differ only in their return type. */
__device__ __inline__ static float real_product_float(float a, float b) {
    return a * b;
}
__device__ __inline__ static float2 real_product_double(float a, float b) {
    return df_two_prod(a, b);
}

/* sub-pixel detector coordinate: subpix * count + subpix/2, base association at single
   precision, compensated pairs at df64. */
__device__ __inline__ static float subpixel_coord(float subpix, float count) {
    return subpix * count + subpix / 2.0f;
}
__device__ __inline__ static float2 subpixel_coord(float2 subpix, float count) {
    return df_add(df_mul_f(subpix, count), df_mul_f(subpix, 0.5f));
}

/* one detector-basis position component: Fdet*fv + Sdet*sv + Odet*ov + pv. The single
   form keeps the base left-to-right association; the df64 form pairs the products as the
   compensated chain does. */
__device__ __inline__ static float detector_position(float Fdet, float Sdet, float Odet, float fv, float sv, float ov, float pv) {
    return Fdet * fv + Sdet * sv + Odet * ov + pv;
}
__device__ __inline__ static float2 detector_position(float2 Fdet, float2 Sdet, float2 Odet, float2 fv, float2 sv, float2 ov, float2 pv) {
    return df_add(df_add(df_mul(Fdet, fv), df_mul(Sdet, sv)), df_add(df_mul(Odet, ov), pv));
}

/* working-precision diffracted ray. The single form reuses the already-computed float
   unit vector (the correction block's unitize output). The df64 form normalizes the df
   pixel position directly (df sqrt + df divide) so the compensated position carries into
   the scattering vector without truncation. */
__device__ __inline__ static void diffracted_ray(const float * diffracted_f, const float * pixel_pos, float * diffracted) {
    diffracted[1] = diffracted_f[1];
    diffracted[2] = diffracted_f[2];
    diffracted[3] = diffracted_f[3];
}
__device__ __inline__ static void diffracted_ray(const float * diffracted_f, const float2 * pixel_pos, float2 * diffracted) {
    const float2 px1 = pixel_pos[1], px2 = pixel_pos[2], px3 = pixel_pos[3];
    const float2 magd_sqr = df_add(df_add(df_mul(px1, px1), df_mul(px2, px2)), df_mul(px3, px3));
    const float2 magd = df_sqrt(magd_sqr);
    if (magd.x != 0.0f) {
        diffracted[0] = magd;
        diffracted[1] = df_div(px1, magd);
        diffracted[2] = df_div(px2, magd);
        diffracted[3] = df_div(px3, magd);
    } else {
        diffracted[0] = diffracted[1] = diffracted[2] = diffracted[3] = make_float2(0.0f, 0.0f);
    }
}

/* working-precision scattering vector sc = (diffracted - source)/lambda, 4-element
   convention with element 0 unused. Single: plain float subtract + divide. df64: a
   compensated subtract + divide on both operands. */
__device__ __inline__ static void scattering_vector(const float * diffracted, const float * neg_source, float lambda, float * sc) {
    sc[1] = (diffracted[1] - neg_source[1]) / lambda;
    sc[2] = (diffracted[2] - neg_source[2]) / lambda;
    sc[3] = (diffracted[3] - neg_source[3]) / lambda;
}
__device__ __inline__ static void scattering_vector(const float2 * diffracted, const float2 * neg_source, float2 lambda, float2 * sc) {
    sc[1] = df_div(df_sub(diffracted[1], neg_source[1]), lambda);
    sc[2] = df_div(df_sub(diffracted[2], neg_source[2]), lambda);
    sc[3] = df_div(df_sub(diffracted[3], neg_source[3]), lambda);
}

/* float pre-filter at the dmin site: narrows the Real source vector and lambda to
   feed the cheap resolution cutoff, ahead of the working-precision vector above. At
   single precision this signature coincides with the float form above, so the same
   function serves both call sites; only df64 needs this as a distinct overload. */
__device__ __inline__ static void scattering_vector(const float * diffracted_f, const float2 * neg_src, float2 lambda, float * scattering) {
    scattering[1] = (diffracted_f[1] - real_to_float(neg_src[1])) / real_to_float(lambda);
    scattering[2] = (diffracted_f[2] - real_to_float(neg_src[2])) / real_to_float(lambda);
    scattering[3] = (diffracted_f[3] - real_to_float(neg_src[3])) / real_to_float(lambda);
}

/* inverse of effective detector-thickness increase: dot product of odet_vector and the
   float representative of the diffracted ray. odet_vector is read via __ldg as a
   caching hint; it lives in a read-only const-restrict struct, so this changes no
   results but lets ptxas schedule this cold branch's loads independently of the hot
   pixel_pos/geometry loads, relieving register pressure. The df64 form narrows each
   __ldg'd component with real_to_float before the product, on the same loads. */
__device__ __inline__ static float parallax(const float * __restrict__ odet, const float * diffracted_f) {
    return __ldg(&odet[1]) * diffracted_f[1] + __ldg(&odet[2]) * diffracted_f[2] + __ldg(&odet[3]) * diffracted_f[3];
}
__device__ __inline__ static float parallax(const float2 * __restrict__ odet, const float * diffracted_f) {
    return real_to_float(__ldg(&odet[1])) * diffracted_f[1] + real_to_float(__ldg(&odet[2])) * diffracted_f[2] + real_to_float(__ldg(&odet[3])) * diffracted_f[3];
}

/* sin and cos of a compensated-pair angle, returned as compensated pairs, computed with
   float-only arithmetic. The hardware sinf/cosf carry ~1 ulp of rounding error; that error,
   though tiny, shifts a sharp interference fringe enough to redistribute a fraction of a
   percent of the total flux on a curved detector at fine reciprocal-space sampling. Rebuilding
   the trig in df arithmetic keeps the low word the hardware call would discard.

   Cody-Waite range reduction folds phi into [-pi/4, pi/4] by subtracting k*(pi/2) with pi/2
   held as a (hi, lo) pair, then Horner polynomials evaluated entirely in df pick up sin/cos of
   the reduced angle; the quadrant k selects and signs the two results. Coefficients are the
   Taylor terms carried as pairs so their low words survive too; four terms beyond the constant
   drive the polynomial truncation error below the ~1 ulp the hardware call loses. */
__device__ __inline__ static void df_sincos(float2 phi, float2 * sinphi, float2 * cosphi) {
    const float2 PIO2 = make_float2(1.570796371e+00f, -4.371138829e-08f);
    const float  TWO_OVER_PI = 6.366197467e-01f;
    /* reduce phi to r in [-pi/4, pi/4]; k tracks the quadrant */
    const float kf = rintf(phi.x * TWO_OVER_PI);
    const float2 r = df_sub(phi, df_mul_f(PIO2, kf));
    const float2 r2 = df_mul(r, r);
    /* sin(r) = r * (1 + r2*(c1 + r2*(c2 + r2*(c3 + r2*c4)))) */
    float2 s = make_float2(2.755731884e-06f, 3.793571224e-14f);
    s = df_add(df_mul(s, r2), make_float2(-1.984127011e-04f, 2.725596875e-12f));
    s = df_add(df_mul(s, r2), make_float2( 8.333333768e-03f, -4.346172033e-10f));
    s = df_add(df_mul(s, r2), make_float2(-1.666666716e-01f,  4.967053879e-09f));
    s = df_add_f(df_mul(s, r2), 1.0f);
    const float2 sin_r = df_mul(r, s);
    /* cos(r) = 1 + r2*(d1 + r2*(d2 + r2*(d3 + r2*d4))) */
    float2 c = make_float2(2.480158764e-05f, -3.406996094e-13f);
    c = df_add(df_mul(c, r2), make_float2(-1.388888923e-03f,  3.363109444e-11f));
    c = df_add(df_mul(c, r2), make_float2( 4.166666791e-02f, -1.241763470e-09f));
    c = df_add(df_mul(c, r2), make_float2(-5.000000000e-01f,  0.000000000e+00f));
    const float2 cos_r = df_add_f(df_mul(c, r2), 1.0f);
    /* quadrant fold: q = k mod 4 in {0,1,2,3} */
    const int q = ((int) kf) & 3;
    const float2 neg_sin = make_float2(-sin_r.x, -sin_r.y);
    const float2 neg_cos = make_float2(-cos_r.x, -cos_r.y);
    if (q == 0)      { *sinphi = sin_r;   *cosphi = cos_r; }
    else if (q == 1) { *sinphi = cos_r;   *cosphi = neg_sin; }
    else if (q == 2) { *sinphi = neg_sin; *cosphi = neg_cos; }
    else             { *sinphi = neg_cos; *cosphi = sin_r; }
}

/* compensated-pair form of the axis rotation: rotate the (hi, lo) vector v about the
   (hi, lo) unit axis by the (hi, lo) angle phi, writing the (hi, lo) result into newv.
   sin(phi)/cos(phi) are rebuilt as compensated pairs by df_sincos, so the rotation angle
   stays above single precision through the trig term as well as the vector algebra; every
   vector product and sum is carried as a compensated pair so the rotated position keeps its
   low word instead of collapsing to float. Mirrors the float rotate_axis term for term:
       newv = v*cos + (axis x v)*sin + axis*(axis . v)*(1 - cos)
   4-element convention, element 0 (magnitude) left untouched. */
__device__ __inline__ static void rotate_axis(const float2 * __restrict__ v, float2 * newv, const float2 * __restrict__ axis, const float2 phi) {
    float2 sinphi, cosphi;
    df_sincos(phi, &sinphi, &cosphi);
    const float2 a1 = axis[1];
    const float2 a2 = axis[2];
    const float2 a3 = axis[3];
    const float2 v1 = v[1];
    const float2 v2 = v[2];
    const float2 v3 = v[3];
    float2 dot = df_add(df_add(df_mul(a1, v1), df_mul(a2, v2)), df_mul(a3, v3));
    dot = df_mul(dot, df_sub(make_float2(1.0f, 0.0f), cosphi));

    newv[1] = df_add(df_add(df_mul(a1, dot), df_mul(v1, cosphi)),
                     df_mul(df_sub(df_mul(a2, v3), df_mul(a3, v2)), sinphi));
    newv[2] = df_add(df_add(df_mul(a2, dot), df_mul(v2, cosphi)),
                     df_mul(df_sub(df_mul(a3, v1), df_mul(a1, v3)), sinphi));
    newv[3] = df_add(df_add(df_mul(a3, dot), df_mul(v3, cosphi)),
                     df_mul(df_sub(df_mul(a1, v2), df_mul(a2, v1)), sinphi));
}

/* curved-detector pixel rotation: construct a detector pixel that is always "distance"
   from the sample by rotating pixel_pos about sdet_vector then fdet_vector. The float
   form is the base kernel's exact rotation, operating directly on pixel_pos. The df64
   form carries the rotation angle AND its sin/cos above single precision: the angle is
   pixel_pos[k]/distance formed as a compensated pair, and rotate_axis rebuilds sin/cos in df
   (via df_sincos) so neither the angle's low bits nor the trig rounding collapse to float. At
   short wavelength and large crystals the reciprocal-space sampling is so fine that a
   single-precision angle and single-precision sin/cos shift the rotated pixel a fraction of a
   lattice fringe, redistributing a fraction of a percent of the total flux versus the double
   CPU reference; carrying both in df closes that gap with float-only arithmetic. The
   distance*beam vector rotated below keeps single precision -- its storage precision was
   measured not to affect the result. */
__device__ __inline__ static void curved_position(const float * sdet_vector, const float * fdet_vector, float distance,
        const float * dbvector, float * pixel_pos) {
    float newvector[4];
    rotate_axis(dbvector, newvector, sdet_vector, pixel_pos[2] / distance);
    rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3] / distance);
}
__device__ __inline__ static void curved_position(const float2 * sdet_vector, const float2 * fdet_vector, float distance,
        const float * dbvector, float2 * pixel_pos) {
    /* rotation angles = pixel offset / distance, formed as compensated pairs so their low
       words survive; distance is exact as a float, so a df divide by (distance, 0) suffices. */
    const float2 dist_pair = make_float2(distance, 0.0f);
    const float2 phi_s = df_div(pixel_pos[2], dist_pair);
    const float2 phi_f = df_div(pixel_pos[3], dist_pair);
    /* dbvector is the single-precision sample->distance * beam_vector input; widen it to
       compensated pairs so the whole rotation runs in the pair representation. sdet_vector
       and fdet_vector are already Real (float2) here, so they enter the rotation as pairs
       instead of being narrowed to float. */
    float2 dbv[4];
    dbv[1] = make_real_double(dbvector[1]);
    dbv[2] = make_real_double(dbvector[2]);
    dbv[3] = make_real_double(dbvector[3]);
    float2 newvector[4];
    rotate_axis(dbv, newvector, sdet_vector, phi_s);
    rotate_axis(newvector, pixel_pos, fdet_vector, phi_f);
}

/* make a unit vector pointing in same direction and report magnitude (both args can
   be same vector) */
__device__ static float unitize(float * vector, float * new_unit_vector) {

    float v1 = vector[1];
    float v2 = vector[2];
    float v3 = vector[3];

    float mag = norm3d_fma_rn(v1, v2, v3);

    if (mag != 0.0f) {
        /* normalize it */
        new_unit_vector[0] = mag;
        new_unit_vector[1] = v1 / mag;
        new_unit_vector[2] = v2 / mag;
        new_unit_vector[3] = v3 / mag;
    } else {
        /* can't normalize, report zero vector */
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;
    }
    return mag;
}

/* the working-precision overload narrows the pixel position to float, then calls the
   plain-float form above with the same math in the same order */
__device__ static float unitize(float2 * vector, float * new_unit_vector) {
    float v[4] = { real_to_float(vector[0]), real_to_float(vector[1]), real_to_float(vector[2]), real_to_float(vector[3]) };
    return unitize(v, new_unit_vector);
}

/* amorphous water background intensity per sub-pixel. Single: the base linear chain,
   using r_e_sqr and fluence separately. df64: the compensated chain using the folded
   r_e_sqr*fluence pair, popped to a single float. */
__device__ __inline__ static float water_bg(float water_F, float r_e_sqr, float fluence, float r_e_sqr_fluence,
        float water_size, float Avogadro, float water_MW) {
    return water_F * water_F * r_e_sqr * fluence * water_size * water_size * water_size * 1e6f * Avogadro / water_MW;
}
__device__ __inline__ static float water_bg(float2 water_F, float2 r_e_sqr, float2 fluence, float2 r_e_sqr_fluence,
        float water_size, float2 Avogadro, float2 water_MW) {
    float2 I_bg_df = df_mul(water_F, water_F);
    I_bg_df = df_mul(I_bg_df, r_e_sqr_fluence);
    I_bg_df = df_mul_f(I_bg_df, water_size);
    I_bg_df = df_mul_f(I_bg_df, water_size);
    I_bg_df = df_mul_f(I_bg_df, water_size);
    I_bg_df = df_mul_f(I_bg_df, 1e6f);
    I_bg_df = df_mul(I_bg_df, Avogadro);
    I_bg_df = df_div(I_bg_df, water_MW);
    return I_bg_df.x + I_bg_df.y;
}

/* photons-per-pixel scale: r_e_sqr * fluence * polar * I / steps. Single: the base
   scale expression, using r_e_sqr and fluence separately. df64: the compensated chain
   using the folded r_e_sqr*fluence pair, popped to a single float. */
__device__ __inline__ static float photon_scale(float r_e_sqr, float fluence, float r_e_sqr_fluence, float polar, float I, long steps) {
    return (r_e_sqr * fluence * polar * I) / steps;
}
__device__ __inline__ static float photon_scale(float2 r_e_sqr, float2 fluence, float2 r_e_sqr_fluence, float polar, float I, long steps) {
    float2 photons_df = df_mul_f(r_e_sqr_fluence, polar);
    photons_df = df_mul_f(photons_df, I);
    photons_df = df_div(photons_df, make_float2((float) steps, 0.0f));
    return photons_df.x + photons_df.y;
}
