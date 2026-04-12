/*
 ============================================================================
 Name        : nanoBraggCUDA.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <driver_types.h>
#include "nanotypes.h"
#include "nanoBraggCUDA.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define CUDAREAL float

#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_TOTAL (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
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

struct matrix3x3 {
    CUDAREAL e00;
    CUDAREAL e01;
    CUDAREAL e02;
    CUDAREAL e10;
    CUDAREAL e11;
    CUDAREAL e12;
    CUDAREAL e20;
    CUDAREAL e21;
    CUDAREAL e22;
};

struct vector3 {
    CUDAREAL v1;
    CUDAREAL v2;
    CUDAREAL v3;
};

struct constParams {
    CUDAREAL Avogadro;
    CUDAREAL r_e_sqr;
};

struct beamSource {
    CUDAREAL neg_unit_source_vector[VECTOR_SIZE];
    CUDAREAL intensity;
    CUDAREAL lambda;
};

struct beamParams {
    CUDAREAL beam_vector[VECTOR_SIZE];
    CUDAREAL fluence;
    bool calc_polar;
    CUDAREAL polarization;
    CUDAREAL polar_vector[VECTOR_SIZE];
    short sources;
};

struct detectorParams {
    short spixels;
    short fpixels;
    short roi_xmin;
    short roi_xmax;
    short roi_ymin;
    short roi_ymax;
    short oversample; //Render Setting
    bool point_pixel; //Render Setting
    CUDAREAL pixel_size; //Render Setting
    CUDAREAL subpixel_size; //Render Setting
    long steps; //Render Setting - can be calculated
    CUDAREAL detector_thickstep; //Render Setting
    short detector_thicksteps; //Render Setting
    CUDAREAL detector_thick; //Render Setting - can be calculated
    CUDAREAL detector_mu;
    bool curved_detector;
    CUDAREAL sdet_vector[VECTOR_SIZE];
    CUDAREAL fdet_vector[VECTOR_SIZE];
    CUDAREAL odet_vector[VECTOR_SIZE];
    CUDAREAL pix0_vector[VECTOR_SIZE];
};

struct sampleParams {
    CUDAREAL distance;
    CUDAREAL close_distance;
    CUDAREAL water_size;
    CUDAREAL water_F;
    CUDAREAL water_MW;
};

struct unitCell {
    CUDAREAL a0[VECTOR_SIZE];
    CUDAREAL b0[VECTOR_SIZE];
    CUDAREAL c0[VECTOR_SIZE];
    CUDAREAL V_cell;
};

struct crystalParams {
    unitCell uc;
    CUDAREAL Na;
    CUDAREAL Nb;
    CUDAREAL Nc;
    CUDAREAL V_cell;
    CUDAREAL default_F;
    CUDAREAL dmin;
    shapetype xtal_shape;
    short mosaic_domains;
    bool rotate_umat;
    structureFactorParams fhklParams;
};

struct goniometerParams {
    CUDAREAL phi0;
    CUDAREAL phistep; //Render Setting
    short phisteps; //Render Setting
    CUDAREAL spindle_vector[VECTOR_SIZE];
};

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

static void doubleVectorToRealVector(CUDAREAL * dest, double * src, size_t vector_items) {
    for (size_t i = 0; i < vector_items; i++) {
        dest[i] = src[i];
    }
}

static cudaError_t cudaMemcpyVectorDoubleToDevice(CUDAREAL *dst, double *src, size_t vector_items) {
    CUDAREAL * temp = new CUDAREAL[vector_items];
    doubleVectorToRealVector(temp, src, vector_items);
    cudaError_t ret = cudaMemcpy(dst, temp, sizeof(*dst) * vector_items, cudaMemcpyHostToDevice);
    delete temp;
    return ret;
}

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
double unitizeCPU(double * vector, double * new_unit_vector) {

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

__global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixesl, float * floatimage, float * omega_reduction, float * max_I_x_reduction,
        float * max_I_y_reduction, bool * rangemap);

__global__ void nanoBraggSpotsCUDAKernel_CC6_1(detectorParams * detectorPtr, beamParams * beamPtr, goniometerParams * goniometerPtr, sampleParams * samplePtr,
        crystalParams * crystalPtr, constParams * constantsPtr, const beamSource * __restrict__ beam_sources, const CUDAREAL * __restrict__ Fhkl,
        const matrix3x3 * __restrict__ mosaic_umats, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/, float * max_I_y_reduction /*out*/, bool * rangemap);

__global__ void nanoBraggSpotsCUDAKernel_CC9_0(const detectorParams * __restrict__ detectorPtr, const beamParams * __restrict__ beamPtr, const goniometerParams * __restrict__ goniometerPtr, const sampleParams * __restrict__ samplePtr,
        crystalParams * crystalPtr, const constParams * __restrict__ constantsPtr, const beamSource * __restrict__ beam_sources, const CUDAREAL * __restrict__ Fhkl,
        const matrix3x3 * __restrict__ mosaic_umats, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/, float * max_I_y_reduction /*out*/, bool * rangemap);

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
        double * max_I_y /*out*/) {

    int total_pixels = spixels * fpixels;

    /*allocate and zero reductions */
    bool * rangemap = (bool*) calloc(total_pixels, sizeof(bool));
    float * omega_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_x_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_y_reduction = (float*) calloc(total_pixels, sizeof(float));

    /* clear memory */
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
    detector.subpixel_size = subpixel_size;
    detector.steps = steps;
    detector.detector_thickstep = detector_thickstep;
    detector.detector_thicksteps = detector_thicksteps;
    detector.detector_thick = detector_thick;
    detector.detector_mu = detector_mu;
    detector.curved_detector = curved_detector;
    doubleVectorToRealVector(detector.sdet_vector, sdet_vector, VECTOR_SIZE);
    doubleVectorToRealVector(detector.fdet_vector, fdet_vector, VECTOR_SIZE);
    doubleVectorToRealVector(detector.odet_vector, odet_vector, VECTOR_SIZE);
    doubleVectorToRealVector(detector.pix0_vector, pix0_vector, VECTOR_SIZE);

    beamParams beam;
    doubleVectorToRealVector(beam.beam_vector, beam_vector, VECTOR_SIZE);
    beam.fluence = fluence;
    beam.calc_polar = !nopolar;
    beam.polarization = polarization;
    //  Unitize polar vector before sending it to the GPU. Optimization do it only once here rather than multiple time per pixel in the GPU.
    double unit_polar_vector[VECTOR_SIZE];
    unitizeCPU(polar_vector, unit_polar_vector);
    doubleVectorToRealVector(beam.polar_vector, unit_polar_vector, VECTOR_SIZE);
    beam.sources = sources;

    goniometerParams goniometer;
    goniometer.phi0 = phi0;
    goniometer.phistep = phistep;
    goniometer.phisteps = phisteps;
    doubleVectorToRealVector(goniometer.spindle_vector, spindle_vector, VECTOR_SIZE);

    sampleParams sample;
    sample.distance = distance;
    sample.close_distance = close_distance;
    sample.water_F = water_F;
    sample.water_size = water_size;
    sample.water_MW = water_MW;

    crystalParams crystal;
    crystal.default_F = default_F;
    crystal.dmin = dmin;
    crystal.Na = Na;
    crystal.Nb = Nb;
    crystal.Nc = Nc;
    crystal.uc.V_cell = V_cell;
    doubleVectorToRealVector(crystal.uc.a0, a0, VECTOR_SIZE);
    doubleVectorToRealVector(crystal.uc.b0, b0, VECTOR_SIZE);
    doubleVectorToRealVector(crystal.uc.c0, c0, VECTOR_SIZE);
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
    crystal.rotate_umat = mosaic_spread > 0.0;

    // Pad hkl with default_F value;
    int h_min_pad = h_min - 1, h_max_pad = h_max + 1, h_range_pad = h_range + 2;
    int k_min_pad = k_min - 1, k_max_pad = k_max + 1, k_range_pad = k_range + 2;
    int l_min_pad = l_min - 1, l_max_pad = l_max + 1, l_range_pad = l_range + 2;
    int hklsize_pad = h_range_pad * k_range_pad * l_range_pad;
    CUDAREAL * FhklLinearPad = (CUDAREAL*) calloc(hklsize_pad, sizeof(*FhklLinearPad));
    for (int h = 0; h < h_range_pad; h++) {
        for (int k = 0; k < k_range_pad; k++) {
            for (int l = 0; l < l_range_pad; l++) {
                //  convert Fhkl double to CUDAREAL
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
    constants.Avogadro = Avogadro;
    constants.r_e_sqr = r_e_sqr;

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

    // Repackage beam sources. Unitize source vectors and pack them together contiguously.
    beamSource * beam_sources = new beamSource[beam.sources];
    for (int i = 0; i < beam.sources; i++) {
        double unitSource[VECTOR_SIZE] = { 0.0, -source_X[i], -source_Y[i], -source_Z[i] };
        unitizeCPU(unitSource, unitSource);
        doubleVectorToRealVector(beam_sources[i].neg_unit_source_vector, unitSource, VECTOR_SIZE);
        beam_sources[i].lambda = source_lambda[i];
        beam_sources[i].intensity = source_I[i];
    }
    beamSource * cu_beam_sources = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam_sources, sizeof(*cu_beam_sources) * beam.sources));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_beam_sources, beam_sources, sizeof(*cu_beam_sources) * beam.sources, cudaMemcpyHostToDevice));

    // Convert mosaic domains.
    matrix3x3 * mosaic_umat3x3s = new matrix3x3[crystal.mosaic_domains];
    for (int i = 0; i < crystal.mosaic_domains; i++) {
        mosaic_umat3x3s[i].e00 = mosaic_umats[i * 9 + 0];
        mosaic_umat3x3s[i].e01 = mosaic_umats[i * 9 + 1];
        mosaic_umat3x3s[i].e02 = mosaic_umats[i * 9 + 2];
        mosaic_umat3x3s[i].e10 = mosaic_umats[i * 9 + 3];
        mosaic_umat3x3s[i].e11 = mosaic_umats[i * 9 + 4];
        mosaic_umat3x3s[i].e12 = mosaic_umats[i * 9 + 5];
        mosaic_umat3x3s[i].e20 = mosaic_umats[i * 9 + 6];
        mosaic_umat3x3s[i].e21 = mosaic_umats[i * 9 + 7];
        mosaic_umat3x3s[i].e22 = mosaic_umats[i * 9 + 8];
    }
    matrix3x3 * cu_mosaic_umats = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_mosaic_umats, sizeof(*cu_mosaic_umats) * crystal.mosaic_domains));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_mosaic_umats, mosaic_umat3x3s, sizeof(*cu_mosaic_umats) * crystal.mosaic_domains, cudaMemcpyHostToDevice));

    CUDAREAL * cu_Fhkl = NULL;
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
    cudaDeviceProp deviceProps = { 0 };
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProps, deviceId));
    int smCount = deviceProps.multiProcessorCount;
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));

//	CUDA_CHECK_RETURN(cudaFuncSetCacheConfig(nanoBraggSpotsCUDAKernel, cudaFuncCachePreferShared));
//	CUDA_CHECK_RETURN(cudaFuncSetCacheConfig(nanoBraggSpotsCUDAKernel, cudaFuncCachePreferL1));

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    //  dim3 numBlocks((spixels - 1) / threadsPerBlock.x + 1, (fpixels - 1) / threadsPerBlock.y + 1);
    dim3 numBlocks(smCount * 32, 1);

    //  initialize the device memory within a kernel.
    //	nanoBraggSpotsInitCUDAKernel<<<numBlocks, threadsPerBlock>>>(cu_spixels, cu_fpixels, cu_floatimage, cu_omega_reduction, cu_max_I_x_reduction, cu_max_I_y_reduction, cu_rangemap);
    //  CUDA_CHECK_RETURN(cudaPeekAtLastError());
    //  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    nanoBraggSpotsCUDAKernel_CC9_0<<<numBlocks, threadsPerBlock>>>(cu_detector, cu_beam, cu_goniometer, cu_sample, cu_crystal, cu_constants, cu_beam_sources, cu_Fhkl,
            cu_mosaic_umats, cu_maskimage, cu_floatimage /*out*/, cu_omega_reduction/*out*/, cu_max_I_x_reduction/*out*/, cu_max_I_y_reduction /*out*/, cu_rangemap /*out*/);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

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
    CUDA_CHECK_RETURN(cudaFree(cu_mosaic_umats));
    CUDA_CHECK_RETURN(cudaFree(cu_floatimage));
    CUDA_CHECK_RETURN(cudaFree(cu_omega_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_x_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_y_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_maskimage));
    CUDA_CHECK_RETURN(cudaFree(cu_rangemap));

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

    delete beam_sources;
    delete mosaic_umat3x3s;
    free(FhklLinearPad);
    free(rangemap);
    free(omega_reduction);
    free(max_I_x_reduction);
    free(max_I_y_reduction);
}

/* cubic spline interpolation functions */
__device__ static void polint(CUDAREAL *xa, CUDAREAL *ya, CUDAREAL x,
CUDAREAL *y);
__device__ static void polin2(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL ya[4][4],
CUDAREAL x1, CUDAREAL x2, CUDAREAL *y);
__device__ static void polin3(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL *x3a,
CUDAREAL ya[4][4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL x3,
CUDAREAL *y);
/* rotate a 3-vector about a unit vector axis */
__device__ static CUDAREAL *rotate_axis(const CUDAREAL * __restrict__ v,
CUDAREAL *newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi);
__device__ static CUDAREAL *rotate_axis_ldg(const CUDAREAL * __restrict__ v,
CUDAREAL * newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi);
/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
__device__ static CUDAREAL unitize(CUDAREAL * vector,
CUDAREAL *new_unit_vector);
/* vector cross product where vector magnitude is 0th element */
__device__ static CUDAREAL *cross_product(const CUDAREAL * x, const CUDAREAL * y, CUDAREAL * z);
/* vector inner product where vector magnitude is 0th element */
__device__ __inline__ static CUDAREAL dot_product(const CUDAREAL * x, const CUDAREAL * y);
__device__ __inline__ static CUDAREAL dot_product_ldg(const CUDAREAL * __restrict__ x, CUDAREAL * y);
/* measure magnitude of vector and put it in 0th element */
__device__ static void magnitude(CUDAREAL *vector);
/* scale the magnitude of a vector */
__device__ static CUDAREAL vector_scale(CUDAREAL *vector, CUDAREAL *new_vector,
CUDAREAL scale);
/* rotate a 3-vector using a 9-element unitary matrix */
__device__ __inline__ void rotate_umat_ldg(CUDAREAL * v, CUDAREAL *newv, const matrix3x3 * umat);
/* Fourier transform of a truncated lattice */
__device__ __inline__ static CUDAREAL sincg(CUDAREAL x, CUDAREAL N);
/* Fourier transform of a sphere */
__device__ static CUDAREAL sinc3(CUDAREAL x);
/* polarization factor from vectors */
__device__ static CUDAREAL polarization_factor(CUDAREAL kahn_factor, const CUDAREAL * __restrict__ unitIncident, CUDAREAL *unitDiffracted,
        const CUDAREAL * __restrict__ unitAxis);
__device__ CUDAREAL polarization_factor_ldg(CUDAREAL kahn_factor, const CUDAREAL * __restrict__ unitIncident, CUDAREAL *unitDiffracted,
        const CUDAREAL * __restrict__ unitAxis);

__device__ __inline__ static long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range);
__device__ __inline__ static const CUDAREAL * vector_address(const CUDAREAL * base_address, int idx, int vector_size);

__device__ __inline__ CUDAREAL quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const CUDAREAL * __restrict__ Fhkl);
__device__ __inline__ void quickFcell_ldg(int hkls, CUDAREAL h, int h_max, int h_min, CUDAREAL k, int k_max, int k_min, CUDAREAL l, int l_max, int l_min,
        int h_range, int k_range, int l_range, const CUDAREAL * __restrict__ Fhkl, CUDAREAL * F_cell);

__device__ __inline__ CUDAREAL norm3d_fma_rn(CUDAREAL v1, CUDAREAL v2, CUDAREAL v3);

__global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixels, float * floatimage, float * omega_reduction, float * max_I_x_reduction,
        float * max_I_y_reduction, bool * rangemap) {

    const int total_pixels = spixels * fpixels;
    const int fstride = gridDim.x * blockDim.x;
    const int sstride = gridDim.y * blockDim.y;
    const int stride = fstride * sstride;

    for (int pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
        const int fpixel = pixIdx % fpixels;
        const int spixel = pixIdx / fpixels;

        /* position in pixel array */
        int j = spixel * fpixels + fpixel;

        if (j < total_pixels) {
            floatimage[j] = 0;
            omega_reduction[j] = 0;
            max_I_x_reduction[j] = 0;
            max_I_y_reduction[j] = 0;
            rangemap[j] = false;
        }
    }
}

__global__ void nanoBraggSpotsCUDAKernel_CC6_1(detectorParams * detector, beamParams * beam, goniometerParams * goniometer, sampleParams * sample,
        crystalParams * crystal, constParams * constants, const beamSource * __restrict__ beam_sources, const CUDAREAL * __restrict__ Fhkl,
        const matrix3x3 * __restrict__ mosaic_umats, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
        float * max_I_y_reduction /*out*/, bool * rangemap) {

    __shared__ short s_phisteps;
    __shared__ CUDAREAL s_phi0, s_phistep;
    __shared__ short s_mosaic_domains;

    __shared__ CUDAREAL s_Na, s_Nb, s_Nc;
    __shared__ short s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, s_h_range, s_k_range, s_l_range;

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        s_phisteps = goniometer->phisteps;
        s_phi0 = goniometer->phi0;
        s_phistep = goniometer->phistep;

        s_mosaic_domains = crystal->mosaic_domains;

        s_Na = crystal->Na;
        s_Nb = crystal->Nb;
        s_Nc = crystal->Nc;

        s_hkls = crystal->fhklParams.hkls;
        s_h_max = crystal->fhklParams.h_max;
        s_h_min = crystal->fhklParams.h_min;
        s_k_max = crystal->fhklParams.k_max;
        s_k_min = crystal->fhklParams.k_min;
        s_l_max = crystal->fhklParams.l_max;
        s_l_min = crystal->fhklParams.l_min;
        s_h_range = crystal->fhklParams.h_range;
        s_k_range = crystal->fhklParams.k_range;
        s_l_range = crystal->fhklParams.l_range;

    }
    __syncthreads();

    long total_pixels = detector->spixels * detector->fpixels;
    long fstride = gridDim.x * blockDim.x;
    long sstride = gridDim.y * blockDim.y;
    long stride = fstride * sstride;

    /* add background from something amorphous */
    CUDAREAL F_bg = sample->water_F;
    CUDAREAL I_bg = F_bg * F_bg * constants->r_e_sqr * beam->fluence * sample->water_size * sample->water_size * sample->water_size * 1e6 * constants->Avogadro / sample->water_MW;

    for (long pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
        short fpixel = pixIdx % detector->fpixels;
        short spixel = pixIdx / detector->fpixels;

        /* allow for just one part of detector to be rendered */
        if (fpixel < detector->roi_xmin || fpixel > detector->roi_xmax || spixel < detector->roi_ymin || spixel > detector->roi_ymax) { //ROI region of interest
            continue;
        }

        /* allow for the use of a mask */
        if (maskimage != NULL) {
            /* skip any flagged pixels in the mask */
            if (maskimage[pixIdx] == 0) {
                continue;
            }
        }

        /* reset photon count for this pixel */
        CUDAREAL I = I_bg;
        CUDAREAL omega_sub_reduction = 0.0;
        CUDAREAL max_I_x_sub_reduction = 0.0;
        CUDAREAL max_I_y_sub_reduction = 0.0;
        CUDAREAL polar = 0.0;

        /* loop over sub-pixels */
        for (short subS = 0; subS < detector->oversample; ++subS) { // Y voxel
            for (short subF = 0; subF < detector->oversample; ++subF) { // X voxel
                /* absolute mm position on detector (relative to its origin) */
                CUDAREAL Fdet = detector->subpixel_size * (fpixel * detector->oversample + subF) + detector->subpixel_size / 2.0; // X voxel
                CUDAREAL Sdet = detector->subpixel_size * (spixel * detector->oversample + subS) + detector->subpixel_size / 2.0; // Y voxel

                max_I_x_sub_reduction = Fdet;
                max_I_y_sub_reduction = Sdet;

                for (short thick_tic = 0; thick_tic < detector->detector_thicksteps; ++thick_tic) {
                    /* assume "distance" is to the front of the detector sensor layer */
                    CUDAREAL Odet = thick_tic * detector->detector_thickstep; // Z Orthagonal voxel.

                    /* construct detector subpixel position in 3D space */
                    CUDAREAL pixel_pos[4];
                    pixel_pos[1] = Fdet * __ldg(&detector->fdet_vector[1]) + Sdet * __ldg(&detector->sdet_vector[1]) + Odet * __ldg(&detector->odet_vector[1]) + __ldg(&detector->pix0_vector[1]); // X
                    pixel_pos[2] = Fdet * __ldg(&detector->fdet_vector[2]) + Sdet * __ldg(&detector->sdet_vector[2]) + Odet * __ldg(&detector->odet_vector[2]) + __ldg(&detector->pix0_vector[2]); // Y
                    pixel_pos[3] = Fdet * __ldg(&detector->fdet_vector[3]) + Sdet * __ldg(&detector->sdet_vector[3]) + Odet * __ldg(&detector->odet_vector[3]) + __ldg(&detector->pix0_vector[3]); // Z

                    /* construct the diffracted-beam unit vector to this sub-pixel */
                    CUDAREAL diffracted[4];
                    CUDAREAL airpath = unitize(pixel_pos, diffracted);

                    /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                    CUDAREAL omega_pixel = detector->pixel_size * detector->pixel_size / airpath / airpath * sample->close_distance / airpath;

                    /* now calculate detector thickness effects */
                    CUDAREAL capture_fraction = 1.0;

                    /* loop over sources now */
                    for (short source = 0; source < beam->sources; ++source) {

                        CUDAREAL lambda = __ldg(&beam_sources[source].lambda);

                        CUDAREAL scattering[4];
                        scattering[1] = (diffracted[1] - __ldg(&beam_sources[source].neg_unit_source_vector[1])) / lambda;
                        scattering[2] = (diffracted[2] - __ldg(&beam_sources[source].neg_unit_source_vector[2])) / lambda;
                        scattering[3] = (diffracted[3] - __ldg(&beam_sources[source].neg_unit_source_vector[3])) / lambda;

                        /* polarization factor */
                        /* need to compute polarization factor */
                        CUDAREAL incident[4];
                        incident[1] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                        incident[2] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                        incident[3] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                        polar = polarization_factor(beam->polarization, incident, diffracted, beam->polar_vector);

                        /* sweep over phi angles */
                        for (short phi_tic = 0; phi_tic < s_phisteps; ++phi_tic) {
                            CUDAREAL phi = s_phistep * phi_tic + s_phi0;

                            CUDAREAL ap[4];
                            CUDAREAL bp[4];
                            CUDAREAL cp[4];

                            /* rotate about spindle if necessary */
                            rotate_axis_ldg(crystal->uc.a0, ap, goniometer->spindle_vector, phi);
                            rotate_axis_ldg(crystal->uc.b0, bp, goniometer->spindle_vector, phi);
                            rotate_axis_ldg(crystal->uc.c0, cp, goniometer->spindle_vector, phi);

                            /* enumerate mosaic domains */
                            for (short mos_tic = 0; mos_tic < s_mosaic_domains; ++mos_tic) {
                                /* apply mosaic rotation after phi rotation */
                                CUDAREAL a[4];
                                CUDAREAL b[4];
                                CUDAREAL c[4];
                                rotate_umat_ldg(ap, a, &mosaic_umats[mos_tic]);
                                rotate_umat_ldg(bp, b, &mosaic_umats[mos_tic]);
                                rotate_umat_ldg(cp, c, &mosaic_umats[mos_tic]);
                                CUDAREAL h = dot_product(a, scattering);
                                CUDAREAL k = dot_product(b, scattering);
                                CUDAREAL l = dot_product(c, scattering);

                                /* round off to nearest whole index */
                                short h0 = ceil(h - 0.5);
                                short k0 = ceil(k - 0.5);
                                short l0 = ceil(l - 0.5);

                                CUDAREAL F_latt = sincg(M_PI * h, s_Na) * sincg(M_PI * k, s_Nb) * sincg(M_PI * l, s_Nc);
                                CUDAREAL F_cell =
                                        s_hkls ? quickFcell_ldg(s_hkls, h0, s_h_max, s_h_min, k0, s_k_max, s_k_min, l0, s_l_max, s_l_min, s_h_range, s_k_range, s_l_range, Fhkl) : crystal->default_F;

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
        floatimage[pixIdx] = I_bg + (constants->r_e_sqr * beam->fluence * polar * I) / detector->steps;
        omega_reduction[pixIdx] = omega_sub_reduction; // shared contention
        max_I_x_reduction[pixIdx] = max_I_x_sub_reduction;
        max_I_y_reduction[pixIdx] = max_I_y_sub_reduction;
        rangemap[pixIdx] = true;
    }
}

__global__ void nanoBraggSpotsCUDAKernel_CC9_0(const detectorParams * __restrict__ detector, const beamParams * __restrict__ beam, const goniometerParams * __restrict__ goniometer, const sampleParams * __restrict__ sample,
        crystalParams * crystal, const constParams * __restrict__ constants, const beamSource * __restrict__ beam_sources, const CUDAREAL * __restrict__ Fhkl,
        const matrix3x3 * __restrict__ mosaic_umats, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
        float * max_I_y_reduction /*out*/, bool * rangemap) {

    __shared__ CUDAREAL s_Na, s_Nb, s_Nc;
    __shared__ short s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, s_h_range, s_k_range, s_l_range;

    if (threadIdx.x == 0 && threadIdx.y == 0) {


        s_Na = crystal->Na;
        s_Nb = crystal->Nb;
        s_Nc = crystal->Nc;

        s_hkls = crystal->fhklParams.hkls;
        s_h_max = crystal->fhklParams.h_max;
        s_h_min = crystal->fhklParams.h_min;
        s_k_max = crystal->fhklParams.k_max;
        s_k_min = crystal->fhklParams.k_min;
        s_l_max = crystal->fhklParams.l_max;
        s_l_min = crystal->fhklParams.l_min;
        s_h_range = crystal->fhklParams.h_range;
        s_k_range = crystal->fhklParams.k_range;
        s_l_range = crystal->fhklParams.l_range;

//      printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", s_hkls, s_h_max, s_h_min, s_k_max, s_k_min, s_l_max, s_l_min, s_h_range, s_k_range, s_l_range);

    }
    __syncthreads();

    const long total_pixels = detector->spixels * detector->fpixels;
    const long fstride = gridDim.x * blockDim.x;
    const long sstride = gridDim.y * blockDim.y;
    const long stride = fstride * sstride;

    /* add background from something amorphous */
    const CUDAREAL F_bg = sample->water_F;
    const CUDAREAL I_bg = F_bg * F_bg * constants->r_e_sqr * beam->fluence * sample->water_size * sample->water_size * sample->water_size * 1e6 * constants->Avogadro / sample->water_MW;


    for (long pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
        const short fpixel = pixIdx % detector->fpixels;
        const short spixel = pixIdx / detector->fpixels;

        /* allow for just one part of detector to be rendered */
        if (fpixel < detector->roi_xmin || fpixel > detector->roi_xmax || spixel < detector->roi_ymin || spixel > detector->roi_ymax) { //ROI region of interest
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

        /* reset photon count for this pixel */
        CUDAREAL I = I_bg;
        CUDAREAL omega_sub_reduction = 0.0;
        CUDAREAL max_I_x_sub_reduction = 0.0;
        CUDAREAL max_I_y_sub_reduction = 0.0;
        CUDAREAL polar = 0.0;
        if (beam->calc_polar) {
            polar = 1.0;
        }

        /* add this now to avoid problems with skipping later */
        // move this to the bottom to avoid accessing global device memory. floatimage[j] = I_bg;
        /* loop over sub-pixels */
        for (short subS = 0; subS < detector->oversample; ++subS) { // Y voxel
            for (short subF = 0; subF < detector->oversample; ++subF) { // X voxel
                /* absolute mm position on detector (relative to its origin) */
                CUDAREAL Fdet = detector->subpixel_size * (fpixel * detector->oversample + subF) + detector->subpixel_size / (CUDAREAL)2.0; // X voxel
                CUDAREAL Sdet = detector->subpixel_size * (spixel * detector->oversample + subS) + detector->subpixel_size / (CUDAREAL)2.0; // Y voxel

                max_I_x_sub_reduction = Fdet;
                max_I_y_sub_reduction = Sdet;

                for (short thick_tic = 0; thick_tic < detector->detector_thicksteps; ++thick_tic) {
                    /* assume "distance" is to the front of the detector sensor layer */
                    CUDAREAL Odet = thick_tic * detector->detector_thickstep; // Z Orthagonal voxel.

                    /* construct detector subpixel position in 3D space */
                    CUDAREAL pixel_pos[4];
                    pixel_pos[1] = Fdet * detector->fdet_vector[1] + Sdet * detector->sdet_vector[1] + Odet * detector->odet_vector[1] + detector->pix0_vector[1]; // X
                    pixel_pos[2] = Fdet * detector->fdet_vector[2] + Sdet * detector->sdet_vector[2] + Odet * detector->odet_vector[2] + detector->pix0_vector[2]; // Y
                    pixel_pos[3] = Fdet * detector->fdet_vector[3] + Sdet * detector->sdet_vector[3] + Odet * detector->odet_vector[3] + detector->pix0_vector[3]; // Z

                    if (detector->curved_detector) {
                        /* construct detector pixel that is always "distance" from the sample */
                        CUDAREAL dbvector[4];
                        dbvector[1] = sample->distance * beam->beam_vector[1];
                        dbvector[2] = sample->distance * beam->beam_vector[2];
                        dbvector[3] = sample->distance * beam->beam_vector[3];
                        /* treat detector pixel coordinates as radians */
                        CUDAREAL newvector[4];
                        rotate_axis(dbvector, newvector, detector->sdet_vector, pixel_pos[2] / sample->distance);
                        rotate_axis(newvector, pixel_pos, detector->fdet_vector, pixel_pos[3] / sample->distance);
                    }
                    /* construct the diffracted-beam unit vector to this sub-pixel */
                    CUDAREAL diffracted[4];
                    CUDAREAL airpath = unitize(pixel_pos, diffracted);

                    /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                    CUDAREAL omega_pixel = detector->pixel_size * detector->pixel_size / airpath / airpath * sample->close_distance / airpath;
                    /* option to turn off obliquity effect, inverse-square-law only */
                    if (detector->point_pixel) {
                        omega_pixel = 1.0 / airpath / airpath;
                    }

                    /* now calculate detector thickness effects */
                    CUDAREAL capture_fraction = 1.0;
                    if (detector->detector_thick > 0.0) {
                        /* inverse of effective thickness increase */
                        CUDAREAL parallax = dot_product_ldg(detector->odet_vector, diffracted);
                        capture_fraction = exp(-thick_tic * detector->detector_thickstep * detector->detector_mu / parallax)
                                - exp(-(thick_tic + 1) * detector->detector_thickstep * detector->detector_mu / parallax);
                    }

                    /* loop over sources now */
                    for (short source = 0; source < beam->sources; ++source) {

                        /* retrieve stuff from cache */
                        CUDAREAL lambda = __ldg(&beam_sources[source].lambda);

                        /* construct the incident beam unit vector while recovering source distance */
                        /* construct the scattering vector for this pixel */
                        CUDAREAL scattering[4];
                        scattering[1] = (diffracted[1] - __ldg(&beam_sources[source].neg_unit_source_vector[1])) / lambda;
                        scattering[2] = (diffracted[2] - __ldg(&beam_sources[source].neg_unit_source_vector[2])) / lambda;
                        scattering[3] = (diffracted[3] - __ldg(&beam_sources[source].neg_unit_source_vector[3])) / lambda;

                        /* sin(theta)/lambda is half the scattering vector length */
                        CUDAREAL stol = (CUDAREAL)0.5 * norm3d_fma_rn(scattering[1], scattering[2], scattering[3]);

                        /* rough cut to speed things up when we aren't using whole detector */
                        if (crystal->dmin > 0.0 && stol > 0.0) {
                            if (crystal->dmin > 0.5f / stol) {
                                continue;
                            }
                        }

                        /* polarization factor */
                        if (beam->calc_polar) {
                            /* need to compute polarization factor */
                            CUDAREAL incident[4];
                            incident[1] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                            incident[2] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                            incident[3] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                            polar = polarization_factor(beam->polarization, incident, diffracted, beam->polar_vector);
                        }

                        /* sweep over phi angles */
                        for (short phi_tic = 0; phi_tic < goniometer->phisteps; ++phi_tic) {
                            CUDAREAL phi = goniometer->phistep * phi_tic + goniometer->phi0;

//                          CUDAREAL ap[] = { 0.0, 0.0, 0.0, 0.0 };
//                          CUDAREAL bp[] = { 0.0, 0.0, 0.0, 0.0 };
//                          CUDAREAL cp[] = { 0.0, 0.0, 0.0, 0.0 };
                            CUDAREAL ap[4];
                            CUDAREAL bp[4];
                            CUDAREAL cp[4];

                            /* rotate about spindle if necessary */
                            rotate_axis_ldg(crystal->uc.a0, ap, goniometer->spindle_vector, phi);
                            rotate_axis_ldg(crystal->uc.b0, bp, goniometer->spindle_vector, phi);
                            rotate_axis_ldg(crystal->uc.c0, cp, goniometer->spindle_vector, phi);

                            /* enumerate mosaic domains */
                            for (short mos_tic = 0; mos_tic < crystal->mosaic_domains; ++mos_tic) {
                                /* apply mosaic rotation after phi rotation */
                                CUDAREAL a[4];
                                CUDAREAL b[4];
                                CUDAREAL c[4];
                                if (crystal->rotate_umat) {
                                    rotate_umat_ldg(ap, a, &mosaic_umats[mos_tic]);
                                    rotate_umat_ldg(bp, b, &mosaic_umats[mos_tic]);
                                    rotate_umat_ldg(cp, c, &mosaic_umats[mos_tic]);
                                } else {
                                    a[1] = ap[1];
                                    a[2] = ap[2];
                                    a[3] = ap[3];
                                    b[1] = bp[1];
                                    b[2] = bp[2];
                                    b[3] = bp[3];
                                    c[1] = cp[1];
                                    c[2] = cp[2];
                                    c[3] = cp[3];
                                }
                                //                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+0],mosaic_umats[mos_tic*9+1],mosaic_umats[mos_tic*9+2]);
                                //                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+3],mosaic_umats[mos_tic*9+4],mosaic_umats[mos_tic*9+5]);
                                //                                  printf("%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+6],mosaic_umats[mos_tic*9+7],mosaic_umats[mos_tic*9+8]);

                                /* construct fractional Miller indicies */

                                CUDAREAL h = dot_product(a, scattering);
                                CUDAREAL k = dot_product(b, scattering);
                                CUDAREAL l = dot_product(c, scattering);

                                /* round off to nearest whole index */
                                short h0 = ceil(h - (CUDAREAL)0.5);
                                short k0 = ceil(k - (CUDAREAL)0.5);
                                short l0 = ceil(l - (CUDAREAL)0.5);

                                /* structure factor of the lattice (paralelpiped crystal)
                                 F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
                                 */
                                CUDAREAL F_latt = 1.0; // Shape transform for the crystal.
                                if (crystal->xtal_shape == SQUARE) {
                                    /* xtal is a paralelpiped */
                                    if (crystal->Na > 1) {
                                        F_latt *= sincg((CUDAREAL)M_PI * h, crystal->Na);
                                    }
                                    if (crystal->Nb > 1) {
                                        F_latt *= sincg((CUDAREAL)M_PI * k, crystal->Nb);
                                    }
                                    if (crystal->Nc > 1) {
                                        F_latt *= sincg((CUDAREAL)M_PI * l, crystal->Nc);
                                    }
                                } else if (crystal->xtal_shape == ROUND) {
                                    /* use sinc3 for elliptical xtal shape,
                                     correcting for sqrt of volume ratio between cube and sphere */
                                    CUDAREAL hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * 0.723601254558268 * sinc3(M_PI * sqrt(hrad_sqr * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == GAUSS) {
                                    /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                    CUDAREAL hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * exp(-(hrad_sqr / 0.63 * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == TOPHAT) {
                                    /* make a flat-top spot of same height and volume as square_xtal spots */
                                    CUDAREAL hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * (hrad_sqr * 1.0f /*fudge*/ < 0.3969);
                                }
                                /* no need to go further if result will be zero? */
//                              if (F_latt == 0.0)
//                                  continue;
                                /* structure factor of the unit cell */
//                              CUDAREAL F_cell = s_default_F;
                                CUDAREAL F_cell =
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
        const double photons = I_bg + (constants->r_e_sqr * beam->fluence * polar * I) / detector->steps;
        floatimage[j] = photons;
        omega_reduction[j] = omega_sub_reduction; // shared contention
        max_I_x_reduction[j] = max_I_x_sub_reduction;
        max_I_y_reduction[j] = max_I_y_sub_reduction;
        rangemap[j] = true;
    }
}

__device__ __inline__ CUDAREAL quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const CUDAREAL * __restrict__ Fhkl) {
//	if (hkls && (h0 <= h_max) && (h0 >= h_min) && (k0 <= k_max) && (k0 >= k_min) && (l0 <= l_max) && (l0 >= l_min)) {
//		/* just take nearest-neighbor */
//		*F_cell = __ldg(&Fhkl[flatten3dindex(h0 - h_min, k0 - k_min, l0 - l_min, h_range, k_range, l_range)]);
//	}
    short h = min(max(h0 - h_min, 0), h_range);
    short k = min(max(k0 - k_min, 0), k_range);
    short l = min(max(l0 - l_min, 0), l_range);
    return __ldg(&Fhkl[flatten3dindex(h, k, l, h_range, k_range, l_range)]);
}

__device__ __inline__ void quickFcell_ldg(int hkls, CUDAREAL h, int h_max, int h_min, CUDAREAL k, int k_max, int k_min, CUDAREAL l, int l_max, int l_min,
        int h_range, int k_range, int l_range, const CUDAREAL * __restrict__ Fhkl, CUDAREAL * F_cell) {
//	if (hkls && (h0 <= h_max) && (h0 >= h_min) && (k0 <= k_max) && (k0 >= k_min) && (l0 <= l_max) && (l0 >= l_min)) {
//		/* just take nearest-neighbor */
//		*F_cell = __ldg(&Fhkl[flatten3dindex(h0 - h_min, k0 - k_min, l0 - l_min, h_range, k_range, l_range)]);
//	}
    int h0 = min(max((int) ceil(h - h_min - 0.5), 0), h_range);
    int k0 = min(max((int) ceil(k - k_min - 0.5), 0), k_range);
    int l0 = min(max((int) ceil(l - l_min - 0.5), 0), l_range);
    *F_cell = __ldg(&Fhkl[flatten3dindex(h0, k0, l0, h_range, k_range, l_range)]);
}

__device__ __inline__ long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range) {
    return x * y_range * z_range + y * z_range + z;
}

__device__ __inline__ static const CUDAREAL * vector_address(const CUDAREAL * base_address, int idx, int vector_size) {
    return &base_address[idx * vector_size];
}

/* rotate a point about a unit vector axis */
__device__ CUDAREAL *rotate_axis(const CUDAREAL * __restrict__ v,
CUDAREAL * newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi) {

    const CUDAREAL sinphi = sin(phi);
    const CUDAREAL cosphi = cos(phi);
    const CUDAREAL a1 = axis[1];
    const CUDAREAL a2 = axis[2];
    const CUDAREAL a3 = axis[3];
    const CUDAREAL v1 = v[1];
    const CUDAREAL v2 = v[2];
    const CUDAREAL v3 = v[3];
    const CUDAREAL dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0 - cosphi);

    newv[1] = __fmaf_rn(a1, dot, v1 * cosphi) + __fmaf_rn(-a3, v2, a2 * v3) * sinphi;
    newv[2] = __fmaf_rn(a2, dot, v2 * cosphi) + __fmaf_rn(+a3, v1, -a1 * v3) * sinphi;
    newv[3] = __fmaf_rn(a3, dot, v3 * cosphi) + __fmaf_rn(-a2, v1, a1 * v2) * sinphi;
    // newv[1] = a1 * dot + v1 * cosphi + (+a3 * v1 - a1 * v3) * sinphi;
    // newv[2] = a2 * dot + v2 * cosphi + (+a3 * v1 - a1 * v3) * sinphi;
    // newv[3] = a3 * dot + v3 * cosphi + (-a2 * v1 + a1 * v2) * sinphi;

    return newv;
}

/* rotate a point about a unit vector axis */
__device__ CUDAREAL *rotate_axis_ldg(const CUDAREAL * __restrict__ v,
CUDAREAL * newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi) {

    const CUDAREAL sinphi = sin(phi);
    const CUDAREAL cosphi = cos(phi);
    const CUDAREAL a1 = __ldg(&axis[1]);
    const CUDAREAL a2 = __ldg(&axis[2]);
    const CUDAREAL a3 = __ldg(&axis[3]);
    const CUDAREAL v1 = __ldg(&v[1]);
    const CUDAREAL v2 = __ldg(&v[2]);
    const CUDAREAL v3 = __ldg(&v[3]);
    
//    const CUDAREAL dot = __fmaf_rn(a3, v3, __fmaf_rn(a1, v1, a2 * v2)) * (1.0 - cosphi);
    
    const CUDAREAL dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0f - cosphi);

    newv[1] = __fmaf_rn(a1, dot, v1 * cosphi) + __fmaf_rn(-a3, v2, a2 * v3) * sinphi;
    newv[2] = __fmaf_rn(a2, dot, v2 * cosphi) + __fmaf_rn(+a3, v1, -a1 * v3) * sinphi;
    newv[3] = __fmaf_rn(a3, dot, v3 * cosphi) + __fmaf_rn(-a2, v1, a1 * v2) * sinphi;
    // newv[1] = (a1 * dot) + (v1 * cosphi) + (-a3 * v2 + a2 * v3) * sinphi;
    // newv[2] = (a2 * dot) + (v2 * cosphi) + (+a3 * v1 - a1 * v3) * sinphi;
    // newv[3] = (a3 * dot) + (v3 * cosphi) + (-a2 * v1 + a1 * v2) * sinphi;

    return newv;
}

/* make provided vector a unit vector */
__device__ CUDAREAL unitize(CUDAREAL * vector, CUDAREAL * new_unit_vector) {

    CUDAREAL v1 = vector[1];
    CUDAREAL v2 = vector[2];
    CUDAREAL v3 = vector[3];
    //	CUDAREAL mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

    //  CUDAREAL mag = norm3d(v1, v2, v3);
    CUDAREAL mag = norm3d_fma_rn(v1, v2, v3);

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

/* vector cross product where vector magnitude is 0th element */
__device__ CUDAREAL *cross_product(const CUDAREAL * x, const CUDAREAL * y,
CUDAREAL * z) {
    z[1] = x[2] * y[3] - x[3] * y[2];
    z[2] = x[3] * y[1] - x[1] * y[3];
    z[3] = x[1] * y[2] - x[2] * y[1];
    z[0] = 0.0;

    return z;
}

/* vector inner product where vector magnitude is 0th element */
__device__ CUDAREAL dot_product(const CUDAREAL * x, const CUDAREAL * y) {
    return x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

__device__ CUDAREAL dot_product_ldg(const CUDAREAL * __restrict__ x,
CUDAREAL * y) {
    return __ldg(&x[1]) * y[1] + __ldg(&x[2]) * y[2] + __ldg(&x[3]) * y[3];
}

/* measure magnitude of provided vector */
__device__ void magnitude(CUDAREAL *vector) {

    /* measure the magnitude */
    vector[0] = sqrt(vector[1] * vector[1] + vector[2] * vector[2] + vector[3] * vector[3]);
}

/* scale magnitude of provided vector */
__device__ CUDAREAL vector_scale(CUDAREAL *vector, CUDAREAL *new_vector,
CUDAREAL scale) {

    new_vector[1] = scale * vector[1];
    new_vector[2] = scale * vector[2];
    new_vector[3] = scale * vector[3];
    magnitude(new_vector);

    return new_vector[0];
}

/* rotate a vector using a 9-element unitary matrix */
__device__ void rotate_umat_ldg(CUDAREAL * v, CUDAREAL *newv, const matrix3x3 * umat) {

    /* for convenience, assign matrix x-y coordinate */
    CUDAREAL uxx = __ldg(&umat->e00);
    CUDAREAL uxy = __ldg(&umat->e01);
    CUDAREAL uxz = __ldg(&umat->e02);
    CUDAREAL uyx = __ldg(&umat->e10);
    CUDAREAL uyy = __ldg(&umat->e11);
    CUDAREAL uyz = __ldg(&umat->e12);
    CUDAREAL uzx = __ldg(&umat->e20);
    CUDAREAL uzy = __ldg(&umat->e21);
    CUDAREAL uzz = __ldg(&umat->e22);
    CUDAREAL v1 = v[1];
    CUDAREAL v2 = v[2];
    CUDAREAL v3 = v[3];

    /* rotate the vector (x=1,y=2,z=3) */
    newv[1] = uxx * v1 + uxy * v2 + uxz * v3;
    newv[2] = uyx * v1 + uyy * v2 + uyz * v3;
    newv[3] = uzx * v1 + uzy * v2 + uzz * v3;
}

/* Fourier transform of a grating */
__device__ CUDAREAL sincg(CUDAREAL x, CUDAREAL N) {
    if (x != 0.0)
        return sin(x * N) / sin(x);

    return N;

}

/* Fourier transform of a sphere */
__device__ CUDAREAL sinc3(CUDAREAL x) {
    if (x != 0.0)
        return 3.0f * (sin(x) / x - cos(x)) / (x * x);

    return 1.0;

}

__device__ void polint(CUDAREAL *xa, CUDAREAL *ya, CUDAREAL x, CUDAREAL *y) {
    CUDAREAL x0, x1, x2, x3;
    x0 = (x - xa[1]) * (x - xa[2]) * (x - xa[3]) * ya[0] / ((xa[0] - xa[1]) * (xa[0] - xa[2]) * (xa[0] - xa[3]));
    x1 = (x - xa[0]) * (x - xa[2]) * (x - xa[3]) * ya[1] / ((xa[1] - xa[0]) * (xa[1] - xa[2]) * (xa[1] - xa[3]));
    x2 = (x - xa[0]) * (x - xa[1]) * (x - xa[3]) * ya[2] / ((xa[2] - xa[0]) * (xa[2] - xa[1]) * (xa[2] - xa[3]));
    x3 = (x - xa[0]) * (x - xa[1]) * (x - xa[2]) * ya[3] / ((xa[3] - xa[0]) * (xa[3] - xa[1]) * (xa[3] - xa[2]));
    *y = x0 + x1 + x2 + x3;
}

__device__ void polin2(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL ya[4][4],
CUDAREAL x1, CUDAREAL x2, CUDAREAL *y) {
    int j;
    CUDAREAL ymtmp[4];
    for (j = 1; j <= 4; j++) {
        polint(x2a, ya[j - 1], x2, &ymtmp[j - 1]);
    }
    polint(x1a, ymtmp, x1, y);
}

__device__ void polin3(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL *x3a,
CUDAREAL ya[4][4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL x3,
CUDAREAL *y) {
    int j;
    CUDAREAL ymtmp[4];

    for (j = 1; j <= 4; j++) {
        polin2(x2a, x3a, &ya[j - 1][0], x2, x3, &ymtmp[j - 1]);
    }
    polint(x1a, ymtmp, x1, y);
}

/* polarization factor */
__device__ CUDAREAL polarization_factor(CUDAREAL kahn_factor, const CUDAREAL * __restrict__ unitIncident, CUDAREAL *unitDiffracted,
        const CUDAREAL * __restrict__ unitAxis) {
    CUDAREAL cos2theta, cos2theta_sqr, sin2theta_sqr;
    CUDAREAL psi = 0.0f;
    CUDAREAL E_in[4], B_in[4], E_out[4], B_out[4];

    //  these are already unitized before entering this loop. Optimize this out.
    //	unitize(incident, incident);
    //	unitize(diffracted, diffracted);

    /* component of diffracted unit vector along incident beam unit vector */
    cos2theta = dot_product(unitIncident, unitDiffracted);
    cos2theta_sqr = cos2theta * cos2theta;
    sin2theta_sqr = 1.0f - cos2theta_sqr;

    if (kahn_factor != 0.0) {
        /* tricky bit here is deciding which direciton the E-vector lies in for each source
         here we assume it is closest to the "axis" defined above */

        // this is already unitized. Optimize this out.
        // unitize(unitAxis, unitAxis);
        /* cross product to get "vertical" axis that is orthogonal to the cannonical "polarization" */
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

__device__ CUDAREAL polarization_factor_ldg(CUDAREAL kahn_factor, const CUDAREAL * __restrict__ unitIncident, CUDAREAL *unitDiffracted,
        const CUDAREAL * __restrict__ unitAxis) {
    CUDAREAL cos2theta, cos2theta_sqr, sin2theta_sqr;
    CUDAREAL psi = 0.0;
    CUDAREAL E_in[4], B_in[4], E_out[4], B_out[4];

    //  these are already unitized before entering this loop. Optimize this out.
    //	unitize(incident, incident);
    //	unitize(diffracted, diffracted);

    /* component of diffracted unit vector along incident beam unit vector */
    cos2theta = dot_product_ldg(unitIncident, unitDiffracted);
    cos2theta_sqr = cos2theta * cos2theta;
    sin2theta_sqr = 1 - cos2theta_sqr;

    if (kahn_factor != 0.0) {
        /* tricky bit here is deciding which direciton the E-vector lies in for each source
         here we assume it is closest to the "axis" defined above */

        // this is already unitized. Optimize this out.
        // unitize(unitAxis, unitAxis);
        /* cross product to get "vertical" axis that is orthogonal to the cannonical "polarization" */
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

__device__ CUDAREAL norm3d_fma_rn(CUDAREAL v1, CUDAREAL v2, CUDAREAL v3) {
    CUDAREAL q_sqr = v3 * v3;
    q_sqr = __fmaf_rn(v2, v2, q_sqr);
    q_sqr = __fmaf_rn(v1, v1, q_sqr);
    return sqrt(q_sqr);
}

