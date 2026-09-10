/*
 * nanoBraggCUDA.h
 *
 *  Created on: Jan 2, 2018
 *      Author: giles
 */

#ifndef NANOBRAGGCPU_H_
#define NANOBRAGGCPU_H_

/* cubic spline interpolation functions */
extern void polint(double *xa, double *ya, double x, double *y);
extern void polin2(double *x1a, double *x2a, double ya[4][4], double x1, double x2, double *y);
extern void polin3(double *x1a, double *x2a, double *x3a, double ya[4][4][4], double x1, double x2, double x3, double *y);
/* rotate a 3-vector in space applied in order phix,phiy,phiz*/
double *rotate(double *v, double *newv, double phix, double phiy, double phiz);
/* rotate a 3-vector about a unit vector axis */
extern double *rotate_axis(double *v, double *newv, double *axis, double phi);
/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
extern double unitize(double *vector, double *new_unit_vector);
/* vector cross product where vector magnitude is 0th element */
extern double *cross_product(double *x, double *y, double *z);
/* vector inner product where vector magnitude is 0th element */
extern double dot_product(double *x, double *y);
/* measure magnitude of vector and put it in 0th element */
extern double magnitude(double *vector);
/* scale the magnitude of a vector */
extern double vector_scale(double *vector, double *new_vector, double scale);
/* compute difference between two vectors */
double vector_diff(double *vector, double *origin_vector, double *new_vector);
/* force the magnitude of vector to given value */
double vector_rescale(double *vector, double *new_vector, double magnitude);
/* rotate a 3-vector using a 9-element unitary matrix */
extern double *rotate_umat(double *v, double *newv, double *umat);
/* Fourier transform of a truncated lattice */
extern double sincg(double x, double N);
/* Fourier transform of a sphere */
extern double sinc3(double x);
/* Fourier transform of a spherically-truncated lattice */
double sinc_conv_sinc3(double x);
/* polarization factor from vectors */
extern double polarization_factor(double kahn_factor, double *incident, double *diffracted, double *axis);

extern void nanoBraggSpotsCPU(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
		double pixel_size, double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu,
		double sdet_vector[4], double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance,
		double beam_vector[4], double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources,
		double *source_X, double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4],
		shapetype xtal_shape, double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell,
		double water_size, double water_F, double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F,
		int interpolate, double *** Fhkl, int h_min, int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls,
		int nopolar, double polar_vector[4], double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/,
		double * omega_sum/*out*/, int * sumn /*out*/, double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/,
		double * max_I_y /*out*/);

#endif /* NANOBRAGGCPU_H_ */
