/*
 * nanoBraggCUDA.h
 *
 *  Created on: Jan 2, 2018
 *      Author: giles
 */

#ifndef NANOBRAGGCUDA_H_
#define NANOBRAGGCUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

void nanoBraggSpotsCUDA(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel, double pixel_size,
		double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu, double sdet_vector[4],
		double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance, double beam_vector[4],
		double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources, double *source_X,
		double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4], shapetype xtal_shape,
		double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell, double water_size, double water_F,
		double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F, int interpolate, double *** Fhkl, int h_min,
		int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls, int nopolar, double polar_vector[4],
		double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/, double * omega_sum/*out*/, int * sumn /*out*/,
		double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/, double * max_I_y /*out*/);

#ifdef __cplusplus
}
#endif

#endif /* NANOBRAGGCUDA_H_ */
