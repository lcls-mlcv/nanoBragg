/*
 ============================================================================
 Name        : nanoBraggCPU.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdlib.h>
#include <math.h>
#include "nanotypes.h"
#include "nanoBraggCPU.h"

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
		double * max_I_y /*out*/) {

	int total_pixels = spixels * fpixels;
	double * omega_reduction = (double*) calloc(total_pixels, sizeof(double));
	int spixel, fpixel;
	for (spixel = 0; spixel < spixels; ++spixel) { // Slow pixels (Y)
		for (fpixel = 0; fpixel < fpixels; ++fpixel) { // Fast pixels (X)
			/* allow for just one part of detector to be rendered */
			if (fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) { //ROI region of interest
				continue;
			}

			/* position in pixel array */
			int j = spixel * fpixels + fpixel;

			/* allow for the use of a mask */
			if (maskimage != NULL) {
				/* skip any flagged pixels in the mask */
				if (maskimage[j] == 0) {
					continue;
				}
			}

			/* reset photon count for this pixel */
			double I = 0.0;
			double polar = 0.0;
			double omega_pixel = 0.0;
			double Fdet = 0.0;
			double Sdet = 0.0;

			/* add background from something amorphous */
			double F_bg = water_F;
			double I_bg = F_bg * F_bg * r_e_sqr * fluence * polar * water_size * water_size * water_size * 1e6 * Avogadro / water_MW * omega_pixel;

			/* add this now to avoid problems with skipping later */
			floatimage[j] = I_bg;

			/* loop over sub-pixels */
			int subS, subF;
			for (subS = 0; subS < oversample; ++subS) { // Y voxel
				for (subF = 0; subF < oversample; ++subF) { // X voxel
					/* absolute mm position on detector (relative to its origin) */
					Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0; // X voxel
					Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0; // Y voxel
//                  Fdet = pixel_size*fpixel;
//                  Sdet = pixel_size*spixel;

					int thick_tic;
					for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic) {
						/* assume "distance" is to the front of the detector sensor layer */
						double Odet = thick_tic * detector_thickstep; // Z Orthagonal voxel.

						/* construct detector subpixel position in 3D space */
//                      pixel_X = distance;
//                      pixel_Y = Sdet-Ybeam;
//                      pixel_Z = Fdet-Xbeam;
						double pixel_pos[] = { 0, 0, 0, 0 };
						pixel_pos[1] = Fdet * fdet_vector[1] + Sdet * sdet_vector[1] + Odet * odet_vector[1] + pix0_vector[1]; // X_
						pixel_pos[2] = Fdet * fdet_vector[2] + Sdet * sdet_vector[2] + Odet * odet_vector[2] + pix0_vector[2]; // Y
						pixel_pos[3] = Fdet * fdet_vector[3] + Sdet * sdet_vector[3] + Odet * odet_vector[3] + pix0_vector[3]; // Z
						pixel_pos[0] = 0.0; // Magnitiude of vector
						if (curved_detector) {
							/* construct detector pixel that is always "distance" from the sample */
							double dbvector[] = { 0, 0, 0, 0 };
							dbvector[1] = distance * beam_vector[1];
							dbvector[2] = distance * beam_vector[2];
							dbvector[3] = distance * beam_vector[3];
							/* treat detector pixel coordinates as radians */
							double newvector[] = { 0.0, 0.0, 0.0, 0.0 };
							rotate_axis(dbvector, newvector, sdet_vector, pixel_pos[2] / distance);
							rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3] / distance);
//                          rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
						}
						/* construct the diffracted-beam unit vector to this sub-pixel */
						double diffracted[4] = { 0.0, 0.0, 0.0, 0.0 };
						double airpath = unitize(pixel_pos, diffracted);

						/* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
						omega_pixel = pixel_size * pixel_size / airpath / airpath * close_distance / airpath; //JAMES. Serial?
						/* option to turn off obliquity effect, inverse-square-law only */
						if (point_pixel) {
							omega_pixel = 1.0 / airpath / airpath;
						}
						omega_reduction[j] = omega_pixel; // shared contention

						/* now calculate detector thickness effects */
						double capture_fraction = 1.0;
						if (detector_thick > 0.0) {
							/* inverse of effective thickness increase */
							double parallax = dot_product(diffracted, odet_vector);
							capture_fraction = exp(-thick_tic * detector_thickstep * detector_mu / parallax)
									- exp(-(thick_tic + 1) * detector_thickstep * detector_mu / parallax);
						}

						/* loop over sources now */
						int source;
						for (source = 0; source < sources; ++source) {

							/* retrieve stuff from cache */
							double incident[] = { 0.0, 0.0, 0.0, 0.0 };
							incident[1] = -source_X[source];
							incident[2] = -source_Y[source];
							incident[3] = -source_Z[source];
							double lambda = source_lambda[source];

							/* construct the incident beam unit vector while recovering source distance */
							double source_path = unitize(incident, incident);

							/* construct the scattering vector for this pixel */
							double scattering[] = { 0.0, 0.0, 0.0, 0.0 };
							scattering[1] = (diffracted[1] - incident[1]) / lambda;
							scattering[2] = (diffracted[2] - incident[2]) / lambda;
							scattering[3] = (diffracted[3] - incident[3]) / lambda;

							/* sin(theta)/lambda is half the scattering vector length */
							double stol = 0.5 * magnitude(scattering);

							/* rough cut to speed things up when we aren't using whole detector */
							if (dmin > 0.0 && stol > 0.0) {
								if (dmin > 0.5 / stol) {
									continue;
								}
							}

							/* sweep over phi angles */
							int phi_tic;
							for (phi_tic = 0; phi_tic < phisteps; ++phi_tic) {
								double phi = phi0 + phistep * phi_tic;

								double ap[] = { 0.0, 0.0, 0.0, 0.0 };
								double bp[] = { 0.0, 0.0, 0.0, 0.0 };
								double cp[] = { 0.0, 0.0, 0.0, 0.0 };

								/* rotate about spindle if necessary */
								rotate_axis(a0, ap, spindle_vector, phi);
								rotate_axis(b0, bp, spindle_vector, phi);
								rotate_axis(c0, cp, spindle_vector, phi);

								/* enumerate mosaic domains */
								int mos_tic;
								for (mos_tic = 0; mos_tic < mosaic_domains; ++mos_tic) {
									/* apply mosaic rotation after phi rotation */
									double a[] = { 0.0, 0.0, 0.0, 0.0 };
									double b[] = { 0.0, 0.0, 0.0, 0.0 };
									double c[] = { 0.0, 0.0, 0.0, 0.0 };
									if (mosaic_spread > 0.0) {
										rotate_umat(ap, a, &mosaic_umats[mos_tic * 9]);
										rotate_umat(bp, b, &mosaic_umats[mos_tic * 9]);
										rotate_umat(cp, c, &mosaic_umats[mos_tic * 9]);
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
									double h = dot_product(a, scattering);
									double k = dot_product(b, scattering);
									double l = dot_product(c, scattering);

									/* round off to nearest whole index */
									int h0 = ceil(h - 0.5);
									int k0 = ceil(k - 0.5);
									int l0 = ceil(l - 0.5);

									/* structure factor of the lattice (paralelpiped crystal)
									 F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
									 */
									double F_latt = 1.0; // Shape transform for the crystal.
									double hrad_sqr = 0.0;
									if (xtal_shape == SQUARE) {
										/* xtal is a paralelpiped */
										if (Na > 1) {
											F_latt *= sincg(M_PI * h, Na);
										}
										if (Nb > 1) {
											F_latt *= sincg(M_PI * k, Nb);
										}
										if (Nc > 1) {
											F_latt *= sincg(M_PI * l, Nc);
										}
									} else {
										/* handy radius in reciprocal space, squared */
										hrad_sqr = (h - h0) * (h - h0) * Na * Na + (k - k0) * (k - k0) * Nb * Nb + (l - l0) * (l - l0) * Nc * Nc;
									}
									if (xtal_shape == ROUND) {
										/* use sinc3 for elliptical xtal shape,
										 correcting for sqrt of volume ratio between cube and sphere */
										F_latt = Na * Nb * Nc * 0.723601254558268 * sinc3(M_PI * sqrt(hrad_sqr * fudge));
									}
									if (xtal_shape == GAUSS) {
										/* fudge the radius so that volume and FWHM are similar to square_xtal spots */
										F_latt = Na * Nb * Nc * exp(-(hrad_sqr / 0.63 * fudge));
									}
									if (xtal_shape == TOPHAT) {
										/* make a flat-top spot of same height and volume as square_xtal spots */
										F_latt = Na * Nb * Nc * (hrad_sqr * fudge < 0.3969);
									}
									/* no need to go further if result will be zero? */
									if (F_latt == 0.0 && water_size == 0.0)
										continue;

									/* find nearest point on Ewald sphere surface? */
									if (integral_form) {

										/* need to calculate reciprocal matrix */
										/* various cross products */
										double a_cross_b[] = { 0.0, 0.0, 0.0, 0.0 };
										double b_cross_c[] = { 0.0, 0.0, 0.0, 0.0 };
										double c_cross_a[] = { 0.0, 0.0, 0.0, 0.0 };
										cross_product(a, b, a_cross_b);
										cross_product(b, c, b_cross_c);
										cross_product(c, a, c_cross_a);

										/* new reciprocal-space cell vectors */
										double a_star[] = { 0.0, 0.0, 0.0, 0.0 };
										double b_star[] = { 0.0, 0.0, 0.0, 0.0 };
										double c_star[] = { 0.0, 0.0, 0.0, 0.0 };
										vector_scale(b_cross_c, a_star, 1e20 / V_cell);
										vector_scale(c_cross_a, b_star, 1e20 / V_cell);
										vector_scale(a_cross_b, c_star, 1e20 / V_cell);

										/* reciprocal-space coordinates of nearest relp */
										double relp[] = { 0.0, 0.0, 0.0, 0.0 };
										relp[1] = h0 * a_star[1] + k0 * b_star[1] + l0 * c_star[1];
										relp[2] = h0 * a_star[2] + k0 * b_star[2] + l0 * c_star[2];
										relp[3] = h0 * a_star[3] + k0 * b_star[3] + l0 * c_star[3];
//                                      d_star = magnitude(relp)

										/* reciprocal-space coordinates of center of Ewald sphere */
										double Ewald0[] = { 0.0, 0.0, 0.0, 0.0 };
										Ewald0[1] = -incident[1] / lambda / 1e10;
										Ewald0[2] = -incident[2] / lambda / 1e10;
										Ewald0[3] = -incident[3] / lambda / 1e10;
//                                      1/lambda = magnitude(Ewald0)

										/* distance from Ewald sphere in lambda=1 units */
										double dEwald0[] = { 0.0, 0.0, 0.0, 0.0 };
										dEwald0[1] = relp[1] - Ewald0[1];
										dEwald0[2] = relp[2] - Ewald0[2];
										dEwald0[3] = relp[3] - Ewald0[3];
										double d_r = magnitude(dEwald0) - 1.0;

										/* unit vector of diffracted ray through relp */
										double diffracted0[] = { 0.0, 0.0, 0.0, 0.0 };
										unitize(dEwald0, diffracted0);

										/* intersection with detector plane */
										double xd = dot_product(fdet_vector, diffracted0);
										double yd = dot_product(sdet_vector, diffracted0);
										double zd = dot_product(odet_vector, diffracted0);

										/* where does the central direct-beam hit */
										double xd0 = dot_product(fdet_vector, incident);
										double yd0 = dot_product(sdet_vector, incident);
										double zd0 = dot_product(odet_vector, incident);

										/* convert to mm coordinates */
										double Fdet0 = distance * (xd / zd) + Xbeam;
										double Sdet0 = distance * (yd / zd) + Ybeam;

										//printf("GOTHERE %g %g   %g %g\n",Fdet,Sdet,Fdet0,Sdet0);
										double test = exp(-((Fdet - Fdet0) * (Fdet - Fdet0) + (Sdet - Sdet0) * (Sdet - Sdet0) + d_r * d_r) / 1e-8);
									} // end of integral form

									/* structure factor of the unit cell */
									int h0_flr = 0, k0_flr = 0, l0_flr = 0;
									double F_cell = default_F;
									if (interpolate) {
										h0_flr = floor(h);
										k0_flr = floor(k);
										l0_flr = floor(l);

										if (((h - h_min + 3) > h_range) || (h - 2 < h_min) || ((k - k_min + 3) > k_range) || (k - 2 < k_min)
												|| ((l - l_min + 3) > l_range) || (l - 2 < l_min)) {
//											if (babble) {
//												babble = 0;
//												printf("WARNING: out of range for three point interpolation: h,k,l,h0,k0,l0: %g,%g,%g,%d,%d,%d \n", h, k, l, h0,
//														k0, l0);
//												printf("WARNING: further warnings will not be printed! ");
//											}
											interpolate = 0;
										}

										/* only interpolate if it is safe */
										if (interpolate) {
											/* integer versions of nearest HKL indicies */
											int h_interp[] = { 0.0, 0.0, 0.0, 0.0 };
											int k_interp[] = { 0.0, 0.0, 0.0, 0.0 };
											int l_interp[] = { 0.0, 0.0, 0.0, 0.0 };
											h_interp[0] = h0_flr - 1;
											h_interp[1] = h0_flr;
											h_interp[2] = h0_flr + 1;
											h_interp[3] = h0_flr + 2;
											k_interp[0] = k0_flr - 1;
											k_interp[1] = k0_flr;
											k_interp[2] = k0_flr + 1;
											k_interp[3] = k0_flr + 2;
											l_interp[0] = l0_flr - 1;
											l_interp[1] = l0_flr;
											l_interp[2] = l0_flr + 1;
											l_interp[3] = l0_flr + 2;

											/* polin function needs doubles */
											double h_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
											double k_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
											double l_interp_d[] = { 0.0, 0.0, 0.0, 0.0 };
											h_interp_d[0] = (double) h_interp[0];
											h_interp_d[1] = (double) h_interp[1];
											h_interp_d[2] = (double) h_interp[2];
											h_interp_d[3] = (double) h_interp[3];
											k_interp_d[0] = (double) k_interp[0];
											k_interp_d[1] = (double) k_interp[1];
											k_interp_d[2] = (double) k_interp[2];
											k_interp_d[3] = (double) k_interp[3];
											l_interp_d[0] = (double) l_interp[0];
											l_interp_d[1] = (double) l_interp[1];
											l_interp_d[2] = (double) l_interp[2];
											l_interp_d[3] = (double) l_interp[3];

											/* now populate the "y" values (nearest four structure factors in each direction) */
											double sub_Fhkl[4][4][4];
											int i1, i2, i3;
											for (i1 = 0; i1 < 4; i1++) {
												for (i2 = 0; i2 < 4; i2++) {
													for (i3 = 0; i3 < 4; i3++) {
														sub_Fhkl[i1][i2][i3] = Fhkl[h_interp[i1] - h_min][k_interp[i2] - k_min][l_interp[i3] - l_min];
													}
												}
											}

											/* run the tricubic polynomial interpolation */
											polin3(h_interp_d, k_interp_d, l_interp_d, sub_Fhkl, h, k, l, &F_cell);
										}
									}
									if (!interpolate) {
										if (hkls && (h0 <= h_max) && (h0 >= h_min) && (k0 <= k_max) && (k0 >= k_min) && (l0 <= l_max) && (l0 >= l_min)) {
											/* just take nearest-neighbor */
											F_cell = Fhkl[(h0 - h_min)][k0 - k_min][l0 - l_min];
										} else {
											F_cell = default_F;  // usually zero
										}
									}

									/* now we have the structure factor for this pixel */

									/* polarization factor */
									if (!nopolar) {
										/* need to compute polarization factor */
										polar = polarization_factor(polarization, incident, diffracted, polar_vector);
									} else {
										polar = 1.0;
									}

									/* convert amplitudes into intensity (photons per steradian) */
									I += F_cell * F_cell * F_latt * F_latt * capture_fraction;
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

			floatimage[j] += r_e_sqr * fluence * polar * I / steps * omega_pixel;
//          floatimage[j] = test;
			if (floatimage[j] > (double) *max_I) {
				*max_I = floatimage[j];
				*max_I_x = Fdet;
				*max_I_y = Sdet;
			}
			*sum += floatimage[j];
			*sumsqr += floatimage[j] * floatimage[j];
			++(*sumn);

#if 0
			if (printout) {
				if ((fpixel == printout_fpixel && spixel == printout_spixel) || printout_fpixel < 0) {
					twotheta = atan2(sqrt(pixel_pos[2] * pixel_pos[2] + pixel_pos[3] * pixel_pos[3]), pixel_pos[1]);
					test = sin(twotheta / 2.0) / (lambda0 * 1e10);
					printf("%4d %4d : stol = %g or %g\n", fpixel, spixel, stol, test);
					printf("at %g %g %g\n", pixel_pos[1], pixel_pos[2], pixel_pos[3]);
					printf("hkl= %f %f %f  hkl0= %d %d %d\n", h, k, l, h0, k0, l0);
					printf(" F_cell=%g  F_latt=%g   I = %g\n", F_cell, F_latt, I);
					printf("I/steps %15.10g\n", I / steps);
					printf("polar   %15.10g\n", polar);
					printf("omega   %15.10g\n", omega_pixel);
					printf("capfrac %15.10g\n", capture_fraction);
					printf("pixel   %15.10g\n", floatimage[j]);
					printf("real-space cell vectors (Angstrom):\n");
					printf("     %-10s  %-10s  %-10s\n", "a", "b", "c");
					printf("X: %11.8f %11.8f %11.8f\n", a[1] * 1e10, b[1] * 1e10, c[1] * 1e10);
					printf("Y: %11.8f %11.8f %11.8f\n", a[2] * 1e10, b[2] * 1e10, c[2] * 1e10);
					printf("Z: %11.8f %11.8f %11.8f\n", a[3] * 1e10, b[3] * 1e10, c[3] * 1e10);
				}
			} else {
				if (progress_meter && progress_pixels / 100 > 0) {
					if (progress_pixel % (progress_pixels / 20) == 0
							|| ((10 * progress_pixel < progress_pixels || 10 * progress_pixel > 9 * progress_pixels)
									&& (progress_pixel % (progress_pixels / 100) == 0))) {
						printf("%lu%% done\n", progress_pixel * 100 / progress_pixels);
					}
				}
			}
			++progress_pixel;
#endif
		}
	}
	*omega_sum = 0;
	int i;
	for (i = 0; i < total_pixels; i++) {
		*omega_sum += omega_reduction[i];
	}
	free(omega_reduction);
}

/* rotate a point about a unit vector axis */
extern double *rotate_axis(double *v, double *newv, double *axis, double phi) {

	double sinphi = sin(phi);
	double cosphi = cos(phi);
	double dot = (axis[1] * v[1] + axis[2] * v[2] + axis[3] * v[3]) * (1.0 - cosphi);
	double temp[4];

	temp[1] = axis[1] * dot + v[1] * cosphi + (-axis[3] * v[2] + axis[2] * v[3]) * sinphi;
	temp[2] = axis[2] * dot + v[2] * cosphi + (+axis[3] * v[1] - axis[1] * v[3]) * sinphi;
	temp[3] = axis[3] * dot + v[3] * cosphi + (-axis[2] * v[1] + axis[1] * v[2]) * sinphi;
	newv[1] = temp[1];
	newv[2] = temp[2];
	newv[3] = temp[3];

	return newv;
}

/* make provided vector a unit vector */
extern double unitize(double *vector, double *new_unit_vector) {
	double mag;

	/* measure the magnitude */
	mag = magnitude(vector);

	if (mag != 0.0) {
		/* normalize it */
		new_unit_vector[1] = vector[1] / mag;
		new_unit_vector[2] = vector[2] / mag;
		new_unit_vector[3] = vector[3] / mag;
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
extern double *cross_product(double *x, double *y, double *z) {
	z[1] = x[2] * y[3] - x[3] * y[2];
	z[2] = x[3] * y[1] - x[1] * y[3];
	z[3] = x[1] * y[2] - x[2] * y[1];
	z[0] = 0.0;

	return z;
}

/* vector inner product where vector magnitude is 0th element */
extern double dot_product(double *x, double *y) {
	return x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

/* measure magnitude of provided vector */
extern double magnitude(double *vector) {

	/* measure the magnitude */
	vector[0] = sqrt(vector[1] * vector[1] + vector[2] * vector[2] + vector[3] * vector[3]);

	return vector[0];
}

/* scale magnitude of provided vector */
extern double vector_scale(double *vector, double *new_vector, double scale) {

	new_vector[1] = scale * vector[1];
	new_vector[2] = scale * vector[2];
	new_vector[3] = scale * vector[3];

	return magnitude(new_vector);
}

/* enforce magnitude of provided vector */
extern double vector_rescale(double *vector, double *new_vector, double new_magnitude) {
	double oldmag;

	oldmag = magnitude(vector);
	if (oldmag <= 0.0)
		oldmag = 1.0;
	new_vector[1] = new_magnitude / oldmag * vector[1];
	new_vector[2] = new_magnitude / oldmag * vector[2];
	new_vector[3] = new_magnitude / oldmag * vector[3];

	return magnitude(new_vector);
}

/* difference between two given vectors */
extern double vector_diff(double *vector, double *origin_vector, double *new_vector) {

	new_vector[1] = vector[1] - origin_vector[1];
	new_vector[2] = vector[2] - origin_vector[2];
	new_vector[3] = vector[3] - origin_vector[3];
	return magnitude(new_vector);
}

/* rotate a vector using a 9-element unitary matrix */
extern double *rotate_umat(double *v, double *newv, double umat[9]) {

	double uxx, uxy, uxz, uyx, uyy, uyz, uzx, uzy, uzz;

	/* for convenience, assign matrix x-y coordinate */
	uxx = umat[0];
	uxy = umat[1];
	uxz = umat[2];
	uyx = umat[3];
	uyy = umat[4];
	uyz = umat[5];
	uzx = umat[6];
	uzy = umat[7];
	uzz = umat[8];

	/* rotate the vector (x=1,y=2,z=3) */
	newv[1] = uxx * v[1] + uxy * v[2] + uxz * v[3];
	newv[2] = uyx * v[1] + uyy * v[2] + uyz * v[3];
	newv[3] = uzx * v[1] + uzy * v[2] + uzz * v[3];

	return newv;
}

/* Fourier transform of a grating */
extern double sincg(double x, double N) {
	if (x == 0.0)
		return N;

	return sin(x * N) / sin(x);
}

/* Fourier transform of a sphere */
extern double sinc3(double x) {
	if (x == 0.0)
		return 1.0;

	return 3.0 * (sin(x) / x - cos(x)) / (x * x);
}

/* Fourier transform of a spherically-truncated lattice */
extern double sinc_conv_sinc3(double x) {
	if (x == 0.0)
		return 1.0;

	return 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
}

extern void polint(double *xa, double *ya, double x, double *y) {
	double x0, x1, x2, x3;
	x0 = (x - xa[1]) * (x - xa[2]) * (x - xa[3]) * ya[0] / ((xa[0] - xa[1]) * (xa[0] - xa[2]) * (xa[0] - xa[3]));
	x1 = (x - xa[0]) * (x - xa[2]) * (x - xa[3]) * ya[1] / ((xa[1] - xa[0]) * (xa[1] - xa[2]) * (xa[1] - xa[3]));
	x2 = (x - xa[0]) * (x - xa[1]) * (x - xa[3]) * ya[2] / ((xa[2] - xa[0]) * (xa[2] - xa[1]) * (xa[2] - xa[3]));
	x3 = (x - xa[0]) * (x - xa[1]) * (x - xa[2]) * ya[3] / ((xa[3] - xa[0]) * (xa[3] - xa[1]) * (xa[3] - xa[2]));
	*y = x0 + x1 + x2 + x3;
}

extern void polin2(double *x1a, double *x2a, double ya[4][4], double x1, double x2, double *y) {
	int j;
	double ymtmp[4];
	for (j = 1; j <= 4; j++) {
		polint(x2a, ya[j - 1], x2, &ymtmp[j - 1]);
	}
	polint(x1a, ymtmp, x1, y);
}

extern void polin3(double *x1a, double *x2a, double *x3a, double ya[4][4][4], double x1, double x2, double x3, double *y) {
	int j;
	double ymtmp[4];

	for (j = 1; j <= 4; j++) {
		polin2(x2a, x3a, &ya[j - 1][0], x2, x3, &ymtmp[j - 1]);
	}
	polint(x1a, ymtmp, x1, y);
}

/* polarization factor */
extern double polarization_factor(double kahn_factor, double *incident, double *diffracted, double *axis) {
	double cos2theta, cos2theta_sqr, sin2theta_sqr;
	double psi;
	double E_in[4];
	double B_in[4];
	double E_out[4];
	double B_out[4];

	unitize(incident, incident);
	unitize(diffracted, diffracted);
	unitize(axis, axis);

	/* component of diffracted unit vector along incident beam unit vector */
	cos2theta = dot_product(incident, diffracted);
	cos2theta_sqr = cos2theta * cos2theta;
	sin2theta_sqr = 1 - cos2theta_sqr;

	if (kahn_factor != 0.0) {
		/* tricky bit here is deciding which direciton the E-vector lies in for each source
		 here we assume it is closest to the "axis" defined above */

		/* cross product to get "vertical" axis that is orthogonal to the cannonical "polarization" */
		cross_product(axis, incident, B_in);
		/* make it a unit vector */
		unitize(B_in, B_in);

		/* cross product with incident beam to get E-vector direction */
		cross_product(incident, B_in, E_in);
		/* make it a unit vector */
		unitize(E_in, E_in);

		/* get components of diffracted ray projected onto the E-B plane */
		E_out[0] = dot_product(diffracted, E_in);
		B_out[0] = dot_product(diffracted, B_in);

		/* compute the angle of the diffracted ray projected onto the incident E-B plane */
		psi = -atan2(B_out[0], E_out[0]);
	}

	/* correction for polarized incident beam */
	return 0.5 * (1.0 + cos2theta_sqr - kahn_factor * cos(2 * psi) * sin2theta_sqr);
}

extern double *rotate(double *v, double *newv, double phix, double phiy, double phiz) {

	double rxx, rxy, rxz, ryx, ryy, ryz, rzx, rzy, rzz;
	double new_x, new_y, new_z, rotated_x, rotated_y, rotated_z;

	new_x = v[1];
	new_y = v[2];
	new_z = v[3];

	if (phix != 0) {
		/* rotate around x axis */
		//rxx= 1;         rxy= 0;         rxz= 0;
		ryx = 0;
		ryy = cos(phix);
		ryz = -sin(phix);
		rzx = 0;
		rzy = sin(phix);
		rzz = cos(phix);

		rotated_x = new_x;
		rotated_y = new_y * ryy + new_z * ryz;
		rotated_z = new_y * rzy + new_z * rzz;
		new_x = rotated_x;
		new_y = rotated_y;
		new_z = rotated_z;
	}

	if (phiy != 0) {
		/* rotate around y axis */
		rxx = cos(phiy);
		rxy = 0;
		rxz = sin(phiy);
		//ryx= 0;         ryy= 1;         ryz= 0;
		rzx = -sin(phiy);
		rzy = 0;
		rzz = cos(phiy);

		rotated_x = new_x * rxx + new_y * rxy + new_z * rxz;
		rotated_y = new_y;
		rotated_z = new_x * rzx + new_y * rzy + new_z * rzz;
		new_x = rotated_x;
		new_y = rotated_y;
		new_z = rotated_z;
	}

	if (phiz != 0) {
		/* rotate around z axis */
		rxx = cos(phiz);
		rxy = -sin(phiz);
		rxz = 0;
		ryx = sin(phiz);
		ryy = cos(phiz);
		ryz = 0;
		//rzx= 0;         rzy= 0;         rzz= 1;

		rotated_x = new_x * rxx + new_y * rxy;
		rotated_y = new_x * ryx + new_y * ryy;
		rotated_z = new_z;
		new_x = rotated_x;
		new_y = rotated_y;
		new_z = rotated_z;
	}

	newv[1] = new_x;
	newv[2] = new_y;
	newv[3] = new_z;

	return newv;
}

