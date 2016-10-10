#ifndef __TUM3D__INTEGRATOR_KERNEL_DEFINES_H__
#define __TUM3D__INTEGRATOR_KERNEL_DEFINES_H__


// usage: appropriately #define INTEGRATE(advect, filter), then do ADVECT_SWITCH;

#define FILTER_CASE(advect, filter) \
	case filter: { INTEGRATE(advect, filter); } break
#define FILTER_SWITCH(advect) \
	switch(filterMode) { \
		FILTER_CASE(advect, TEXTURE_FILTER_LINEAR); \
		/*FILTER_CASE(advect, TEXTURE_FILTER_CUBIC);*/ \
		FILTER_CASE(advect, TEXTURE_FILTER_CATROM); \
		FILTER_CASE(advect, TEXTURE_FILTER_CATROM_STAGGERED); \
		FILTER_CASE(advect, TEXTURE_FILTER_LAGRANGE4); \
		FILTER_CASE(advect, TEXTURE_FILTER_LAGRANGE6); \
		FILTER_CASE(advect, TEXTURE_FILTER_LAGRANGE8); \
		FILTER_CASE(advect, TEXTURE_FILTER_LAGRANGE16); \
		/*FILTER_CASE(advect, TEXTURE_FILTER_WENO4);*/ \
		/*FILTER_CASE(advect, TEXTURE_ANALYTIC_DOUBLEGYRE);*/ \
	}
#define ADVECT_CASE(advect) \
	case advect: { FILTER_SWITCH(advect); } break
#define ADVECT_SWITCH \
	switch(advectMode) { \
		/*ADVECT_CASE(ADVECT_EULER);*/ \
		/*ADVECT_CASE(ADVECT_HEUN);*/ \
		/*ADVECT_CASE(ADVECT_RK3);*/ \
		/*ADVECT_CASE(ADVECT_RK4);*/ \
		/*ADVECT_CASE(ADVECT_BS32);*/ \
		/*ADVECT_CASE(ADVECT_RKF34);*/ \
		/*ADVECT_CASE(ADVECT_RKF45);*/ \
		/*ADVECT_CASE(ADVECT_RKF54);*/ \
		ADVECT_CASE(ADVECT_RK547M); \
	}
#define ADVECT_DENSE_SWITCH \
	switch(advectMode) { \
		ADVECT_CASE(ADVECT_RK547M); \
	}


#endif
