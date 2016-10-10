#ifndef __TUM3D__RAYCASTER_KERNEL_DEFINES_H__
#define __TUM3D__RAYCASTER_KERNEL_DEFINES_H__


#define RAYCASTER_ENABLE_LINEAR
#define RAYCASTER_ENABLE_CUBIC
//#define RAYCASTER_ENABLE_CATROM
//#define RAYCASTER_ENABLE_LAGRANGE4
//#define RAYCASTER_ENABLE_LAGRANGE6
//#define RAYCASTER_ENABLE_WENO4


#define RAYCASTER_RENDER_VOLUME_RT(kernel, fromJac, filter, computemode, colormode) \
	kernel<fromJac, filter, computemode, colormode> \
	<<<params.gridSize, params.blockSize>>> \
	(params.brickMinScreen, params.brickSizeScreen, params.renderTargetOffset, params.boxMin, params.boxMax, params.world2texOffset, params.world2texScale)

#define RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, computemode, colormode) \
	switch(MeasureIsFromJacobian(params.measure1)) { \
		case true:  RAYCASTER_RENDER_VOLUME_RT(kernel, true,  filter, computemode, colormode); break; \
		case false: RAYCASTER_RENDER_VOLUME_RT(kernel, false, filter, computemode, colormode); break; \
	}

// for non-multiscale kernels: precomputed vs. on-the-fly measure computation, but no color mode switch
#define RAYCASTER_COMPUTE_SWITCH_RT(kernel, filter, colormode) \
	switch(params.measureComputeMode) { \
		case MEASURE_COMPUTE_ONTHEFLY : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, colormode); break; \
		default                       : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_PRECOMP_DISCARD, colormode); break; \
	}



#define RAYCASTER_RENDER_VOLUME_RT2(kernel, fromJac1, fromJac2, filter, computemode, colormode) \
	kernel<fromJac1, fromJac2, filter, computemode, colormode> \
	<<<params.gridSize, params.blockSize>>> \
	(params.brickMinScreen, params.brickSizeScreen, params.renderTargetOffset, params.boxMin, params.boxMax, params.world2texOffset, params.world2texScale)

#define RAYCASTER_MEASURE_SWITCH_RT2(kernel, filter, computemode, colormode) \
	if(MeasureIsFromJacobian(params.measure1)) { \
		if(MeasureIsFromJacobian(params.measure2)) { \
			RAYCASTER_RENDER_VOLUME_RT2(kernel, true,  true, filter, computemode, colormode); \
		} else { \
			RAYCASTER_RENDER_VOLUME_RT2(kernel, true, false, filter, computemode, colormode); \
		} \
	} else { \
		if(MeasureIsFromJacobian(params.measure2)) { \
			RAYCASTER_RENDER_VOLUME_RT2(kernel, false, true,  filter, computemode, colormode); \
		} else { \
			RAYCASTER_RENDER_VOLUME_RT2(kernel, false, false, filter, computemode, colormode); \
		} \
	}

// for multiscale kernels: color mode (vorticity alignment vs. uniform color), but no measure compute mode switch
#define RAYCASTER_COLOR_SWITCH_RT2(kernel, filter) \
	switch(params.colorMode) { \
		/*case COLOR_MODE_VORTICITY_ALIGNMENT : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_VORTICITY_ALIGNMENT); break;*/ \
		default								: RAYCASTER_MEASURE_SWITCH_RT2(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); break; \
	}


#endif
