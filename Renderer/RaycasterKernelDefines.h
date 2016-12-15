#ifndef __TUM3D__RAYCASTER_KERNEL_DEFINES_H__
#define __TUM3D__RAYCASTER_KERNEL_DEFINES_H__


#define RAYCASTER_ENABLE_LINEAR
#define RAYCASTER_ENABLE_CUBIC
//#define RAYCASTER_ENABLE_CATROM
//#define RAYCASTER_ENABLE_LAGRANGE4
//#define RAYCASTER_ENABLE_LAGRANGE6
//#define RAYCASTER_ENABLE_WENO4


#define RAYCASTER_RENDER_VOLUME_RT(kernel, measureSource, filter, computemode, colormode) \
	kernel<measureSource, filter, computemode, colormode> \
	<<<params.gridSize, params.blockSize>>> \
	(params.brickMinScreen, params.brickSizeScreen, params.renderTargetOffset, params.boxMin, params.boxMax, params.world2texOffset, params.world2texScale)

#define RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, computemode, colormode) \
	switch(GetMeasureSource(params.measure1)) { \
		case MEASURE_SOURCE_RAW:		RAYCASTER_RENDER_VOLUME_RT(kernel, MEASURE_SOURCE_RAW,  filter, computemode, colormode); break; \
		case MEASURE_SOURCE_HEAT_CURRENT:	RAYCASTER_RENDER_VOLUME_RT(kernel, MEASURE_SOURCE_HEAT_CURRENT, filter, computemode, colormode); break; \
		case MEASURE_SOURCE_JACOBIAN:		RAYCASTER_RENDER_VOLUME_RT(kernel, MEASURE_SOURCE_JACOBIAN, filter, computemode, colormode); break; \
	}

// for non-multiscale kernels: precomputed vs. on-the-fly measure computation, but no color mode switch
#define RAYCASTER_COMPUTE_SWITCH_RT(kernel, filter, colormode) \
	switch(params.measureComputeMode) { \
		case MEASURE_COMPUTE_ONTHEFLY : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, colormode); break; \
		default                       : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_PRECOMP_DISCARD, colormode); break; \
	}



#define RAYCASTER_RENDER_VOLUME_RT2(kernel, measureSource1, measureSource2, filter, computemode, colormode) \
	kernel<measureSource1, measureSource2, filter, computemode, colormode> \
	<<<params.gridSize, params.blockSize>>> \
	(params.brickMinScreen, params.brickSizeScreen, params.renderTargetOffset, params.boxMin, params.boxMax, params.world2texOffset, params.world2texScale)

#define RAYCASTER_MEASURE_SWITCH_RT2(kernel, filter, computemode, colormode) \
    switch(GetMeasureSource(params.measure1)) { \
    case MEASURE_SOURCE_RAW: \
        switch(GetMeasureSource(params.measure2)) { \
        case MEASURE_SOURCE_RAW: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_RAW, MEASURE_SOURCE_RAW, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_HEAT_CURRENT: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_RAW, MEASURE_SOURCE_HEAT_CURRENT, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_JACOBIAN: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_RAW, MEASURE_SOURCE_JACOBIAN, filter, computemode, colormode); break; \
		} \
        break; \
    case MEASURE_SOURCE_HEAT_CURRENT: \
        switch(GetMeasureSource(params.measure2)) { \
        case MEASURE_SOURCE_RAW: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_HEAT_CURRENT, MEASURE_SOURCE_RAW, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_HEAT_CURRENT: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_HEAT_CURRENT, MEASURE_SOURCE_HEAT_CURRENT, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_JACOBIAN: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_HEAT_CURRENT, MEASURE_SOURCE_JACOBIAN, filter, computemode, colormode); break; \
		} \
        break; \
    case MEASURE_SOURCE_JACOBIAN: \
        switch(GetMeasureSource(params.measure2)) { \
        case MEASURE_SOURCE_RAW: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_JACOBIAN, MEASURE_SOURCE_RAW, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_HEAT_CURRENT: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_JACOBIAN, MEASURE_SOURCE_HEAT_CURRENT, filter, computemode, colormode); break; \
        case MEASURE_SOURCE_JACOBIAN: RAYCASTER_RENDER_VOLUME_RT2(kernel, MEASURE_SOURCE_JACOBIAN, MEASURE_SOURCE_JACOBIAN, filter, computemode, colormode); break; \
		} \
        break; \
	}

// for multiscale kernels: color mode (vorticity alignment vs. uniform color), but no measure compute mode switch
#define RAYCASTER_COLOR_SWITCH_RT2(kernel, filter) \
	switch(params.colorMode) { \
		/*case COLOR_MODE_VORTICITY_ALIGNMENT : RAYCASTER_MEASURE_SWITCH_RT(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_VORTICITY_ALIGNMENT); break;*/ \
		default								: RAYCASTER_MEASURE_SWITCH_RT2(kernel, filter, MEASURE_COMPUTE_ONTHEFLY, COLOR_MODE_UI); break; \
	}


#endif
