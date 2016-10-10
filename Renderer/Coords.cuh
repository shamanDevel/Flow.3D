#ifndef __TUM3D__COORDS_CUH__
#define __TUM3D__COORDS_CUH__



/******************************************************************************
** World to tex space conversion
******************************************************************************/

#define w2t(val) (((val) + world2texOffset) * world2texScale)
#define w2tx(val) (((val) + world2texOffset.x) * world2texScale)
#define w2ty(val) (((val) + world2texOffset.y) * world2texScale)
#define w2tz(val) (((val) + world2texOffset.z) * world2texScale)

#define t2w(val) (((val) / world2texScale) - world2texOffset)


#define time2tex(val) (((val) + time2texOffset) * time2texScale)


#endif
