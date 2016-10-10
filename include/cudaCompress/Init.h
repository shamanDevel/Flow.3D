#ifndef __TUM3D_CUDACOMPRESS__INIT_H__
#define __TUM3D_CUDACOMPRESS__INIT_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

class Instance;

CUCOMP_DLL Instance* createInstance(int cudaDevice, uint blockCountMax, uint elemCountPerBlockMax, uint offsetIntervalMin, uint log2HuffmanDistinctSymbolCountMax = 0);
CUCOMP_DLL void  destroyInstance(Instance* pInstance);

CUCOMP_DLL int getInstanceCudaDevice(const Instance* pInstance);
CUCOMP_DLL uint getInstanceBlockCountMax(const Instance* pInstance);
CUCOMP_DLL uint getInstanceElemCountPerBlockMax(const Instance* pInstance);
CUCOMP_DLL uint getInstanceOffsetIntervalMin(const Instance* pInstance);
CUCOMP_DLL uint getInstanceLog2HuffmanDistinctSymbolCountMax(const Instance* pInstance);
CUCOMP_DLL bool getInstanceUseLongSymbols(const Instance* pInstance);

}


#endif
