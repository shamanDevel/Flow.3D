Be aware:

- Make sure that the project is set to compile cuda code with your exact GPU compute capability. If it is not configured correctly, every time the application is recompiled intermediate cuda code will have to be translated to machine code, effectively halting the first time initialization. To configure that, go to 'Project Properties -> CUDA C/C++ -> Devie -> Code Generation' and modify the compute version to match your graphics card (ex: for a GTX 970 it should be 'compute_52,sm_52' and for a GTX 1070 it should be 'compute_61,sm_61'). You can also define multiple cuda compute capability versions at the same time (this is in fact the ideal). However, bear in mind that build times will increase as the number of targets increases as well.

Known issues:

- Can't load timevol from external or network drives.
- On some machines, startup after recompiling can take 5-10 minutes. Cause unclear, possibly related to the on-board graphics chip (which is a pretty wild guess though).
- Loading settings before the .timevol causes all lines to be rendered with the background colour.

Syntax for preprocessor:

    Preprocessor.exe --json .\_datainfo.json --overwrite --channels velx:1,vely:1,velz:1,temp:1 -- outfilename

Individual fields from `_datainfo.json` can be overridden with explicit command-line flags.





TODO:

make 4k volumes

Tracer Features:
- store vel max per brick
- Periodic tracing?
- CatRom in time?
- Tracing progress
- inconsistent results: disk wait bailout -> budget reset
- Interpolated (dense) output also for other integrators?
- Flow graph

Tracer Efficiency:
- loading: separate thread?
- don't shift time slots around (ringbuffer!)
- start threads only for active lines?
- sort checkpoints by locality?
- cudaCompress: multiBlock DWT?
- cudaCompress: put quantMap into cudaArray
- cudaCompress: merge small cudaMemcpys
- BuildIndexBuffer?

flow graph: always round up?

line normals: store as scalar?
simple lines (pos+time only)


Renderer:
fix near plane clipping
timing: track I/O waiting time
per-brick timings: optionally only for full bricks
clip plane
non-compressed: parallel upload+render?
Preintegration
MIDA kernel? tf importance channel?
float render target?
Stereo: per-eye brick sorting
filtering: optimize x kernel, proper boundaries (wrap)
render modes: DVR + Iso/SIVR? Two-sided iso?
precomputed measure volume: automatically choose quant step

Multi-GPU:
fix tracing + multi-GPU rendering
multi-GPU images not equal to single-GPU?
multi-GPU filtering

Compress:
streams? (-> cudpp)
