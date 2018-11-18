Compiling
---------

- This solution should be compiled using **Visual Studio 2013** and **Cuda Toolkit 10**;
- Make sure that the project is set to compile cuda code with your exact GPU compute capability. If not configured correctly, every time the application has to be recompiled, intermediate cuda code will have to be translated into machine code, effectively halting the first initialization of the tool for a few minutes. You can change the target cuda capability by going to **Project Properties -> CUDA C/C++ -> Devie -> Code Generation** and modifying the compute version to match your graphics card's e.g., for a GTX 970 and a GTX 1070 it should be set to **compute_52,sm_52** and **compute_61,sm_61**, respectively. You can also define multiple cuda compute capability versions at the same time (this is in fact the ideal approach). However, bear in mind that build times and final binary size will increase as the number of targets increases;
- The only dependencies are **[cudaCompress](https://github.com/m0bl0/cudaCompress)** and **[FX11](https://github.com/Microsoft/FX11)**. **cudaCompress** is included in the solution and both Preprocessor and Renderer projects are set to look for the built **cudaCompres.lib** in the target directory. The advantage of this setup is that the project can be easily upgraded to new Visual Studio versions without having to clone **cudaCompress** to generate the a new updated lib. Ideally the FX11 project should be included in the solution as well and both **cudaCompress** and **FX11** would be git submodules.


Known issues
------------

- Can't load timevol from external or network drives;
- Orthographic size does not accounts for the viewport size - the current default values where chosen so that the viewport exactly fits the domain on a top-down view direction;
- Raycasting does not work with orthographic projection;
- Seed texture is inverted in one axis;
- Tracing progress is not being properly updated;

The following functionalities were not ported while replacing AntTweakBar with **[ImGui](https://github.com/ocornut/imgui)**:
- Screenshots;
- Command line arguments;
- Batch tracing and image sequence.

The following functionalities were not ported while including support for multiple dataset tracing:
- Muti GPU tracing has not been tested and most likely does not work anymore;
- Heatmap;
- Flowgraph (it would be required one per dataset);
- Save and load of tracing and rendering settings;
- Save and load of transfer functions;
- Save and load of lines;
- Raycasting transfer function;
- Slice rendering;
- Seed texture;
- Save timevol datasets as `.raw` and `.la3d`;
- Stereo rendering and CPU tracing are not guaranteed to work - CPU tracing was only implemented for stream lines since the beginning anyways and stereo rendering should not be present in this tool in the first place.

The code for almost all previously mentioned functionalities is still present in the project - usually surrounded by `#ifdef` preprocessor - and should be easily reintegrated into the tool. In order to check how things were implemented before one can checkout to the tag `1.0v` (this version requires Visual Studio 2013 and Cuda Toolkit 8).

Usage
-----

Syntax for preprocessor:

    Preprocessor.exe --json .\_datainfo.json --overwrite --channels velx:1,vely:1,velz:1,temp:1 -- outfilename

Individual fields from `_datainfo.json` can be overridden with explicit command-line flags. If you have a single file with all three velocity components interleaved, you could instead use `--channels vel:3,temp:1` for example. You can also override individual values from the JSON. For example you could additionally pass in `--tmax 10` if you don't want to compress all timesteps you have.

This will generate a compressed version of the data (in the directory pointed to by `outPath`). You can load the `.timevol` file from this output into the visualisation tool. For our Rayleigh-Bénard data, you'll also want to have images of the domain segmentation ready a) using unique colours for each cell and b) using easily distinguishable colours (e.g. a 4-colouring). The domain-segmenation.nb Mathematica notebook generates these. You can load the texture with unique colours (and ideally thickened borders) as a Seed Texture, and the texture with distinguishable colours as both the Color Texture and Slice Texture.



TODO
----

- Replace tum3D math library with **[glm](https://github.com/g-truc/glm)**. It is actively maintained, supports SIMD instructions, works within Cuda kernels and is easier to use;
- Thee current version of **[ImGui](https://github.com/ocornut/imgui)** in use was taken from the unstable docking branch. It should be updated as soon as docking is merged into master;
- Update **[FX11](https://github.com/Microsoft/FX11)** to the lastest release;
- Upgrade to Visual Studio 2017 and Cuda Toolkit 10 - [it seems that compatibilty issues have been solved with Cuda 10](https://blogs.msdn.microsoft.com/vcblog/2018/10/01/cuda-10-is-now-available-with-support-for-the-latest-visual-studio-2017-versions/);
- Either update stb_image to the lastest version or use ImGui's copy;
- Improve memory management code. GPUResources and CompressVolumResources are duplicated within different classes;
- `g_texVolume1` cuda texture is shared between Raycaster and Integrator. Create one reference for each;
- Allow filtering and raycasting of multiple datasets;
- Particles are currently sorted within each tracing instance. All particles should be merged and sorted together.
- Bring back missing functionalties mentioned in `Known Issues` section :)

TODO (Legacy)
----

- make 4k volumes

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

Flow graph: 
- always round up?

Line normals: 
- store as scalar?
 -simple lines (pos+time only)


Renderer:
- fix near plane clipping
- timing: track I/O waiting time
- per-brick timings: optionally only for full bricks
- clip plane
- non-compressed: parallel upload+render?
- Preintegration
- MIDA kernel? tf importance channel?
- float render target?
- Stereo: per-eye brick sorting
- filtering: optimize x kernel, proper boundaries (wrap)
- render modes: DVR + Iso/SIVR? Two-sided iso?
- precomputed measure volume: automatically choose quant step

Multi-GPU:
- fix tracing + multi-GPU rendering
- multi-GPU images not equal to single-GPU?
- multi-GPU filtering

Compress:
- streams? (-> cudpp)
