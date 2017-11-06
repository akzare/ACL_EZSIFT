===============================================================
This C++11 code implements this source code:

https://sourceforge.net/projects/ezsift/

The original code is written by : Robert Wang (robertwgh (at) gmail.com)

It has been migrated to use ARM Computation Library (ACL) by: Armin Zare Zadeh (ali.a.zarezadeh@gmail.com)

ACL: https://github.com/ARM-software/ComputeLibrary

This version of the EZSIFT code can compute the SIFT keypoints 
detection and matching on NEON and MALI GPUs. In order to compile 
the code for the NEON Technology, only in the main.cpp file, 
these two lines must be uncommented:

#define SIFTType arm_compute::NEEZSIFT
const char *ALGNAME = "NE_";

Likewise for the Mali GPU the -DARM_COMPUTE_CL=1 compile switch 
must be one and also uncommented these lines:
#define SIFTType arm_compute::CLEZSIFT
const char *ALGNAME = "CL_";

This will automatically build the code for NEON or Mali. 
Basically, this new version of the code uses C++ templates 
to switch between these two computation technologies. This 
version measures the total elapsed time for the computation 
of the SIFT algorithm and puts it as text on the output images. 
Based on the used technology, the output images have the NE_ or CL_ 
prefixes. In both cases, only the computation of images hierarchies 
(octaves, Gaussian, difference of Gaussian, Gradient & Rotation) 
are performed either on the NEON or Mali, the rest of the computation 
is performed on the CPU.
