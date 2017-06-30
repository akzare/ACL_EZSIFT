/*	Copyright (c) 2013, Robert Wang, email: robertwgh (at) gmail.com
    Copyright (c) 2017, Armin Zare Zadeh, email: ali.a.zarezadeh (at) gmail.com
	All rights reserved. https://sourceforge.net/p/ezsift

	Some algorithms used in this code referred to:
	1. OpenCV: http://opencv.org/
	2. VLFeat: http://www.vlfeat.org/

	The SIFT algorithm was developed by David Lowe. More information can be found from:
	David G. Lowe, "Distinctive image features from scale-invariant keypoints," 
	International Journal of Computer Vision, 60, 2 (2004), pp. 91-110.

	Pay attention that the SIFT algorithm is patented. It is your responsibility to use the code
	in a legal way. Patent information:
	Method and apparatus for identifying scale invariant features in an image 
	and use of same for locating an object in an image	David G. Lowe, US Patent 6,711,293 
	(March 23, 2004). Provisional application filed March 8, 1999. Asignee: The University of 
	British Columbia.

	Revision history:
		September, 15, 2013: initial version.
		July 8th, 2014, re-organized source code. 
		May 18 2017: ported to run on ARM Neon Technology by using ARM Computation Library (ACL)
*/

#ifndef __ARM_COMPUTE_NEON_EZSIFT_H__
#define __ARM_COMPUTE_NEON_EZSIFT_H__

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"

#include "opencv2/opencv.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <list>

#include <cstdlib>
#include <cstring>
#include <fstream>

/****************************************
 * Constant parameters
 ***************************************/

// default number of sampled intervals per octave
static int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
static float SIFT_SIGMA = 1.6f;

// the radius of Gaussian filter kernel; 
// Gaussian filter mask will be (2*radius+1)x(2*radius+1).
// People use 2 or 3 most.
static float SIFT_GAUSSIAN_FILTER_RADIUS = 3.0f;

// default threshold on keypoint contrast |D(x)|
static float SIFT_CONTR_THR = 8.0f; //8.0f;

// default threshold on keypoint ratio of principle curvatures
static float SIFT_CURV_THR = 10.0f;

// The keypoint refinement smaller than this threshold will be discarded.
static float SIFT_KEYPOINT_SUBPiXEL_THR = 0.6f;

// double image size before pyramid construction?
static bool SIFT_IMG_DBL = true;

// assumed gaussian blur for input image
static float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static float SIFT_ORI_SIG_FCTR = 1.5f; // Can affect the orientation computation.

// determines the radius of the region used in orientation assignment
static float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR; // Can affect the orientation computation.

// orientation magnitude relative to max that results in new feature
static float SIFT_ORI_PEAK_RATIO = 0.8f;

// maximum number of orientations for each keypoint location
//static const float SIFT_ORI_MAX_ORI = 4;

// determines the size of a single descriptor orientation histogram
static float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static float SIFT_INT_DESCR_FCTR = 512.f;

// default width of descriptor histogram array
static int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static int SIFT_DESCR_HIST_BINS = 8;

// default value of the nearest-neighbour distance ratio threshold
// |DR_nearest|/|DR_2nd_nearest|<SIFT_MATCH_NNDR_THR is considered as a match.
static float SIFT_MATCH_NNDR_THR = 0.65f;

#if 0
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

/****************************************
 * Definitions
 ***************************************/
#define DEGREE_OF_DESCRIPTORS (128)
typedef struct _SiftKeypoint
{
  int octave; // octave number
  int layer;  // layer number
  float rlayer; // real number of layer number

  float r; // normalized row coordinate
  float c; // normalized col coordinate
  float scale; // normalized scale

  float ri;	//row coordinate in that layer.
  float ci;	//column coordinate in that layer.
  float layer_scale; // the scale of that layer

  float ori; // orientation in degrees.
  float mag; // magnitude

  float descriptors[DEGREE_OF_DESCRIPTORS];
} SiftKeypoint;

// Match pair structure. Use for interest point matching.
typedef struct _MatchPair
{
  int r1;
  int c1;
  int r2;
  int c2;
} MatchPair;

// *** Dump functions to get intermediate results ***
#define DUMP_OCTAVE_IMAGE			0
#define DUMP_GAUSSIAN_PYRAMID_IMAGE	0
#define DUMP_DOG_IMAGE				0


// *** Macro definition ***
// Macro definition
#define PI		3.141592653589793f
#define _2PI	6.283185307179586f
#define PI_4	0.785398163397448f
#define PI_3_4	2.356194490192345f
#define SQRT2	1.414213562373095f

/* ========================================================== */
/* determinant of matrix
 *
 * Computes determinant of matrix m, returning d
 */

#define DETERMINANT_3X3(d,m)                    \
{                               \
  d = m[0][0] * (m[1][1]*m[2][2] - m[1][2] * m[2][1]);     \
  d -= m[0][1] * (m[1][0]*m[2][2] - m[1][2] * m[2][0]);    \
  d += m[0][2] * (m[1][0]*m[2][1] - m[1][1] * m[2][0]);    \
}

/* ========================================================== */
/* compute adjoint of matrix and scale
 *
 * Computes adjoint of matrix m, scales it by s, returning a
 */

#define SCALE_ADJOINT_3X3(a,s,m)                \
{                               \
  a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1]); \
  a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2]); \
  a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0]); \
                                \
  a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2]); \
  a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0]); \
  a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1]); \
                                \
  a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1]); \
  a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2]); \
  a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]); \
}

/* ========================================================== */
/* matrix times vector */

#define MAT_DOT_VEC_3X3(p,m,v)                  \
{                               \
  p[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2];       \
  p[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2];       \
  p[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2];       \
}


namespace arm_compute
{
class ITensor;
class ICLTensor;

template <class tensor>
inline void write_ppm(tensor &inTensor, const std::string &ppm_filename)
{
  ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(&inTensor, Format::RGB888, Format::U8);
  ARM_COMPUTE_ERROR_ON(inTensor.info()->num_dimensions() > 2);

  std::ofstream fs;

  try
  {
    fs.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
    fs.open(ppm_filename, std::ios::out | std::ios::binary);

    const unsigned int width  = inTensor.info()->tensor_shape()[0];
    const unsigned int height = inTensor.info()->tensor_shape()[1];

    fs << "P6\n"
       << width << " " << height << " 255\n";

#ifdef ARM_COMPUTE_CL
  // Map buffer if creating a CLTensor
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    inTensor.map();
  }
#endif

  switch(inTensor.info()->format())
  {
    case Format::U8:
    {
      Window window;
      window.set(Window::DimX, Window::Dimension(0, width, 1));
      window.set(Window::DimY, Window::Dimension(0, height, 1));

      Iterator in(&inTensor, window);

      execute_window_loop(window, [&](const Coordinates & id)
      {
        const unsigned char value = *in.ptr();

        fs << value << value << value;
      },
      in);

      break;
    }
    case Format::RGB888:
    {
      Window window;
      window.set(Window::DimX, Window::Dimension(0, width, width));
      window.set(Window::DimY, Window::Dimension(0, height, 1));

      Iterator in(&inTensor, window);

      execute_window_loop(window, [&](const Coordinates & id)
      {
        fs.write(reinterpret_cast<std::fstream::char_type *>(in.ptr()), width * inTensor.info()->element_size());
      },
      in);

      break;
    }
    default:
    ARM_COMPUTE_ERROR("Unsupported format");
  }

#ifdef ARM_COMPUTE_CL
  // Unmap buffer if creating a CLTensor
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    inTensor.unmap();
  }
#endif
  }
  catch(const std::ofstream::failure &e)
  {
    ARM_COMPUTE_ERROR("Writing %s: (%s)", ppm_filename.c_str(), e.what());
  }

}

/** Basic function to run NEMagnitudePhaseKernel */
class NEGradRot : public INESimpleFunction
{
public:
  /** Initialise the kernel's inputs.
   *
   * @param[in]  input1      First tensor input. Data type supported: S16.
   * @param[in]  input2      Second tensor input. Data type supported: S16.
   * @param[out] outputMag   Output tensor. Data type supported: S16.
   * @param[out] outputPhase Output tensor. Data type supported: U8.
   * @param[in]  use_fp16    (Optional) If true the FP16 kernels will be used. If false F32 kernels are used.
   */
  void configure(const ITensor *input1, const ITensor *input2, ITensor *outputMag, ITensor *outputPhase, bool use_fp16 = false);
};

/** Basic function to run @ref CLMagnitudePhaseKernel. */
class CLGradRot : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input1      First tensor input. Data types supported: S16.
     * @param[in]  input2      Second tensor input. Data types supported: S16.
     * @param[out] outputMag   Output tensor. Data type supported: S16.
     * @param[out] outputPhase Output tensor. Data type supported: U8.
     * @param[in]  mag_type    (Optional) Magnitude calculation type. Default: L2NORM.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *outputMag, ICLTensor *outputPhase, MagnitudeType mag_type = MagnitudeType::L2NORM);
};

// Helper callback function for merge match list.
bool same_match_pair (MatchPair first, MatchPair second);

// Match keypoints from two images, using brutal force method.
// Use Euclidean distance as matching score.
void match_keypoints(std::list<SiftKeypoint> & kpt_list1,
	                 std::list<SiftKeypoint> & kpt_list2,
	                 std::list<MatchPair> & match_list);

/** Basic implementation of the pyramid interface */
template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
class EZSIFT
{
public:
  // Initialize pyramid data-object using the given ne_ezsift's metadata
  void init(const cv::Mat & inimg);
  // Get a specific octave image
  tensor *get_octave(size_t index) const;

  // Dump all the constructed octave images into files
  void dump_octave_image() const;

  // Get a specific Gaussian image at the constructed pyramid
  tensor *get_gaussian_pyramid(size_t index_octave, size_t index_gpyr) const;
  // Dump all the constructed Gaussian pyramid images into files
  void dump_gaussian_pyramid_image() const;
  // Dump the computed Gaussian coefficients
  void dump_gaussian_coefs() const;
  // Dump the computed Difference of Gaussian coefficients
  void dump_dog_pyramid_image() const;
  // Get a specific Difference of Gaussian image at the constructed pyramid
  tensor *get_dog_pyramid(size_t index_octave, size_t index_gpyr) const;
  // Get a specific Gradient of Gaussian image at the constructed pyramid
  tensor *get_grd_pyramid(size_t index_octave, size_t index_gpyr) const;
  // Get a specific Rotation of Gaussian image at the constructed pyramid
  tensor *get_rot_pyramid(size_t index_octave, size_t index_gpyr) const;

  // Compute the EZSIFT on the NEON
//  template <class T>
  int sift(std::list<SiftKeypoint> & kpt_list);

  void write_ppm(tensor &inTensor, const std::string &ppm_filename);

  void scratch_pad();

private:
  // *** Helper functions ***
  inline float my_log2(float n){
    // Visual C++ does not have log2...
    return (float) ((log(n))/0.69314718055994530941723212145818);
  }

  // Build octaves
  void build_octaves();
  // Build Gaussian pyramid
  void build_gaussian_pyramid();
  // Compute Gaussian coefficients
  void compute_gaussian_coefs();
  // Get the specific Gaussian coefficients
  void get_gaussian_coefs(size_t index_gpyr, int16_t *gaussianFunWin) const;
  // Build DoG pyramid
  void build_dog_pyr();
  // Build Gradient/Rotation pyramid
  void build_grd_rot_pyr();

  // Refine local keypoint extrema
  bool refine_local_extrema(SiftKeypoint & kpt);

  // Compute orientation histogram for keypoint detection.
  // using pre-computed gradient information.
  float compute_orientation_hist_with_gradient(
  	SiftKeypoint & kpt,
  	float * & hist);

  // Detect keypoints
  void detect_keypoints(std::list<SiftKeypoint> & kpt_list);

  // Extract descriptor
  void extract_descriptor(std::list<SiftKeypoint> & kpt_list);

  // Input image tensor
  tensor _input_img;

  //Index of the first octave.
  int _firstOctave;
  //Number of layers in one octave; same as s in the paper.
  int _nLayers;
  //Number of Gaussian images in one octave.
  int _nGpyrLayers;
  //Number of DoG images in one octave.
  int _nDogLayers;
  //Number of octaves according to the size of image.
  int _nOctaves;

  // Octaves tensors
  std::unique_ptr<tensor[]> _octaves{ nullptr };

  // Gaussian pyramid tensors
  std::unique_ptr<tensor[]> _gpyr{ nullptr };
  std::unique_ptr<int16_t[]> _gaussian_coefs{ nullptr };

  // DoG pyramid tensors
  std::unique_ptr<tensor[]> _dogPyr{ nullptr };

  // Gardient and Rotation pyramid tensors
  std::unique_ptr<tensor[]> _grdPyr{ nullptr };
  std::unique_ptr<tensor[]> _rotPyr{ nullptr };

};

/** Interface for the kernel which applied a 3x3 convolution to a tensor.*/
#ifndef ARM_COMPUTE_CL
using NEEZSIFT = EZSIFT<Tensor,NEScale,NEConvolution7x7,7,NEAbsoluteDifference,NEDerivative,NEGradRot>;
#endif
#ifdef ARM_COMPUTE_CL
using CLEZSIFT = EZSIFT<CLTensor,CLScale,CLConvolution7x7,7,CLAbsoluteDifference,CLDerivative,CLGradRot>;
#endif

}

#endif /*__ARM_COMPUTE_NEON_EZSIFT_H__ */
