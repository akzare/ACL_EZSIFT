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
		July 8th, 2014: fixed a bug in sample_2x in image.h. The bug only happened for image with odd width or height.
		May 18 2017: ported to run on ARM Neon Technology by using ARM Computation Library (ACL)
*/


#include "acl_ezsift.h"

#include "opencv2/opencv.hpp"

#include "arm_neon.h"

#include <string>
#include <queue>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <fstream>
#include <iostream>
#include <cmath>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NEConvolutionKernel.h"
#include "arm_compute/core/CL/kernels/CLConvolutionKernel.h"
#include "arm_compute/core/NEON/kernels/NEMagnitudePhaseKernel.h"
#include "arm_compute/core/CL/kernels/CLMagnitudePhaseKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"

#include "test_helpers/Utils.h"

using namespace arm_compute;


void NEGradRot::configure(const ITensor *input1, const ITensor *input2, ITensor *outputMag, ITensor *outputPhase, bool use_fp16)
{
  if(use_fp16){
    auto k = arm_compute::cpp14::make_unique<NEMagnitudePhaseFP16Kernel<MagnitudeType::L2NORM, PhaseType::SIGNED>>();
    k->configure(input1, input2, outputMag, outputPhase);
    _kernel = std::move(k);
  }
  else{
    auto k = arm_compute::cpp14::make_unique<NEMagnitudePhaseKernel<MagnitudeType::L2NORM, PhaseType::SIGNED>>();
    k->configure(input1, input2, outputMag, outputPhase);
    _kernel = std::move(k);
  }
}

void CLGradRot::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *outputMag, ICLTensor *outputPhase, MagnitudeType mag_type)
{
  auto k = arm_compute::cpp14::make_unique<CLMagnitudePhaseKernel>();
  k->configure(input1, input2, outputMag, outputPhase, mag_type);
  _kernel = std::move(k);
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::scratch_pad()
{

////////////////////////////////////////////////////////
//  tensor _tst_in_img, _tst_out_img;
//  _tst_in_img.allocator()->init(TensorInfo(5, 5, Format::U8));
//  _tst_in_img.allocator()->allocate();
//  _tst_out_img.allocator()->init(TensorInfo(5, 5, Format::U8));
//  _tst_out_img.allocator()->allocate();
//
//
//  std::cout << "Test:" << std::endl;
//  Window tst_window;
//  tst_window.use_tensor_dimensions(_tst_in_img.info());
//  Iterator tst_it(&_tst_in_img, tst_window);
//  execute_window_loop(tst_window, [&](const Coordinates & id)
//  {
//    *reinterpret_cast<unsigned char *>(tst_it.ptr()) = 'a' + id.x() + id.y()*5;
//    std::cout << "(" << id.y() << "," << id.x() << ")=" << *reinterpret_cast<unsigned char *>(tst_it.ptr()) << "," ;
//    if (id.x() == 4) std::cout << std::endl;
//  },
//  tst_it);
//  std::cout << std::endl;
//
////  Iterator input(_input, win);
//  const unsigned char *_tst_lft_top_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(-1, -1));
//  const unsigned char *_tst_lft_mid_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(-1, 0));
//  const unsigned char *_tst_lft_low_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(-1, 1));
//
//  const unsigned char *_tst_mid_top_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(0, -1));
//  const unsigned char *_tst_mid_mid_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(0, 0));
//  const unsigned char *_tst_mid_low_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(0, 1));
//
//  const unsigned char *_tst_rht_top_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(1, -1));
//  const unsigned char *_tst_rht_mid_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(1, 0));
//  const unsigned char *_tst_rht_low_ptr = _tst_in_img.buffer() + _tst_in_img.info()->offset_element_in_bytes(Coordinates(1, 1));
//
//  // Configure kernel window
//  constexpr unsigned int num_elems_processed_per_iteration = 1;
//  constexpr unsigned int num_elems_read_per_iteration      = 1;
//  constexpr unsigned int num_elems_written_per_iteration   = 1;
//#define BRD 1
//#define BRD_DEF true
//  Window                 win = calculate_max_window(*_tst_in_img.info(), Steps(num_elems_processed_per_iteration), BRD_DEF/*border_undefined*/, BorderSize(BRD));
//  AccessWindowHorizontal output_access(_tst_out_img.info(), 0, num_elems_written_per_iteration);
//
//  update_window_and_padding(win,
//                            AccessWindowRectangle(_tst_in_img.info(), -BorderSize(BRD).left, -BorderSize(BRD).top, num_elems_read_per_iteration, 1),
//                            output_access);
//
//  output_access.set_valid_region(win, _tst_in_img.info()->valid_region(), BRD_DEF/*border_undefined*/, BorderSize(BRD));
//
//  Iterator innn(&_tst_in_img, win);
//  Iterator outtt(&_tst_out_img, win);
//  execute_window_loop(win, [&](const Coordinates & id)
//  {
//	  std::cout << "lft_top:" << *reinterpret_cast<const unsigned char *>(_tst_lft_top_ptr + innn.offset()) << " mid_top:" << *reinterpret_cast<const unsigned char *>(_tst_mid_top_ptr + innn.offset()) << " rht_top:" << *reinterpret_cast<const unsigned char *>(_tst_rht_top_ptr + innn.offset()) << std::endl;
//	  std::cout << "lft_mid:" << *reinterpret_cast<const unsigned char *>(_tst_lft_mid_ptr + innn.offset()) << " mid_mid:" << *reinterpret_cast<const unsigned char *>(_tst_mid_mid_ptr + innn.offset()) << " rht_mid:" << *reinterpret_cast<const unsigned char *>(_tst_rht_mid_ptr + innn.offset()) << std::endl;
//	  std::cout << "lft_low:" << *reinterpret_cast<const unsigned char *>(_tst_lft_low_ptr + innn.offset()) << " mid_low:" << *reinterpret_cast<const unsigned char *>(_tst_mid_low_ptr + innn.offset()) << " rht_low:" << *reinterpret_cast<const unsigned char *>(_tst_rht_low_ptr + innn.offset()) << std::endl;
//	  std::cout << std::endl;
//  },
//  innn,outtt);

#ifdef ARM_COMPUTE_CL
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    CLKernelLibrary::get().init("/home/odroid/acl/ComputeLibrary-master/src/core/CL/cl_kernels/", cl::Context::getDefault(), cl::Device::getDefault());
    CLScheduler::get().init(cl::Context::getDefault(), cl::CommandQueue::getDefault());
//    CLScheduler::get().default_init();
  }
#endif

  scale _scale_octave;

  tensor _tst_in_img;
  _tst_in_img.allocator()->init(TensorInfo(640, 480, Format::U8));
  _tst_in_img.allocator()->allocate();
  tensor _tst_out_img;
  _tst_out_img.allocator()->init(TensorInfo(640*2, 480*2, Format::U8));
  _tst_out_img.allocator()->allocate();

  _scale_octave.configure(&_tst_in_img, &_tst_out_img, InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED);

  _scale_octave.run();
  CLScheduler::get().sync();

  const std::string output_filename = "test.ppm";
//  arm_compute::write_ppm<tensor>(*(_octaves.get() + 0), output_filename);
  arm_compute::write_ppm<tensor>(_tst_out_img, output_filename);

  return;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::init(const cv::Mat & inimg)
{
  cv::Size s = inimg.size();
  unsigned int srcW = s.width, srcH = s.height;

  // Index of the first octave.
  _firstOctave = (SIFT_IMG_DBL)? -1 : 0;
  // Number of layers in one octave; same as s in the paper.
  _nLayers = SIFT_INTVLS;
  // Number of Gaussian images in one octave.
  _nGpyrLayers = _nLayers + 3;
  // Number of DoG images in one octave.
  _nDogLayers = _nLayers + 2;
  // Number of octaves according to the size of image.
  _nOctaves = (int) my_log2((float)std::min(srcW, srcH)) - 3 - _firstOctave -1; // 2 or 3, need further research

  ////////////////////////////////////////////////////////////////////////
  // Init OpenCL
  ////////////////////////////////////////////////////////////////////////
  // Init OpenCL if creating a CLTensor
#ifdef ARM_COMPUTE_CL
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    CLKernelLibrary::get().init("/home/odroid/acl/ComputeLibrary-master/src/core/CL/cl_kernels/", cl::Context::getDefault(), cl::Device::getDefault());
    CLScheduler::get().init(cl::Context::getDefault(), cl::CommandQueue::getDefault());
  }
#endif

  ////////////////////////////////////////////////////////////////////////
  // Construct and allocate the input image tensor
  ////////////////////////////////////////////////////////////////////////
  // Initialize the tensor dimensions and type:
  _input_img.allocator()->init(TensorInfo(srcW, srcH, Format::U8));

  // Allocate the input tensor:
  _input_img.allocator()->allocate();

#ifdef ARM_COMPUTE_CL
  // Map buffer if creating a CLTensor
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    _input_img.map();
  }
#endif

  // Fill the input tensor:
  // Simplest way: create an iterator to iterate through each element of the input tensor:
  Window input_window;
  input_window.use_tensor_dimensions(_input_img.info());

  // Create an iterator for the input image:
  Iterator input_it(&_input_img, input_window);

  // Iterate through the elements of src_data and copy them one by one to the input tensor:
  uchar* src_data = inimg.data;
  execute_window_loop(input_window, [&](const Coordinates & id)
  {
    *reinterpret_cast<unsigned char *>(input_it.ptr()) = src_data[id.y() * srcW + id.x()];
  },
  input_it);
#ifdef ARM_COMPUTE_CL
  // Unmap buffer if creating a CLTensor
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    _input_img.unmap();
  }
#endif

  ////////////////////////////////////////////////////////////////////////
  // Construct and allocate the octave tensors
  ////////////////////////////////////////////////////////////////////////
  // Octave tensors
  _octaves = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves);

  // Initialize the first octave width and height (up-sample x2):
  unsigned int dstW = srcW << 1, dstH = srcH << 1;

  // Initialize the tensor dimensions and type:
  for(size_t i = 0; i < _nOctaves; i++){
    // Initialize the tensor dimensions and type for this octave:
    get_octave(i)->allocator()->init(TensorInfo(dstW, dstH, Format::U8));

    // Allocate the octave tensor:
    get_octave(i)->allocator()->allocate();

    // Initialize the next octave width and height (down-sample x2):
    dstW = dstW >> 1;
    dstH = dstH >> 1;
  }

  ////////////////////////////////////////////////////////////////////////
  // Construct and allocate the Gaussian pyramid tensors
  ////////////////////////////////////////////////////////////////////////
  _gaussian_coefs = arm_compute::cpp14::make_unique<int16_t[]>(conv_matrix_size * conv_matrix_size * _nGpyrLayers);
  compute_gaussian_coefs();

//  dump_gaussian_coefs();

  _gpyr = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nGpyrLayers);

  // Initialize the first octave width and height (up-sample x2):
  dstW = srcW << 1, dstH = srcH << 1;

  // Initialize the tensor dimensions and type:
  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nGpyrLayers; j++){
      // Initialize the tensor dimensions and type for this octave:
      get_gaussian_pyramid(i,j)->allocator()->init(TensorInfo(dstW, dstH, Format::U8));

      // Allocate the Gaussian pyramid tensor:
      get_gaussian_pyramid(i,j)->allocator()->allocate();

    }
    // Initialize the next octave width and height (down-sample x2):
    dstW = dstW >> 1;
    dstH = dstH >> 1;
  }

  ////////////////////////////////////////////////////////////////////////
  // Construct and allocate the Difference of Gaussian pyramid tensors
  ////////////////////////////////////////////////////////////////////////
  _dogPyr = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nDogLayers);
  // Initialize the first octave width and height (up-sample x2):
  dstW = srcW << 1, dstH = srcH << 1;

  // Initialize the tensor dimensions and type:
  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nDogLayers; j++){
      // Initialize the tensor dimensions and type for this octave:
      get_dog_pyramid(i,j)->allocator()->init(TensorInfo(dstW, dstH, Format::U8));

      // Allocate the Difference of Gaussian pyramid tensor:
      get_dog_pyramid(i,j)->allocator()->allocate();

    }
    // Initialize the next octave width and height (down-sample x2):
    dstW = dstW >> 1;
    dstH = dstH >> 1;
  }

  ////////////////////////////////////////////////////////////////////////
  // Construct and allocate the gradient and rotation pyramid tensors
  ////////////////////////////////////////////////////////////////////////
  _grdPyr = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nGpyrLayers);
  _rotPyr = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nGpyrLayers);
  // Initialize the first octave width and height (up-sample x2):
  dstW = srcW << 1, dstH = srcH << 1;

  // Initialize the tensor dimensions and type:
  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nGpyrLayers; j++){
      // Initialize the tensor dimensions and type for this octave:
      get_grd_pyramid(i,j)->allocator()->init(TensorInfo(dstW, dstH, Format::S16));

      // Allocate the Gaussian pyramid tensor:
      get_grd_pyramid(i,j)->allocator()->allocate();

      // Initialize the tensor dimensions and type for this octave:
      get_rot_pyramid(i,j)->allocator()->init(TensorInfo(dstW, dstH, Format::U8));

      // Allocate the Gaussian pyramid tensor:
      get_rot_pyramid(i,j)->allocator()->allocate();

    }
    // Initialize the next octave width and height (down-sample x2):
    dstW = dstW >> 1;
    dstH = dstH >> 1;
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
tensor *EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_octave(size_t index) const
{
  ARM_COMPUTE_ERROR_ON(index >= _nOctaves);

  return (_octaves.get() + index);
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::dump_octave_image() const
{
  ARM_COMPUTE_ERROR_ON(_octaves == nullptr);

  for(size_t i = 0; i < _nOctaves; ++i){
	tensor *tempTensor = get_octave(i);
    const std::string output_filename = "octave_Octave-" + std::to_string(i) + ".ppm";
    arm_compute::write_ppm<tensor>(*tempTensor, output_filename);
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::build_octaves()
{
  // Octaves functions
  std::unique_ptr<scale[]> _scale_octave{ nullptr };

  // Octave scale
  _scale_octave = arm_compute::cpp14::make_unique<scale[]>(_nOctaves);

  // Configure scale on all octaves:
  for(unsigned int i = 0; i < _nOctaves; i++){
    // Configure horizontal kernel
    _scale_octave[i].configure(&_input_img, get_octave(i), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED);
  }

  // Run scale on all octaves:
  for(unsigned int i = 0; i < _nOctaves; i++){
    (_scale_octave.get() + i)->run();
//    NEScheduler::get().multithread(_scale_octave.get() + i);
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
tensor *EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_gaussian_pyramid(size_t index_octave, size_t index_gpyr) const
{
  ARM_COMPUTE_ERROR_ON(index_octave >= _nOctaves);
  ARM_COMPUTE_ERROR_ON(index_gpyr >= _nGpyrLayers);

  return ((_gpyr.get() + index_octave*_nGpyrLayers+index_gpyr));
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::dump_gaussian_pyramid_image() const
{
  ARM_COMPUTE_ERROR_ON(_gpyr == nullptr);

  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nGpyrLayers; j++){
      tensor *tempTensor = get_gaussian_pyramid(i,j);
      const std::string output_filename = "gpyr-" + std::to_string(i) + "-" + std::to_string(j) + ".ppm";
      arm_compute::write_ppm<tensor>(*tempTensor, output_filename);
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_gaussian_coefs(size_t index_gpyr, int16_t *gaussianFun) const
{
  ARM_COMPUTE_ERROR_ON(index_gpyr >= _nGpyrLayers);

  std::copy_n((_gaussian_coefs.get() + index_gpyr*conv_matrix_size*conv_matrix_size), conv_matrix_size*conv_matrix_size, gaussianFun);
  return;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::dump_gaussian_coefs() const
{
  ARM_COMPUTE_ERROR_ON(_gaussian_coefs == nullptr);

  for (int i = 0; i < _nGpyrLayers; i++){
    std::cout << "GpyrLayer:" << i << std::endl;
    for (int j = 0; j < conv_matrix_size; j++){
      for (int k = 0; k < conv_matrix_size; k++){
    	std::cout <<  *(_gaussian_coefs.get() + i*conv_matrix_size*conv_matrix_size + j*conv_matrix_size + k) << "," ;
      }
      std::cout << std::endl;
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::compute_gaussian_coefs()
{
  // Compute all sigmas for different layers
  int nLayers = _nGpyrLayers - 3;
  float sigma, sigma_pre;
  float sigma0 = SIFT_SIGMA;
  float k = powf(2.0f, 1.0f / nLayers);

  std::vector<float> sig(_nGpyrLayers);
  sigma_pre = SIFT_IMG_DBL? 2.0f * SIFT_INIT_SIGMA : SIFT_INIT_SIGMA;
  sig[0] = sqrtf(sigma0 * sigma0 - sigma_pre * sigma_pre);
  for (int i = 1; i < _nGpyrLayers; i ++){
    sigma_pre = powf(k, (float)(i - 1)) * sigma0;
    sigma = sigma_pre * k;
    sig[i] = sqrtf(sigma * sigma - sigma_pre * sigma_pre);
  }

  for (int i = 0; i < _nGpyrLayers; i++){
    // Compute Gaussian filter coefficients
    float factor = SIFT_GAUSSIAN_FILTER_RADIUS;
    int gR = (sig[i] * factor > 1.0f)? (int)ceilf(sig[i] * factor): 1;
    int gW = gR * 2 + 1;
    int l = 0;

    for(int j = 0; j < gW; j++){
      if (j >= ((gW-1)/2-(conv_matrix_size/2)) && j <= ((gW-1)/2+(conv_matrix_size/2))){
        for(int k = 0; k < gW; k ++){
   	      float tmp1 = sqrtf(j * j + k * k);
          float tmp = (float)(((j - gR) * (j - gR) + (k - gR) * (k - gR)) / (sig[i] * sig[i]));
          float tmp2 = expf(-0.5f * tmp) * (1 + tmp1/1000.0f);
          if (k >= ((gW-1)/2-(conv_matrix_size/2)) && k <= ((gW-1)/2+(conv_matrix_size/2))){
            *(_gaussian_coefs.get() + i*conv_matrix_size*conv_matrix_size + l) = static_cast<int16_t>(std::floor(tmp2*10.));
            l++;
          }
        }
      }
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::build_gaussian_pyramid()
{
  // Gaussian pyramid functions
  std::unique_ptr<conv[]> _conv_gpyr{ nullptr };
  std::unique_ptr<scale[]> _scale_gpyr{ nullptr };

  // Gaussian pyramid conv
  _conv_gpyr = arm_compute::cpp14::make_unique<conv[]>(_nOctaves * _nGpyrLayers);
  _scale_gpyr = arm_compute::cpp14::make_unique<scale[]>(_nOctaves);

  // Configure Gaussian Convolution kernels on all Gaussian pyramid:
  for(unsigned int i = 0; i < _nOctaves; i++){
	for (unsigned int j = 0; j < _nGpyrLayers; j++){
      int16_t gaussianFun[conv_matrix_size*conv_matrix_size];
      get_gaussian_coefs(j, gaussianFun);
      // Configure convolution kernel
	  if (i == 0 && j == 0){
        _conv_gpyr[i*_nGpyrLayers+j].configure(get_octave(i), get_gaussian_pyramid(i,j), gaussianFun, 0 /* Let arm_compute calculate the scale */, BorderMode::UNDEFINED);
	  } else if (i > 0 && j == 0){
		_scale_gpyr[i-1].configure(get_gaussian_pyramid(i-1,_nLayers), get_gaussian_pyramid(i,j), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED); // BILINEAR NEAREST_NEIGHBOR
	  } else {
        _conv_gpyr[i*_nGpyrLayers+j].configure(get_gaussian_pyramid(i,j-1), get_gaussian_pyramid(i,j), gaussianFun, 0 /* Let arm_compute calculate the scale */, BorderMode::UNDEFINED);
	  }
    }
  }

  // Run Gaussian Convolution kernels on all Gaussian pyramid:
  for(unsigned int i = 0; i < _nOctaves; i++){
	for (unsigned int j = 0; j < _nGpyrLayers; j++){
	  if (i == 0 && j == 0){
        (_conv_gpyr.get() + i*_nGpyrLayers+j)->run();
	  } else if (i > 0 && j == 0){
        (_scale_gpyr.get() + (i-1))->run();
	  } else {
        (_conv_gpyr.get() + i*_nGpyrLayers+j)->run();
	  }
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::dump_dog_pyramid_image() const
{
  ARM_COMPUTE_ERROR_ON(_dogPyr == nullptr);

  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nDogLayers; j++){
      tensor *tempTensor = get_dog_pyramid(i,j);
      const std::string output_filename = "dog_Octave-" + std::to_string(i) + "_Layer-" + std::to_string(j) + ".ppm";
      arm_compute::write_ppm<tensor>(*tempTensor, output_filename);
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
tensor *EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_dog_pyramid(size_t index_octave, size_t index_gpyr) const
{
  ARM_COMPUTE_ERROR_ON(index_octave >= _nOctaves);
  ARM_COMPUTE_ERROR_ON(index_gpyr >= _nDogLayers);

  return ((_dogPyr.get() + index_octave*_nDogLayers+index_gpyr));
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::build_dog_pyr()
{
  // Arithmetic subtract functions
  std::unique_ptr<absdif[]> _arithsub_dog_pyr{ nullptr };

  // DoG pyramid arithmetic subtract
  _arithsub_dog_pyr = arm_compute::cpp14::make_unique<absdif[]>(_nOctaves * _nDogLayers);

  // Configure Subtraction kernels on all DoG pyramid:
  for(unsigned int i = 0; i < _nOctaves; i++){
	for (unsigned int j = 0; j < _nDogLayers; j++){
	  _arithsub_dog_pyr[i*_nDogLayers+j].configure(get_gaussian_pyramid(i,j+1), get_gaussian_pyramid(i,j), get_dog_pyramid(i,j));
    }
  }

  // Run Subtraction kernels on all DoG pyramid:
  for(unsigned int i = 0; i < _nOctaves; i++){
    for (unsigned int  j = 0; j < _nDogLayers; j++){
      (_arithsub_dog_pyr.get() + i*_nDogLayers+j)->run();
    }
  }
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
tensor *EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_grd_pyramid(size_t index_octave, size_t index_gpyr) const
{
  ARM_COMPUTE_ERROR_ON(index_octave >= _nOctaves);
  ARM_COMPUTE_ERROR_ON(index_gpyr >= _nGpyrLayers);

  return ((_grdPyr.get() + index_octave*_nGpyrLayers+index_gpyr));
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
tensor *EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::get_rot_pyramid(size_t index_octave, size_t index_gpyr) const
{
  ARM_COMPUTE_ERROR_ON(index_octave >= _nOctaves);
  ARM_COMPUTE_ERROR_ON(index_gpyr >= _nGpyrLayers);

  return ((_rotPyr.get() + index_octave*_nGpyrLayers+index_gpyr));
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::build_grd_rot_pyr()
{
  std::unique_ptr<tensor[]> _dx{ nullptr };
  std::unique_ptr<tensor[]> _dy{ nullptr };

  _dx = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nGpyrLayers);
  _dy = arm_compute::cpp14::make_unique<tensor[]>(_nOctaves * _nGpyrLayers);
  // Initialize the first octave width and height (up-sample x2):
  int dstW = get_gaussian_pyramid(0,0)->info()->dimension(0), dstH = get_gaussian_pyramid(0,0)->info()->dimension(1);

  // Initialize the tensor dimensions and type:
  for (int i = 0; i < _nOctaves; i++){
    for (int j = 0; j < _nGpyrLayers; j++){
      // Initialize the tensor dimensions and type for this octave:
      (_dx.get() + i*_nGpyrLayers+j)->allocator()->init(TensorInfo(dstW, dstH, Format::S16));

      // Allocate the Gaussian pyramid tensor:
      (_dx.get() + i*_nGpyrLayers+j)->allocator()->allocate();

      // Initialize the tensor dimensions and type for this octave:
      (_dy.get() + i*_nGpyrLayers+j)->allocator()->init(TensorInfo(dstW, dstH, Format::S16));

      // Allocate the Gaussian pyramid tensor:
      (_dy.get() + i*_nGpyrLayers+j)->allocator()->allocate();

    }
    // Initialize the next octave width and height (down-sample x2):
    dstW = dstW >> 1;
    dstH = dstH >> 1;
  }

  // Derivative functions
  std::unique_ptr<deriv[]> _driv_pyr{ nullptr };
  // Derivative, x and y pyramid
  _driv_pyr = arm_compute::cpp14::make_unique<deriv[]>(_nOctaves * _nGpyrLayers);

  // Magnitude/Phase functions
  std::unique_ptr<gradrot[]> _mag_phase_pyr{ nullptr };
  // Magnitude/Phase pyramid
  _mag_phase_pyr = arm_compute::cpp14::make_unique<gradrot[]>(_nOctaves * _nGpyrLayers);

  for(unsigned int i = 0; i < _nOctaves; i++){
	for (unsigned int j = 0; j < _nGpyrLayers; j++){
      // Configure derivative kernels:
      _driv_pyr[i*_nGpyrLayers+j].configure(get_gaussian_pyramid(i,j), (_dx.get() + i*_nGpyrLayers+j), (_dy.get() + i*_nGpyrLayers+j), BorderMode::UNDEFINED, 0 /*border_value*/);
	  // Configure magnitude/phase kernels:
#ifdef ARM_COMPUTE_CL
	  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	  {
        _mag_phase_pyr[i*_nGpyrLayers+j].configure((_dx.get() + i*_nGpyrLayers+j), (_dy.get() + i*_nGpyrLayers+j), get_grd_pyramid(i,j), get_rot_pyramid(i,j));
	  }
#endif
#ifndef ARM_COMPUTE_CL
	  if(std::is_same<typename std::decay<tensor>::type, Tensor>::value)
	  {
        _mag_phase_pyr[i*_nGpyrLayers+j].configure((_dx.get() + i*_nGpyrLayers+j), (_dy.get() + i*_nGpyrLayers+j), get_grd_pyramid(i,j), get_rot_pyramid(i,j), false);
	  }
#endif
    }
  }

  for(unsigned int i = 0; i < _nOctaves; i++){
	for (unsigned int  j = 0; j < _nGpyrLayers; j++){
      // Run derivative kernels:
      (_driv_pyr.get() + i*_nGpyrLayers+j)->run();

      // Run magnitude kernels:
      (_mag_phase_pyr.get() + i*_nGpyrLayers+j)->run();
    }
  }

}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
bool EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::refine_local_extrema(SiftKeypoint & kpt)
{
  int nGpyrLayers = _nDogLayers + 1;

  int w, h;
  int octave = kpt.octave;
  int layer = kpt.layer;
  int r = (int)kpt.ri;
  int c = (int)kpt.ci;

  int xs_i = 0, xr_i = 0, xc_i = 0;
  float tmp_r, tmp_c, tmp_layer;
  float xr = 0.0f, xc = 0.0f, xs = 0.0f;
  float x_hat[3] = {xc, xr, xs};

  float dx, dy, ds;
  float dxx, dyy, dss, dxs, dys, dxy;

  tmp_r = (float)r;
  tmp_c = (float)c;
  tmp_layer = (float)layer;

  // Interpolation (x,y,sigma) 3D space to find sub-pixel accurate
  // location of keypoints.
  int i = 0;
  for (; i < SIFT_MAX_INTERP_STEPS; i ++){
  	c += xc_i;
  	r += xr_i;

  	w = get_dog_pyramid(octave,layer)->info()->dimension(0);
  	h = get_dog_pyramid(octave,layer)->info()->dimension(1);

    ////////////////////////////////////////////////////////////////
    // Current data
    ////////////////////////////////////////////////////////////////
    // Current data, the left-top side of the window
    const unsigned char *_curr_lft_top_ptr = get_dog_pyramid(octave,layer)->buffer() + (r-1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c - 1;
    // Current data, the left-mid side of the window
    const unsigned char *_curr_lft_mid_ptr = get_dog_pyramid(octave,layer)->buffer() + r     * get_dog_pyramid(octave,layer)->info()->dimension(0) + c - 1;
    // Current data, the left-low side of the window
    const unsigned char *_curr_lft_low_ptr = get_dog_pyramid(octave,layer)->buffer() + (r+1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c - 1;

    // Current data, the right-top side of the window
    const unsigned char *_curr_rht_top_ptr = get_dog_pyramid(octave,layer)->buffer() + (r-1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c + 1;
    // Current data, the right-mid side of the window
    const unsigned char *_curr_rht_mid_ptr = get_dog_pyramid(octave,layer)->buffer() + r     * get_dog_pyramid(octave,layer)->info()->dimension(0) + c + 1;
    // Current data, the right-low side of the window
    const unsigned char *_curr_rht_low_ptr = get_dog_pyramid(octave,layer)->buffer() + (r+1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c + 1;

    // Current data, the mid-top side of the window
    const unsigned char *_curr_mid_top_ptr = get_dog_pyramid(octave,layer)->buffer() + (r-1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c;
    // Current data, the mid-mid side of the window
    const unsigned char *_curr_mid_mid_ptr = get_dog_pyramid(octave,layer)->buffer() + r     * get_dog_pyramid(octave,layer)->info()->dimension(0) + c;
    // Current data, the mid-low side of the window
    const unsigned char *_curr_mid_low_ptr = get_dog_pyramid(octave,layer)->buffer() + (r+1) * get_dog_pyramid(octave,layer)->info()->dimension(0) + c;

    ////////////////////////////////////////////////////////////////
    // Low data
    ////////////////////////////////////////////////////////////////
    // Low data, the left-top side of the window
    const unsigned char *_low_lft_top_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r-1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c - 1;
    // Low data, the left-mid side of the window
    const unsigned char *_low_lft_mid_ptr = get_dog_pyramid(octave,layer-1)->buffer() + r     * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c - 1;
    // Low data, the left-low side of the window
    const unsigned char *_low_lft_low_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r+1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c - 1;

    // Low data, the right-top side of the window
    const unsigned char *_low_rht_top_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r-1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c + 1;
    // Low data, the right-mid side of the window
    const unsigned char *_low_rht_mid_ptr = get_dog_pyramid(octave,layer-1)->buffer() + r     * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c + 1;
    // Low data, the right-low side of the window
    const unsigned char *_low_rht_low_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r+1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c + 1;

    // Low data, the mid-top side of the window
    const unsigned char *_low_mid_top_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r-1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c;
    // Low data, the mid-mid side of the window
    const unsigned char *_low_mid_mid_ptr = get_dog_pyramid(octave,layer-1)->buffer() + r     * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c;
    // Low data, the mid-low side of the window
    const unsigned char *_low_mid_low_ptr = get_dog_pyramid(octave,layer-1)->buffer() + (r+1) * get_dog_pyramid(octave,layer-1)->info()->dimension(0) + c;

    ////////////////////////////////////////////////////////////////
    // High data
    ////////////////////////////////////////////////////////////////
    // High data, the left-top side of the window
    const unsigned char *_high_lft_top_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r-1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c - 1;
    // High data, the left-mid side of the window
    const unsigned char *_high_lft_mid_ptr = get_dog_pyramid(octave,layer+1)->buffer() + r     * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c - 1;
    // High data, the left-low side of the window
    const unsigned char *_high_lft_low_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r+1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c - 1;

    // High data, the right-top side of the window
    const unsigned char *_high_rht_top_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r-1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c + 1;
    // High data, the right-mid side of the window
    const unsigned char *_high_rht_mid_ptr = get_dog_pyramid(octave,layer+1)->buffer() + r     * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c + 1;
    // High data, the right-low side of the window
    const unsigned char *_high_rht_low_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r+1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c + 1;

    // High data, the mid-top side of the window
    const unsigned char *_high_mid_top_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r-1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c;
    // High data, the mid-mid side of the window
    const unsigned char *_high_mid_mid_ptr = get_dog_pyramid(octave,layer+1)->buffer() + r     * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c;
    // High data, the mid-low side of the window
    const unsigned char *_high_mid_low_ptr = get_dog_pyramid(octave,layer+1)->buffer() + (r+1) * get_dog_pyramid(octave,layer+1)->info()->dimension(0) + c;

  	dx = ((*_curr_rht_mid_ptr)*1.0 - (*_curr_lft_mid_ptr)) * 0.5f;
  	dy = ((*_curr_mid_low_ptr)*1.0 - (*_curr_mid_top_ptr)) * 0.5f;
  	ds = ((*_high_mid_mid_ptr)*1.0 - (*_low_mid_mid_ptr)) * 0.5f;
  	float dD[3] = {-dx, -dy, -ds};

  	float v2 = 2.0f * (*_curr_mid_mid_ptr);
  	dxx = ((*_curr_rht_mid_ptr)*1.0 + (*_curr_lft_mid_ptr) - v2);
  	dyy = ((*_curr_mid_low_ptr)*1.0 + (*_curr_mid_top_ptr) - v2);
  	dss = ((*_high_mid_mid_ptr)*1.0 + (*_low_mid_mid_ptr) - v2);
  	dxy = ((*_curr_rht_low_ptr)*1.0 - (*_curr_lft_low_ptr) -
  		   (*_curr_rht_top_ptr)*1.0 + (*_curr_lft_top_ptr)) * 0.25f;
  	dxs = ((*_high_rht_mid_ptr)*1.0 - (*_high_lft_mid_ptr) -
  		   (*_low_rht_mid_ptr)*1.0  + (*_low_lft_mid_ptr)) * 0.25f;
  	dys = ((*_high_mid_low_ptr)*1.0 - (*_high_mid_top_ptr) -
  		   (*_low_mid_low_ptr)*1.0  + (*_low_mid_top_ptr)) * 0.25f;


  	// The scale in two sides of the equation should cancel each other.
  	float H[3][3] = {{dxx, dxy, dxs},
  	                 {dxy, dyy, dys},
  	                 {dxs, dys, dss}};
  	float Hinvert[3][3];
  	float det;

  	// Matrix inversion
  	// INVERT_3X3 = DETERMINANT_3X3, then SCALE_ADJOINT_3X3;
  	// Using INVERT_3X3(Hinvert, det, H) is more convenient;
  	// but using separate ones, we can check det==0 easily.
  	float tmp;
  	DETERMINANT_3X3 (det, H);
  	if (fabsf(det) < (std::numeric_limits<float>::min)()){
  	  break;
  	}
  	tmp = 1.0f / (det);
  	//INVERT_3X3(Hinvert, det, H);
  	SCALE_ADJOINT_3X3 (Hinvert, tmp, H);
  	MAT_DOT_VEC_3X3(x_hat, Hinvert, dD);

  	xs = x_hat[2];
  	xr = x_hat[1];
  	xc = x_hat[0];

  	// Update tmp data for keypoint update.
  	tmp_r = r + xr;
  	tmp_c = c + xc;
  	tmp_layer = layer + xs;

  	// Make sure there is room to move for next iteration.
  	xc_i= ((xc >=  SIFT_KEYPOINT_SUBPiXEL_THR && c < w - 2) ?  1 : 0)
  		+ ((xc <= -SIFT_KEYPOINT_SUBPiXEL_THR && c > 1    ) ? -1 : 0);

  	xr_i= ((xr >=  SIFT_KEYPOINT_SUBPiXEL_THR && r < h - 2) ?  1 : 0)
  		+ ((xr <= -SIFT_KEYPOINT_SUBPiXEL_THR && r > 1    ) ? -1 : 0);

  	if (xc_i == 0 && xr_i == 0 && xs_i == 0){
  	  break;
  	}
  }

  // We MIGHT be able to remove the following two checking conditions.
  // Condition 1
  if (i >= SIFT_MAX_INTERP_STEPS)
    return false;
  // Condition 2.
  if (fabsf(xc) >= 1.5 || fabsf(xr) >= 1.5 || fabsf(xs) >= 1.5)
    return false;

  // If (r, c, layer) is out of range, return false.
  if (tmp_layer < 0 || tmp_layer > (nGpyrLayers - 1)
  	|| tmp_r < 0 || tmp_r > h - 1
  	|| tmp_c < 0 || tmp_c > w - 1)
    return false;

  {
    // Current data, the mid-mid side of the window
    const unsigned char *_curr_mid_mid_ptr = get_dog_pyramid(octave,layer)->buffer() + r * get_dog_pyramid(octave,layer)->info()->dimension(0) + c;

    float value = (*_curr_mid_mid_ptr) + 0.5f * (dx * xc + dy * xr + ds * xs);

    if (fabsf(value) < SIFT_CONTR_THR)
      return false;

    float trH = dxx +  dyy;
    float detH = dxx * dyy - dxy * dxy;
    float response = (SIFT_CURV_THR + 1) * (SIFT_CURV_THR + 1) / (SIFT_CURV_THR);

    if(detH <= 0 || (trH * trH / detH) >= response)
      return false;
  }

  // Coordinates in the current layer.
  kpt.ci = tmp_c;
  kpt.ri = tmp_r;
  kpt.layer_scale = SIFT_SIGMA * powf(2.0f, tmp_layer/SIFT_INTVLS);

  int firstOctave = SIFT_IMG_DBL ? -1 : 0;
  float norm = powf(2.0f, (float) (octave + firstOctave));
  // Coordinates in the normalized format (compared to the original image).
  kpt.c = tmp_c * norm;
  kpt.r = tmp_r * norm;
  kpt.rlayer = tmp_layer;
  kpt.layer = layer;

  // Formula: Scale = sigma0 * 2^octave * 2^(layer/S);
  kpt.scale = kpt.layer_scale * norm;

  return true;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
float EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::compute_orientation_hist_with_gradient(
	SiftKeypoint & kpt,
	float * & hist)
{
  int nBins = SIFT_ORI_HIST_BINS;
  int octave = kpt.octave;
  int layer = kpt.layer;

  float kptr = kpt.ri;
  float kptc = kpt.ci;
  float kpt_scale = kpt.layer_scale;

  int kptr_i = (int) (kptr + 0.5f);
  int kptc_i = (int) (kptc + 0.5f);
  float d_kptr = kptr - kptr_i;
  float d_kptc = kptc - kptc_i;

  float sigma = SIFT_ORI_SIG_FCTR * kpt_scale;
  int win_radius = (int) (SIFT_ORI_RADIUS * kpt_scale);
  int win_width = win_radius * 2 + 1;
  float exp_factor = -1.0f / (2.0f * sigma * sigma);

  int w = get_grd_pyramid(octave,layer)->info()->dimension(0);
  int h = get_grd_pyramid(octave,layer)->info()->dimension(1);

  int r, c;
  float magni, angle, weight;
  int bin;
  float fbin; // float point bin

  float *tmpHist = new float[nBins];
  memset(tmpHist, 0, nBins * sizeof(float));

#ifdef ARM_COMPUTE_CL
	  // Map buffer if creating a CLTensor
	  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	  {
        get_grd_pyramid(octave,layer)->map();
        get_rot_pyramid(octave,layer)->map();
	  }
#endif

  for (int i = -win_radius; i <= win_radius; i ++){ // rows
    r = kptr_i + i;
    if (r <= 0 || r >= h-1) // Cannot calculate dy
      continue;
    for (int j = -win_radius; j <= win_radius; j ++){ // columns
      c = kptc_i + j;
      if (c <= 0 || c >= w-1)
        continue;

      magni = *reinterpret_cast<signed short *>(get_grd_pyramid(octave,layer)->buffer() + (r * get_grd_pyramid(octave,layer)->info()->dimension(0) + c)*sizeof(signed short))*1.0f;
      angle = *reinterpret_cast<unsigned char *>(get_rot_pyramid(octave,layer)->buffer() + (r * get_rot_pyramid(octave,layer)->info()->dimension(0) + c)*sizeof(unsigned char))*1.0f;

//      fbin = angle * nBins / _2PI;
      fbin = angle * nBins / 360.0f;
      weight = expf(((i-d_kptr) * (i-d_kptr) + (j-d_kptc) * (j-d_kptc)) * exp_factor);

#define SIFT_ORI_BILINEAR
#ifdef SIFT_ORI_BILINEAR
      bin = (int) (fbin - 0.5f);
      float d_fbin = fbin - 0.5f - bin;

      float mw = weight * magni;
      float dmw = d_fbin * mw;
      tmpHist[(bin + nBins) % nBins] += mw - dmw;
      tmpHist[(bin + 1) % nBins] += dmw;
#else
      bin = (int) (fbin);
      tmpHist[bin] += magni * weight;
#endif
    }
  }
#ifdef ARM_COMPUTE_CL
	  // Unmap buffer if creating a CLTensor
	  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	  {
        get_grd_pyramid(octave,layer)->unmap();
        get_rot_pyramid(octave,layer)->unmap();
	  }
#endif

#define TMPHIST(idx) (idx < 0? tmpHist[0] : (idx >= nBins ? tmpHist[nBins - 1] : tmpHist[idx]))

#define USE_SMOOTH1	1
#if		USE_SMOOTH1

  // Smooth the histogram. Algorithm comes from OpenCV.
  hist[0] = (tmpHist[0] + tmpHist[2]) * 1.0f / 16.0f +
  	        (tmpHist[0] + tmpHist[1]) * 4.0f / 16.0f +
  	         tmpHist[0] * 6.0f / 16.0f;
  hist[1] = (tmpHist[0] + tmpHist[3]) * 1.0f / 16.0f +
  	        (tmpHist[0] + tmpHist[2]) * 4.0f / 16.0f +
  	         tmpHist[1] * 6.0f / 16.0f;
  hist[nBins - 2] = (tmpHist[nBins - 4] + tmpHist[nBins - 1]) * 1.0f / 16.0f +
  	                (tmpHist[nBins - 3] + tmpHist[nBins - 1]) * 4.0f / 16.0f +
  	                 tmpHist[nBins - 2] * 6.0f / 16.0f;
  hist[nBins - 1] = (tmpHist[nBins - 3] + tmpHist[nBins - 1]) * 1.0f / 16.0f +
  	                (tmpHist[nBins - 2] + tmpHist[nBins - 1]) * 4.0f / 16.0f +
  	                 tmpHist[nBins - 1] * 6.0f / 16.0f;

  for(int i = 2; i < nBins - 2; i ++){
  	hist[i] = (tmpHist[i - 2] + tmpHist[i + 2]) * 1.0f / 16.0f +
  		      (tmpHist[i - 1] + tmpHist[i + 1]) * 4.0f / 16.0f +
  		       tmpHist[i] * 6.0f / 16.0f;
  }

#else
  // Yet another smooth function
  // Algorithm comes from the vl_feat implementation.
  for (int iter = 0; iter < 6; iter ++){
    float prev = TMPHIST(nBins - 1);
    float first = TMPHIST(0);
    int i;
    for (i = 0; i < nBins - 1; i ++){
      float newh = (prev + TMPHIST(i) + TMPHIST(i + 1)) / 3.0f ;
      prev = hist[i];
      hist[i] = newh;
    }
    hist[i] = (prev + hist[i] + first) / 3.0f;
  }
#endif

  // Find the maximum item of the histogram
  float maxitem = hist[0];
  int max_i = 0;
  for (int i = 0; i < nBins; i ++){
    if (maxitem < hist[i]){
      maxitem = hist[i];
      max_i = i;
    }
  }

  kpt.ori = max_i * _2PI / nBins;

  delete [] tmpHist;
  return maxitem;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::detect_keypoints(std::list<SiftKeypoint> & kpt_list)
{
  SiftKeypoint kpt;

  int w, h;
  unsigned char val;

  int nBins = SIFT_ORI_HIST_BINS;
  float * hist = new float[nBins];
  int nGpyrLayers = _nDogLayers + 1;

  // Some paper uses other thresholds, for example 3.0f for all cases
  // In Lowe's paper, |D(x)|<0.03 will be rejected.
  unsigned int contr_thr = (unsigned int)(0.8f * SIFT_CONTR_THR);

///////////////////////////////////////////////
//const std::string ppm_filename1 = "debug_keypoint1.txt";
//const std::string ppm_filename2 = "debug_keypoint2.txt";
//std::ofstream fs1,fs2;
//fs1.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
//fs1.open(ppm_filename1, std::ios::out);
//fs2.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
//fs2.open(ppm_filename2, std::ios::out);
///////////////////////////////////////////////

  for (int i = 0; i < _nOctaves; i++){
    for (int j = 1; j < _nDogLayers - 1; j++){
#ifdef ARM_COMPUTE_CL
	  // Map buffer if creating a CLTensor
	  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	  {
        get_dog_pyramid(i,j-1)->map();
        get_dog_pyramid(i,j  )->map();
        get_dog_pyramid(i,j+1)->map();
	  }
#endif

      ////////////////////////////////////////////////////////////////
      // High data
      ////////////////////////////////////////////////////////////////
      // High data, the left-top side of the window
	  const unsigned char *_high_lft_top_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(-1, -1));
	  // High data, the left-mid side of the window
	  const unsigned char *_high_lft_mid_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(-1, 0));
	  // High data, the left-low side of the window
	  const unsigned char *_high_lft_low_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(-1, 1));

      // High data, the mid-top side of the window
	  const unsigned char *_high_mid_top_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(0, -1));
	  // High data, the mid-mid side of the window
	  const unsigned char *_high_mid_mid_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(0, 0));
	  // High data, the mid-low side of the window
	  const unsigned char *_high_mid_low_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(0, 1));

      // High data, the right-top side of the window
	  const unsigned char *_high_rht_top_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(1, -1));
	  // High data, the right-mid side of the window
	  const unsigned char *_high_rht_mid_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(1, 0));
	  // High data, the right-low side of the window
	  const unsigned char *_high_rht_low_ptr = get_dog_pyramid(i,j+1)->buffer() + get_dog_pyramid(i,j+1)->info()->offset_element_in_bytes(Coordinates(1, 1));

      ////////////////////////////////////////////////////////////////
      // Current data
      ////////////////////////////////////////////////////////////////
      // Current data, the left-top side of the window
	  const unsigned char *_curr_lft_top_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(-1, -1));
	  // Current data, the left-mid side of the window
	  const unsigned char *_curr_lft_mid_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(-1, 0));
	  // Current data, the left-low side of the window
	  const unsigned char *_curr_lft_low_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(-1, 1));

      // Current data, the mid-top side of the window
	  const unsigned char *_curr_mid_top_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(0, -1));
	  // Current data, the mid-mid side of the window
	  const unsigned char *_curr_mid_mid_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(0, 0));
	  // Current data, the mid-low side of the window
	  const unsigned char *_curr_mid_low_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(0, 1));

      // Current data, the right-top side of the window
	  const unsigned char *_curr_rht_top_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(1, -1));
	  // Current data, the right-mid side of the window
	  const unsigned char *_curr_rht_mid_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(1, 0));
	  // Current data, the right-low side of the window
	  const unsigned char *_curr_rht_low_ptr = get_dog_pyramid(i,j  )->buffer() + get_dog_pyramid(i,j  )->info()->offset_element_in_bytes(Coordinates(1, 1));

      ////////////////////////////////////////////////////////////////
      // Low data
      ////////////////////////////////////////////////////////////////
      // Low data, the left-top side of the window
	  const unsigned char *_low_lft_top_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(-1, -1));
	  // Low data, the left-mid side of the window
	  const unsigned char *_low_lft_mid_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(-1, 0));
	  // Low data, the left-low side of the window
	  const unsigned char *_low_lft_low_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(-1, 1));

      // Low data, the mid-top side of the window
	  const unsigned char *_low_mid_top_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(0, -1));
	  // Low data, the mid-mid side of the window
	  const unsigned char *_low_mid_mid_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(0, 0));
	  // Low data, the mid-low side of the window
	  const unsigned char *_low_mid_low_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(0, 1));

      // Low data, the right-top side of the window
	  const unsigned char *_low_rht_top_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(1, -1));
	  // Low data, the right-mid side of the window
	  const unsigned char *_low_rht_mid_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(1, 0));
	  // Low data, the right-low side of the window
	  const unsigned char *_low_rht_low_ptr = get_dog_pyramid(i,j-1)->buffer() + get_dog_pyramid(i,j-1)->info()->offset_element_in_bytes(Coordinates(1, 1));

	  // Configure kernel window
	  constexpr unsigned int num_elems_processed_per_iteration = 1;
	  constexpr unsigned int num_elems_read_per_iteration      = 1;
	  constexpr unsigned int num_elems_written_per_iteration   = 1;

	  Window                 win = calculate_max_window(*get_dog_pyramid(i,j)->info(), Steps(num_elems_processed_per_iteration), true/*border_undefined*/, BorderSize(SIFT_IMG_BORDER));
      AccessWindowHorizontal high_access(get_dog_pyramid(i,j+1)->info(), 0, num_elems_processed_per_iteration);
      AccessWindowHorizontal low_access(get_dog_pyramid(i,j-1)->info(), 0, num_elems_processed_per_iteration);

	  update_window_and_padding(win,
	                            AccessWindowRectangle(get_dog_pyramid(i,j)->info(), -BorderSize(SIFT_IMG_BORDER).left, -BorderSize(SIFT_IMG_BORDER).top, num_elems_read_per_iteration, 1),
								high_access,
								low_access);

	  Iterator high_data_iter(get_dog_pyramid(i,j+1), win);
	  Iterator curr_data_iter(get_dog_pyramid(i,j  ), win);
	  Iterator low_data_iter (get_dog_pyramid(i,j-1), win);
	  execute_window_loop(win, [&](const Coordinates & id)
	  {
		// Pointer to high data
        const unsigned char *high_lft_top_ptr = reinterpret_cast<const unsigned char *>(_high_lft_top_ptr + high_data_iter.offset());
		const unsigned char *high_mid_top_ptr = reinterpret_cast<const unsigned char *>(_high_mid_top_ptr + high_data_iter.offset());
		const unsigned char *high_rht_top_ptr = reinterpret_cast<const unsigned char *>(_high_rht_top_ptr + high_data_iter.offset());

		const unsigned char *high_lft_mid_ptr = reinterpret_cast<const unsigned char *>(_high_lft_mid_ptr + high_data_iter.offset());
		const unsigned char *high_mid_mid_ptr = reinterpret_cast<const unsigned char *>(_high_mid_mid_ptr + high_data_iter.offset());
		const unsigned char *high_rht_mid_ptr = reinterpret_cast<const unsigned char *>(_high_rht_mid_ptr + high_data_iter.offset());

		const unsigned char *high_lft_low_ptr = reinterpret_cast<const unsigned char *>(_high_lft_low_ptr + high_data_iter.offset());
		const unsigned char *high_mid_low_ptr = reinterpret_cast<const unsigned char *>(_high_mid_low_ptr + high_data_iter.offset());
		const unsigned char *high_rht_low_ptr = reinterpret_cast<const unsigned char *>(_high_rht_low_ptr + high_data_iter.offset());

		// Pointer to current data
        const unsigned char *curr_lft_top_ptr = reinterpret_cast<const unsigned char *>(_curr_lft_top_ptr + curr_data_iter.offset());
		const unsigned char *curr_mid_top_ptr = reinterpret_cast<const unsigned char *>(_curr_mid_top_ptr + curr_data_iter.offset());
		const unsigned char *curr_rht_top_ptr = reinterpret_cast<const unsigned char *>(_curr_rht_top_ptr + curr_data_iter.offset());

		const unsigned char *curr_lft_mid_ptr = reinterpret_cast<const unsigned char *>(_curr_lft_mid_ptr + curr_data_iter.offset());
		const unsigned char *curr_mid_mid_ptr = reinterpret_cast<const unsigned char *>(_curr_mid_mid_ptr + curr_data_iter.offset());
		const unsigned char *curr_rht_mid_ptr = reinterpret_cast<const unsigned char *>(_curr_rht_mid_ptr + curr_data_iter.offset());

		const unsigned char *curr_lft_low_ptr = reinterpret_cast<const unsigned char *>(_curr_lft_low_ptr + curr_data_iter.offset());
		const unsigned char *curr_mid_low_ptr = reinterpret_cast<const unsigned char *>(_curr_mid_low_ptr + curr_data_iter.offset());
		const unsigned char *curr_rht_low_ptr = reinterpret_cast<const unsigned char *>(_curr_rht_low_ptr + curr_data_iter.offset());

		// Pointer to low data
        const unsigned char *low_lft_top_ptr = reinterpret_cast<const unsigned char *>(_low_lft_top_ptr + low_data_iter.offset());
		const unsigned char *low_mid_top_ptr = reinterpret_cast<const unsigned char *>(_low_mid_top_ptr + low_data_iter.offset());
		const unsigned char *low_rht_top_ptr = reinterpret_cast<const unsigned char *>(_low_rht_top_ptr + low_data_iter.offset());

		const unsigned char *low_lft_mid_ptr = reinterpret_cast<const unsigned char *>(_low_lft_mid_ptr + low_data_iter.offset());
		const unsigned char *low_mid_mid_ptr = reinterpret_cast<const unsigned char *>(_low_mid_mid_ptr + low_data_iter.offset());
		const unsigned char *low_rht_mid_ptr = reinterpret_cast<const unsigned char *>(_low_rht_mid_ptr + low_data_iter.offset());

		const unsigned char *low_lft_low_ptr = reinterpret_cast<const unsigned char *>(_low_lft_low_ptr + low_data_iter.offset());
		const unsigned char *low_mid_low_ptr = reinterpret_cast<const unsigned char *>(_low_mid_low_ptr + low_data_iter.offset());
		const unsigned char *low_rht_low_ptr = reinterpret_cast<const unsigned char *>(_low_rht_low_ptr + low_data_iter.offset());

        val = *curr_mid_mid_ptr;

        bool bExtrema =
            (val >= contr_thr &&
            val > *high_lft_top_ptr && val > *high_mid_top_ptr && val > *high_rht_top_ptr &&
            val > *high_lft_mid_ptr && val > *high_mid_mid_ptr && val > *high_rht_mid_ptr &&
            val > *high_lft_low_ptr && val > *high_mid_low_ptr && val > *high_rht_low_ptr &&

            val > *curr_lft_top_ptr && val > *curr_mid_top_ptr && val > *curr_rht_top_ptr &&
            val > *curr_lft_mid_ptr							   && val > *curr_rht_mid_ptr &&
            val > *curr_lft_low_ptr && val > *curr_mid_low_ptr && val > *curr_rht_low_ptr &&

            val > *low_lft_top_ptr && val > *low_mid_top_ptr && val > *low_rht_top_ptr &&
            val > *low_lft_mid_ptr && val > *low_mid_mid_ptr && val > *low_rht_mid_ptr &&
            val > *low_lft_low_ptr && val > *low_mid_low_ptr && val > *low_rht_low_ptr);

        if (bExtrema){
          kpt.octave = i;
          kpt.layer = j;
          kpt.ri = (float)id.y();
          kpt.ci = (float)id.x();

/////////////////////////////////////////////////
//fs1 << "oct:"  << kpt.octave
//    << " lyr:" << kpt.layer
//    << " ri:"  << kpt.ri
//    << " ci:"  << kpt.ci
//    << " val:" << val*1.0
//    << " thr:" << contr_thr
//	<< " r:"   << kpt.r
//	<< " c:"   << kpt.c
//	<< " ori:" << kpt.ori
//   << "\n";
/////////////////////////////////////////////////

          bool bGoodKeypoint = refine_local_extrema(kpt);

          if(bGoodKeypoint){
/////////////////////////////////////////////////
//fs2 << "oct:"  << kpt.octave
//	<< " lyr:" << kpt.layer
//	<< " ri:"  << kpt.ri
//	<< " ci:"  << kpt.ci
//	<< " val:" << val*1.0
//	<< " thr:" << contr_thr
//	<< " r:"   << kpt.r
//	<< " c:"   << kpt.c
//	<< " ori:" << kpt.ori
//	<< "\n";
/////////////////////////////////////////////////

            float max_mag = compute_orientation_hist_with_gradient(kpt, hist);

            float threshold = max_mag * SIFT_ORI_PEAK_RATIO;

            for (int ii = 0; ii < nBins; ii ++){
              // Use 3 points to fit a curve and find the accurate location of a keypoints
              int left = ii  > 0 ? ii - 1 : nBins - 1;
              int right = ii < (nBins-1) ? ii + 1 : 0;
              float currHist = hist[ii];
              float lhist = hist[left];
              float rhist = hist[right];
              if (currHist > lhist && currHist > rhist && currHist > threshold){
                // Refer to here: http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
                float accu_ii = ii + 0.5f * (lhist - rhist) / (lhist - 2.0f*currHist + rhist);

                // Since bin index means the starting point of a bin, so the real orientation should be bin index
                // plus 0.5. for example, angles in bin 0 should have a mean value of 5 instead of 0;
                accu_ii += 0.5f;
                accu_ii = accu_ii<0 ? (accu_ii + nBins) : accu_ii>=nBins ? (accu_ii - nBins) : accu_ii;
                // The magnitude should also calculate the max number based on fitting
                // But since we didn't actually use it in image matching, we just lazily
                // use the histogram value.
                kpt.mag = currHist;
                kpt.ori = accu_ii * _2PI / nBins;
                kpt_list.push_back(kpt);
///////////////////////////////////////////////
//fs1 << "oct:"  << kpt.octave
//	<< " lyr:" << kpt.layer
//	<< " ri:"  << kpt.ri
//	<< " ci:"  << kpt.ci
//	<< " r:"   << kpt.r
//	<< " c:"   << kpt.c
//	<< " max:" << max_mag
//	<< " thr:" << threshold
//	<< " mag:" << kpt.mag
//	<< " ori:" << kpt.ori
//	<< "\n";
///////////////////////////////////////////////
              }
            }
          }

        }

	  },
	  high_data_iter,curr_data_iter,low_data_iter);

#ifdef ARM_COMPUTE_CL
	  // Unmap buffer if creating a CLTensor
	  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	  {
        get_dog_pyramid(i,j-1)->unmap();
        get_dog_pyramid(i,j  )->unmap();
        get_dog_pyramid(i,j+1)->unmap();
	  }
#endif

    }
  }

  delete [] hist;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
void EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::extract_descriptor(std::list<SiftKeypoint> & kpt_list)
{
  // Number of subregions, default 4x4 subregions.
  // The width of subregion is determined by the scale of the keypoint.
  // Or, in Lowe's SIFT paper[2004], width of subregion is 16x16.
  int nSubregion = SIFT_DESCR_WIDTH;
  int nHalfSubregion = nSubregion >> 1;

  // Number of histogram bins for each descriptor subregion.
  int nBinsPerSubregion = SIFT_DESCR_HIST_BINS;
  float nBinsPerSubregionPerDegree = (float)nBinsPerSubregion / _2PI;

  // 3-D structure for histogram bins (rbin, cbin, obin);
  // (rbin, cbin, obin) means (row of hist bin, column of hist bin, orientation bin)
  // In Lowe's paper, 4x4 histogram, each has 8 bins.
  // that means for each (rbin, cbin), there are 8 bins in the histogram.

  // In this implementation, histBin is a circular buffer.
  // we expand the cube by 1 for each direction.
  int nBins = nSubregion * nSubregion * nBinsPerSubregion;
  int nHistBins = (nSubregion + 2) * (nSubregion + 2) * (nBinsPerSubregion + 2);
  int nSliceStep = (nSubregion + 2) * (nBinsPerSubregion + 2); // 32
  int nRowStep = (nBinsPerSubregion + 2);
  float * histBin = new float[nHistBins];

  float exp_scale = -2.0f / (nSubregion * nSubregion);  // -1/(2* nSubregion/2 * nSubregion/2)

  for (std::list<SiftKeypoint>::iterator kpt = kpt_list.begin(); kpt != kpt_list.end(); kpt ++){
    // Keypoint information
    int octave = kpt->octave;
    int layer = kpt->layer;

    float kpt_ori = kpt->ori;
    float kptr = kpt->ri;
    float kptc = kpt->ci;
    float kpt_scale = kpt->layer_scale;

    // Nearest coordinate of keypoints
    int kptr_i = (int)(kptr + 0.5f);
    int kptc_i = (int)(kptc + 0.5f);
    float d_kptr = kptr_i - kptr;
    float d_kptc = kptc_i - kptc;

    int layer_index = octave * _nGpyrLayers + layer;
    int w = get_grd_pyramid(octave,layer)->info()->dimension(0);
    int h = get_grd_pyramid(octave,layer)->info()->dimension(1);

    // Note for Gaussian weighting.
    // OpenCV and vl_feat uses non-fixed size of subregion.
    // But they all use (0.5 * 4) as the Gaussian weighting sigma.
    // In Lowe's paper, he uses 16x16 sample region,
    // partition 16x16 region into 16 4x4 subregion.
    float subregion_width = SIFT_DESCR_SCL_FCTR * kpt_scale;
    int win_size = (int)(SQRT2 * subregion_width * (nSubregion + 1) * 0.5f + 0.5f);

    // Normalized cos() and sin() value.
    float sin_t = sinf(kpt_ori) / (float)subregion_width;
    float cos_t = cosf(kpt_ori) / (float)subregion_width;

    // Re-init histBin
    memset(histBin, 0, nHistBins * sizeof(float));

    // Start to calculate the histogram in the sample region.
    float rr, cc;
    float mag, angle, gaussian_weight;

    // Used for tri-linear interpolation.
    //int rbin_i, cbin_i, obin_i;
    float rrotate, crotate;
    float rbin, cbin, obin;
    float d_rbin, d_cbin, d_obin;

    // Boundary of sample region.
    int r, c;
    int left = (std::max)(-win_size, 1 - kptc_i);
    int right = (std::min)(win_size, w - 2 - kptc_i);
    int top = (std::max)(-win_size, 1 - kptr_i);
    int bottom = (std::min)(win_size, h - 2 - kptr_i);

#ifdef ARM_COMPUTE_CL
	// Map buffer if creating a CLTensor
	if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	{
      get_grd_pyramid(octave,layer)->map();
      get_rot_pyramid(octave,layer)->map();
	}
#endif

    for (int i = top; i <= bottom; i ++){ // rows
      for (int j = left; j <= right; j ++){ // columns
        // Accurate position relative to (kptr, kptc)
        rr = i + d_kptr;
        cc = j + d_kptc;

        // Rotate the coordinate of (i, j)
        rrotate = ( cos_t * cc + sin_t * rr);
        crotate = (-sin_t * cc + cos_t * rr);

        // Since for a bin array with 4x4 bins, the center is actually at (1.5, 1.5)
        rbin =  rrotate + nHalfSubregion - 0.5f;
        cbin =  crotate + nHalfSubregion - 0.5f;

        // rbin, cbin range is (-1, d); if outside this range, then the pixel is counted.
        if (rbin <= -1 || rbin >= nSubregion || cbin <= -1 || cbin >= nSubregion)
          continue;

        // All the data need for gradient computation are valid, no border issues.
        r = kptr_i + i;
        c = kptc_i + j;
        mag = *reinterpret_cast<signed short *>(get_grd_pyramid(octave,layer)->buffer() + (r * get_grd_pyramid(octave,layer)->info()->dimension(0) + c)*sizeof(signed short))*1.0f;
        angle = *reinterpret_cast<unsigned char *>(get_rot_pyramid(octave,layer)->buffer() + (r * get_rot_pyramid(octave,layer)->info()->dimension(0) + c)*sizeof(unsigned char))*_2PI/360.0f - kpt_ori;

        float angle1 = (angle < 0) ? (_2PI + angle) : angle; // Adjust angle to [0, 2PI)
        obin = angle1 * nBinsPerSubregionPerDegree;

        int x0, y0, z0;
        int x1, y1, z1;
        y0 = (int) floor(rbin);
        x0 = (int) floor(cbin);
        z0 = (int) floor(obin);
        d_rbin = rbin - y0;
        d_cbin = cbin - x0;
        d_obin = obin - z0;
        x1 = x0 + 1;
        y1 = y0 + 1;
        z1 = z0 + 1;

        // Gaussian weight relative to the center of sample region.
        gaussian_weight = expf((rrotate * rrotate + crotate * crotate) * exp_scale);

        // Gaussian-weighted magnitude
        float gm = mag * gaussian_weight;
        // Tri-linear interpolation

        float vr1, vr0;
        float vrc11, vrc10, vrc01, vrc00;
        float vrco110, vrco111, vrco100, vrco101,
        	vrco010, vrco011, vrco000, vrco001;

        vr1 = gm * d_rbin;
        vr0 = gm - vr1;
        vrc11   = vr1   * d_cbin;
        vrc10   = vr1   - vrc11;
        vrc01   = vr0   * d_cbin;
        vrc00   = vr0   - vrc01;
        vrco111 = vrc11 * d_obin;
        vrco110 = vrc11 - vrco111;
        vrco101 = vrc10 * d_obin;
        vrco100 = vrc10 - vrco101;
        vrco011 = vrc01 * d_obin;
        vrco010 = vrc01 - vrco011;
        vrco001 = vrc00 * d_obin;
        vrco000 = vrc00 - vrco001;

        // int idx =  y0  * nSliceStep + x0  * nRowStep + z0;
        // All coords are offseted by 1. so x=[1, 4], y=[1, 4];
        // data for -1 coord is stored at position 0;
        // data for 8 coord is stored at position 9.
        // z doesn't need to move.
        int idx =  y1  * nSliceStep + x1 * nRowStep + z0;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco000;

        idx ++;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco001;

        idx +=  nRowStep - 1;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco010;

        idx ++;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco011;

        idx += nSliceStep - nRowStep - 1;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco100;

        idx ++;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco101;

        idx +=  nRowStep - 1;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco110;

        idx ++;
        if (idx >= nHistBins) continue;
        histBin[idx] += vrco111;
      }
    }

#ifdef ARM_COMPUTE_CL
	// Unmap buffer if creating a CLTensor
	if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
	{
      get_grd_pyramid(octave,layer)->unmap();
      get_rot_pyramid(octave,layer)->unmap();
	}
#endif

    // Discard all the edges for row and column.
    // Only retrive edges for orientation bins.
    float *dstBins = new float[nBins];
    for (int i = 1; i <= nSubregion; i ++){// slice
      for (int j = 1; j <= nSubregion; j ++){// row
        int idx = i * nSliceStep + j * nRowStep;
        // comments: how this line works.
        // Suppose you want to write w=width, y=1, due to circular buffer,
        // we should write it to w=0, y=1; since we use a circular buffer,
        // it is written into w=width, y=1. Now, we fectch the data back.
        histBin[idx] = histBin[idx + nBinsPerSubregion];

        // comments: how this line works.
        // Suppose you want to write x=-1 y=1, due to circular, it should be
        // at y=1, x=width-1; since we use circular buffer, the value goes to
        // y=0, x=width, now, we need to get it back.
        if ( idx != 0)
        histBin[idx + nBinsPerSubregion + 1] = histBin[idx - 1];

        int idx1 = ((i-1) *nSubregion + j-1)* nBinsPerSubregion;
        for (int k = 0; k < nBinsPerSubregion; k ++){
          dstBins[idx1 + k] = histBin[idx + k];
        }
      }
    }

    // Normalize the histogram
    float sum_square = 0.0f;
    for (int i = 0; i < nBins; i ++)
    sum_square += dstBins[i] * dstBins[i];

#if (USE_FAST_FUNC == 1)
    float thr = fast_sqrt_f(sum_square) * SIFT_DESCR_MAG_THR;
#else
    float thr = sqrtf(sum_square) * SIFT_DESCR_MAG_THR;
#endif

    float tmp = 0.0;
    sum_square = 0.0;
    // Cut off the numbers bigger than 0.2 after normalized.
    for (int i = 0; i < nBins; i ++)
    {
      tmp = std::min(thr, dstBins[i]);
      dstBins[i] = tmp;
      sum_square += tmp * tmp;
    }

    // Re-normalize
    // The numbers are usually too small to store, so we use
    // a constant factor to scale up the numbers.
#if (USE_FAST_FUNC == 1)
  	float norm_factor = SIFT_INT_DESCR_FCTR / fast_sqrt_f(sum_square);
#else
  	float norm_factor = SIFT_INT_DESCR_FCTR / sqrtf(sum_square);
#endif
    for (int i = 0; i < nBins; i ++)
  	  dstBins[i] = dstBins[i] * norm_factor;

    memcpy(kpt->descriptors, dstBins, nBins * sizeof(float));

    if (dstBins) delete [] dstBins;

  }

  if (histBin) delete [] histBin;
}

template <typename tensor,class scale,class conv,unsigned int conv_matrix_size,class absdif,class deriv,class gradrot>
int EZSIFT<tensor,scale,conv,conv_matrix_size,absdif,deriv,gradrot>::sift(std::list<SiftKeypoint> & kpt_list)
{
  // Build image octaves
  build_octaves();
#if (DUMP_OCTAVE_IMAGE == 1)
  // Make sure all the OpenCL jobs are done executing:
  CLScheduler::get().sync();
  dump_octave_image();
#endif

  // Build Gaussian pyramid
  build_gaussian_pyramid();
#if (DUMP_GAUSSIAN_PYRAMID_IMAGE == 1)
  // Make sure all the OpenCL jobs are done executing:
  CLScheduler::get().sync();
  dump_gaussian_pyramid_image();
#endif

  // Build DoG pyramid
  build_dog_pyr();
#if (DUMP_DOG_IMAGE == 1)
  // Make sure all the OpenCL jobs are done executing:
  CLScheduler::get().sync();
  dump_dog_pyramid_image();
#endif

  // Build Gradient/Rotation pyramid
  build_grd_rot_pyr();

#ifdef ARM_COMPUTE_CL
  // Unmap buffer if creating a CLTensor
  if(std::is_same<typename std::decay<tensor>::type, CLTensor>::value)
  {
    // Make sure all the OpenCL jobs are done executing:
    CLScheduler::get().sync();
  }
#endif

  // Detect keypoints
  detect_keypoints(kpt_list);

  // Extract keypoint descriptors
  extract_descriptor(kpt_list);

  return 0;
}

#ifndef ARM_COMPUTE_CL
template class arm_compute::EZSIFT<arm_compute::Tensor,arm_compute::NEScale,arm_compute::NEConvolution7x7,7,arm_compute::NEAbsoluteDifference,arm_compute::NEDerivative,arm_compute::NEGradRot>;
#endif
#ifdef ARM_COMPUTE_CL
template class arm_compute::EZSIFT<arm_compute::CLTensor,arm_compute::CLScale,arm_compute::CLConvolution7x7,7,CLAbsoluteDifference,arm_compute::CLDerivative,arm_compute::CLGradRot>;
#endif

bool arm_compute::same_match_pair (MatchPair first, MatchPair second)
{
  if (first.c1 == second.c1 && first.r1 == second.r1
      && first.c2 == second.c2 && first.r2 == second.r2)
    return true;
  else
    return false;
}

void arm_compute::match_keypoints(std::list<SiftKeypoint> & kpt_list1,
	                              std::list<SiftKeypoint> & kpt_list2,
	                              std::list<MatchPair> & match_list)
{
  std::list<SiftKeypoint>::iterator kpt1;
  std::list<SiftKeypoint>::iterator kpt2;

  for (kpt1 = kpt_list1.begin(); kpt1 != kpt_list1.end(); kpt1 ++){
    // Position of the matched feature.
    int r1 = (int) kpt1->r;
    int c1 = (int) kpt1->c;

    float * descr1 = kpt1->descriptors;
    float score1 = (std::numeric_limits<float>::max)(); // highest score
    float score2 = (std::numeric_limits<float>::max)(); // 2nd highest score

    // Position of the matched feature.
    int r2, c2;
    for (kpt2 = kpt_list2.begin(); kpt2 != kpt_list2.end(); kpt2 ++){
      float score = 0;
      float * descr2 =  kpt2->descriptors;
      float dif;
      for (int i = 0; i < DEGREE_OF_DESCRIPTORS; i ++){
        dif = descr1[i] - descr2[i];
        score += dif * dif;
      }

      if (score < score1){
        score2 = score1;
        score1 = score;
        r2 = (int) kpt2->r;
        c2 = (int) kpt2->c;
      }else if(score < score2){
        score2 = score;
      }
    }

#if (USE_FAST_FUNC == 1)
    if (fast_sqrt_f(score1/score2) < SIFT_MATCH_NNDR_THR)
#else
    if (sqrtf(score1/score2) < SIFT_MATCH_NNDR_THR)
#endif
    {
      MatchPair mp;
      mp.r1 = r1;
      mp.c1 = c1;
      mp.r2 = r2;
      mp.c2 = c2;

      match_list.push_back(mp);
    }
  }

  match_list.unique(::same_match_pair);

#if PRINT_MATCH_KEYPOINTS
  list<MatchPair>::iterator p;
  int match_idx = 0;
  for (p = match_list.begin(); p != match_list.end(); p ++){
    printf("\tMatch %3d: (%4d, %4d) -> (%4d, %4d)\n", match_idx, p->r1, p->c1, p->r2, p->c2);
    match_idx ++;
  }
#endif

}
