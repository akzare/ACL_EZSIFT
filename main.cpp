/*  
Copyright (c) 2013, Robert Wang, email: robertwgh (at) gmail.com
Copyright (c) 2017, Armin Zare Zadeh, email: ali.a.zarezadeh (at) gmail.com
All rights reserved. https://sourceforge.net/p/ezsift

Description: Detect keypoints and extract descriptors from an input image.

Revision history:
  September, 15, 2013: initial version.
  May, 18, 2017: Used OpenCV for reading & writing & manipulating images and ACL Neon for computation on ARM NEON/Mali technology.
*/

#include "opencv2/opencv.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_neon.h"

#include <string>
#include <queue>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "acl_ezsift.h"

using namespace arm_compute;

//#define SIFTType arm_compute::CLEZSIFT
//const char *ALGNAME = "CL_";
#define SIFTType arm_compute::NEEZSIFT
const char *ALGNAME = "NE_";

int64 work_begin;
double work_fps;

inline void WorkBegin() { work_begin = cv::getTickCount(); }

inline void WorkEnd()
{
  int64 delta = cv::getTickCount() - work_begin;
  double freq = cv::getTickFrequency();
  work_fps = delta / freq;
}

class Text
{
  int fontFace;
  double fontScale;
  cv::Scalar color;
  int thickness;
  int lineType;
  bool bottomLeftOrigin;

public:
  Text(int _fontFace, double _fontScale, cv::Scalar _color, int _thickness=1, int _lineType=8, bool _bottomLeftOrigin=false)
        : fontFace(_fontFace), fontScale(_fontScale), color(_color), thickness(_thickness), lineType(_lineType), bottomLeftOrigin(_bottomLeftOrigin) {
  }

  void draw(cv::Mat& image, const cv::Point position, const std::string str)
  {
    cv::putText(image, str, position, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
  }
};

int export_kpt_list_to_file(
    const std::string &filename,
	std::list<SiftKeypoint> & kpt_list,
	bool bIncludeDescpritor)
{
  std::ofstream fs;

  try
  {
    fs.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
    fs.open(filename, std::ios::out);

    fs << kpt_list.size() << "\t128\n";

    std::list<SiftKeypoint>::iterator it;
    for (it = kpt_list.begin(); it != kpt_list.end(); it ++){
      fs << it->octave << "\t" << it->layer << "\t" << it->r << "\t" << it->c << "\t" << it->scale << "\t" << it->ori;
      if (bIncludeDescpritor){
        for (int i = 0; i < 128; i ++){
    	  fs << (int)(it->descriptors[i]) << "\t";
    	}
      }
      fs << "\n";
    }
  }
  catch(const std::ofstream::failure &e)
  {
    ARM_COMPUTE_ERROR("Writing %s: (%s)", filename.c_str(), e.what());
  }

  return 0;
}

int draw_kpt_list_on_image(
	const char * file,
	cv::Mat & _inimg,
	std::list<SiftKeypoint> & kpt_list)
{
  ////////////////////////////////////////////////////////////////////////
  // Draw keypoints
  ////////////////////////////////////////////////////////////////////////
  std::list<SiftKeypoint>::iterator it;
  int r, c;
  // * cR:
  // * radius of the circle
  // * cR = sigma * 4 * (2^O)
  int cR;
  for(it = kpt_list.begin(); it != kpt_list.end(); it ++)	{
    // derive circle radius cR
    cR = (int) it->scale;
    if(cR <= 1){ // avoid zero radius
      cR = 1;
    }
    r = (int) it->r;
    c = (int) it->c;
    int thickness = 1;
    int lineType = 8;
    cv::Point center(c, r);
    cv::circle( _inimg,
	            center,
	            cR,
	            cv::Scalar( 0, 0, 255 ),
	            thickness,
	            lineType );
    cv::circle( _inimg,
	            center,
	            cR+1,
	            cv::Scalar( 0, 0, 255 ),
	            thickness,
	            lineType );
    float ori = it->ori;
    int xe = (int) (c + cos(ori)*cR), ye = (int) (r + sin(ori)*cR);
    cv::Point start(c, r);
    cv::Point end(xe, ye);
    cv::line( _inimg,
	          start,
	          end,
	          cv::Scalar( 0, 0, 0 ),
	          thickness,
	          lineType );
  }
  // Create the Text instance.
  Text text(CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  text.draw(_inimg, cv::Point(30, 30), std::to_string(work_fps) + "  detection time");

  // Generate output image with keypoints drawing
  const std::string img_kpt_filename = std::string(ALGNAME) + std::string(file) + "_psift_output.ppm";
  cv::imwrite(img_kpt_filename.c_str(), _inimg);
  return 0;
}

// Combine two images horizontally
int combine_image(
  cv::Mat & _outimg,
  const cv::Mat & _inimg1,
  const cv::Mat & _inimg2)
{
  cv::Size s1 = _inimg1.size();
  int w1 = s1.width, h1 = s1.height;

  cv::Size s2 = _inimg2.size();
  int w2 = s2.width, h2 = s2.height;

  int dstW = w1 + w2;
  int dstH = (std::max)(h1, h2);

  _outimg.create(cv::Size(dstW, dstH), CV_8UC1);

  unsigned char * srcData1 = _inimg1.data;
  unsigned char * srcData2 = _inimg2.data;
  unsigned char * dstData = _outimg.data;

  for (int r = 0; r < dstH; r ++){
    if (r < h1){
      memcpy(dstData, srcData1, w1 * sizeof(unsigned char));
    }else{
      memset(dstData, 0, w1 * sizeof(unsigned char));
    }
    dstData += w1;

    if (r < h2){
      memcpy(dstData, srcData2, w2 * sizeof(unsigned char));
    }else{
      memset(dstData, 0, w2 * sizeof(unsigned char));
    }
    dstData += w2;
    srcData1 += w1;
    srcData2 += w2;
  }
  return 0;
}

// Draw match lines between matched keypoints between two images.
int draw_match_lines_to_ppm_file(
	const std::string &filename,
    cv::Mat & _outimg,
    const cv::Mat & _inimg1,
	const cv::Mat & _inimg2,
    std::list<MatchPair> & match_list)
{
  combine_image(_outimg, _inimg1, _inimg2);

  cv::Size s = _inimg1.size();

  int thickness = 1;
  int lineType = 8;
  std::list<MatchPair>::iterator p;
  for (p = match_list.begin(); p != match_list.end(); p ++){
    MatchPair mp;
    mp.r1 = p->r1, mp.c1 = p->c1;
    mp.r2 = p->r2, mp.c2 = p->c2 + s.width;
    cv::Point start(mp.c1, mp.r1);
    cv::Point end(mp.c2, mp.r2);
    cv::line( _outimg,
	          start,
	          end,
	          cv::Scalar( 0, 0, 0 ),
	          thickness,
	          lineType );
  }

  // Create the Text instance.
  Text text(CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  text.draw(_outimg, cv::Point(30, 60), std::to_string(work_fps) + "  matching time");

  cv::imwrite(filename.c_str(), _outimg);

  return 0;
}


int main(int argc, char* argv[])
{
  const char *file1 = "img1.pgm";
  const char *file2 = "img2.pgm";


  ////////////////////////////////////////////////////////////////////////
  // Create the named window.
  ////////////////////////////////////////////////////////////////////////
  const std::string window_title = "ne_ezsift";
  cv::namedWindow(window_title, CV_WINDOW_NORMAL);

  cv::Mat curRGBImg1, curRGBImg2;
  cv::Mat curGrayImgIn1, curGrayImgIn2;
  cv::Mat curGrayImgMatch;

  // Read the input image file
  curRGBImg1 = cv::imread(file1, CV_LOAD_IMAGE_COLOR);
  curRGBImg2 = cv::imread(file2, CV_LOAD_IMAGE_COLOR);

  // Check for invalid input
  if(! curRGBImg1.data ){
    std::cout <<  "Could not open or find the image 1" << std::endl;
    return -1;
  }
  if(! curRGBImg2.data ){
    std::cout <<  "Could not open or find the image 2" << std::endl;
    return -1;
  }

  // Convert the color image to gray scale image
  cv::Size size(640,480);//the dst image size,e.g.640x480
  cv::cvtColor(curRGBImg1, curGrayImgIn1, CV_BGR2GRAY);
  cv::resize(curGrayImgIn1,curGrayImgIn1,size);//resize image
  cv::cvtColor(curRGBImg2, curGrayImgIn2, CV_BGR2GRAY);
  cv::resize(curGrayImgIn2,curGrayImgIn2,size);//resize image

  std::list<SiftKeypoint> kpt_list1, kpt_list2;

  ////////////////////////////////////////////////////////////////////////
  // Compute SIFT
  ////////////////////////////////////////////////////////////////////////
  {
	SIFTType ezSIFT; // Instantiate the NEON EZSIFT
    ezSIFT.init(curGrayImgIn1);    // Init the NEON EZSIFT parameters
    WorkBegin();
    ezSIFT.sift(kpt_list1);        // Perform NEON EZSIFT computation
    WorkEnd();
    // Generate keypoints list
    const std::string img_kpt_filename = std::string(ALGNAME) + std::string(file1) + "_psift_key.key";
    export_kpt_list_to_file(img_kpt_filename, kpt_list1, false);
    std::cout << "\nImage1 total keypoints number: \t\t" << kpt_list1.size() << "\n";
    draw_kpt_list_on_image(
         file1,
         curGrayImgIn1,
         kpt_list1);
  }

  {
	SIFTType ezSIFT; // Instantiate the NEON EZSIFT
    ezSIFT.init(curGrayImgIn2);    // Init the NEON EZSIFT parameters
    WorkBegin();
    ezSIFT.sift(kpt_list2);        // Perform NEON EZSIFT computation
    WorkEnd();
    // Generate keypoints list
    const std::string img_kpt_filename = std::string(ALGNAME) + std::string(file2) + "_psift_key.key";
    export_kpt_list_to_file(img_kpt_filename, kpt_list2, false);
    std::cout << "\nImage2 total keypoints number: \t\t" << kpt_list2.size() << "\n";
    draw_kpt_list_on_image(
         file2,
         curGrayImgIn2,
         kpt_list2);
  }

  // Match keypoints.
  std::list<MatchPair> match_list;
  WorkBegin();
  arm_compute::match_keypoints(kpt_list1, kpt_list2, match_list);
  WorkEnd();

  // Draw result image.
  {
    const std::string img_match_filename = std::string(ALGNAME) + "A_B_matching.ppm";
    draw_match_lines_to_ppm_file(img_match_filename, curGrayImgMatch, curGrayImgIn1, curGrayImgIn2, match_list);
    std::cout << "# of matched keypoints: " << match_list.size() << "\n";
  }

  char kb_input = 0;
  const int keywait_ms = 0; // ms

  ////////////////////////////////////////////////////////////////////////
  // Main loop
  //     - Show the streaming image
  //     - Exit the main loop; Hit the 'q' key.
  ////////////////////////////////////////////////////////////////////////
  while (kb_input != 'q') {
	// Show the input image
    cv::imshow(window_title, curGrayImgMatch);
    // Waiting for pressing a key by user
    kb_input = cv::waitKey(keywait_ms);
  }

  return 0;
}
