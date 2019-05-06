#pragma once
using byte = unsigned char;

extern "C"
void DD2Proj_multiGPUs(float x0, float y0, int DNU,
    float* xds, float* yds, float imgXCenter, float imgYCenter,
    float* hangs, int PN, int XN, int YN, int SLN, float* hvol, float* hprj,
    float dx, byte* mask, int gpuNum, int* startIdx);


extern "C"
void DD2Back_multiGPUs(
    float x0, float y0,
    int DNU,
    float* xds, float* yds,
    float detCntIdx,
    float imgXCenter, float imgYCenter,
    float* hangs, int PN,
    int XN, int YN, int SLN,
    float* hvol, float* hprj,
    float dx,
    byte* mask, int gpuNum, int* startIdx);


extern "C"
void HelicalToFan(
    float* Proj,  					// rebinned projection; in order (Channel, View Index, Slice Index)  TODO: need permute after call by MATLAB
    float* Views, 					// rebinned views (View Index, Slice Index) TODO: need permute after call by MATLAB
    float* proj,  					// raw projection data In order : (Height Index, Channel Index, View Index(Total View))
    float* zPos,  					// sampling position
    const int SLN,                  // slice number
    const float SD, 				// source to detector distance
    const float SO,					// source to iso-center distance
    const float BVAngle,        	// The begin view
    const int DetWidth,         	// number of detector columns
    const int DetHeight,        	// number of detector rows
    const float PerDetW,        	// Detector cell size along channel direction
    const float PerDetH,        	// Detector cell size along bench moving direction
    const int DefTimes,         	// Number of views per rotation
    const float DetCenterW,     	// Detector Center Index
    const float SpiralPitchFactor 	// Pitch defined in SIEMENS
);