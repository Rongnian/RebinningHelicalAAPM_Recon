/*
 * Wake Forest Health Sciences & University of Massachusetts Lowell
 * Organization: 
 *  Wake Forest Health Sciences
 *
 * DD2_multislice_mex.cpp
 * Matlab mex gateway routine for the GPU based multi-slice distance-driven
 * fan-beam projector
 *
 * author: Rui Liu (Wake Forest Health Sciences)
 * date: 2016.04.06
 * version: 1.0
 */

#include "mex.h"
#include "matrix.h"

#include "DD2MultiGPU.h"

#include <cstring>
#include <iostream>

typedef unsigned char byte;


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    float x0 = *((float*)mxGetData(prhs[0]));
    float y0 = *((float*)mxGetData(prhs[1]));
    int DNU = *((int*)mxGetData(prhs[2]));
    float* xds = (float*)mxGetPr(prhs[3]);
    float* yds = (float*)mxGetPr(prhs[4]);
    float imgXCenter = *((float*)mxGetData(prhs[5]));
    float imgYCenter = *((float*)mxGetData(prhs[6]));
    float* hangs = (float*)mxGetPr(prhs[7]);
    int PN = *((int*)mxGetData(prhs[8]));
    int XN = *((int*)mxGetData(prhs[9]));
    int YN = *((int*)mxGetData(prhs[10]));
    int SLN = *((int*)mxGetData(prhs[11]));
    float* hvol = (float*)mxGetPr(prhs[12]);
    float dx = *((float*)mxGetData(prhs[13]));
    byte* mask = (byte*)mxGetPr(prhs[14]);
    int gpuNum = *((int*)mxGetData(prhs[15]));
    int* startIdx = (int*)mxGetPr(prhs[16]);
       
    const mwSize dims[] = {SLN, DNU, PN};
    //std::cout<<SLN<<" "<<DNU<<" "<<PN<<"\n";
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    float* hprj = (float*)mxGetPr(plhs[0]);
    
    DD2Proj_multiGPUs(x0, y0, DNU, xds, yds, imgXCenter, imgYCenter,
 		hangs, PN, XN, YN, SLN, hvol, hprj, dx, mask, gpuNum, startIdx);
}

