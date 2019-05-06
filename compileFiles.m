mex cov_4uint8_to_float.cpp
mex cov_8uint8_to_2floats.cpp
mex cov_uint8_to_double.cpp
mex cov_uint8_to_PhoStat.cpp

mex -lDD2_MultiGPUs.lib -lcudart HelicalToFanFunc_mex.cpp
mex -lDD2_MultiGPUs.lib -lcudart DD2Proj.cpp
mex -lDD2_MultiGPUs.lib -lcudart DD2Back.cpp

pcode CollectImageCfg.m
pcode CollectReconCfg.m
pcode ConvertToReconConf.m
pcode DD2MutiSlices.m
pcode HelicalToFan_routine.m
pcode OSSART_AAPM.m
pcode readProj.m