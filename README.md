# PMPP Compositional GPU

Record Table

|flag|function|time (of vector with 1 element/100 elements/10000 elements/65536 elements)|
|---|---|---|
|NK|nop-kernel | 2.842e-05 sec|
|MIK|min-kernel| 4.214e-04 sec /4.235e-04 sec/4.800e-04 sec/6.180e-04  sec|
|MAK|max-kernel| 4.193e-04 sec /4.210e-04 sec/ 4.543e-04 sec/ 5.930e-04 sec|
|SPK|scalarProduct-kernel| 4.26e-04 sec/4.233e-04/5.055e-04 sec/8.760e-04 sec|
|MD|matrix_double_kernel |4.494e-04 sec/5.459e-04/8.197e-03 sec(1000)/1.264 sec(10000) /|
|N|Nopper |1.774e-06 sec|
|I|Increaser|2.227e-06 sec/1.926e-06 sec/1.872e-06 sec/1.796e-05 sec|
|QS|QuickSorter|2.059e-06 sec/1.621e-06 sec/7.840e-05 sec/6.223e-04 sec|
|RA|ReduceAddVector|2.548e-06 sec/3.679e-06/1.849e-05/1.117e-04|
|RM|ReduceMin|2.031e-06 sec|
|RO|Reorderer|1.753e-06 sec/2.401e-06 sec/6.801e-05 sec/4.788e-04 sec|
|SS|SelectionSorter|2.121e-06 sec/2.079e-06 sec/4.509e-03 sec/0.205 sec|

|function_A&function_B|time(A)|time(B)|size(vec_A)|size(vec_B)|time|
|---|---|---|---|---|---|
|MD + N||||||
|NK + MIK||||||
|MAK + SPK||||||
|NK + SPK||||||
|MAK + MIK||||||
|NK + N||||||
|MIK + I||||||
|MAK + QS||||||
|SPK + RA||||||
|NK + RM||||||
|MIK + RO||||||
|MAK + SS||||||

