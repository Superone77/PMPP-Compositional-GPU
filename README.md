# PMPP Compositional GPU

Record Table

|flag|function|time (of vector with 1 element/100 elements/10000 elements/65536 elements)|
|---|---|---|
|NK|nop-kernel | 2.84161e-05 sec|
|MIK|min-kernel| 6.28596e-05 sec /6.48727e-05 sec/1.07273e-04 sec/3.56748e-04  sec|
|MAK|max-kernel| 3.81907e-05 sec /4.10672e-05 sec/ 8.46112e-05 sec/ 2.97976e-04 sec|
|SPK|scalarProduct-kernel| 7.21873e-05 sec/7.00473e-05/1.90116e-04 sec/3.18319e-04 sec|
|N|Nopper |1.774e-06 sec|
|I|Increaser|2.2273e-06 sec/1.9257e-06 sec/1.8722e-06 sec/1.79621e-05 sec|
|QS|QuickSorter|2.0591e-06 sec/1.6206e-06 sec/7.84031e-05 sec/6.2228e-04 sec|
|RA|ReduceAdd||
|RM|ReduceMin||
|RO|Reorderer|1.7526e-06 sec/2.4009e-06 sec/6.80051e-05 sec/4.78822e-04 sec|
|SS|SelectionSorter|2.121e-06 sec/2.0785e-06 sec/4.50931e-03 sec/8.00362e-02 sec|

|function_A&function_B|time(A)|time(B)|size(vec_A)|size(vec_B)|time|
|---|---|---|---|---|---|
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

