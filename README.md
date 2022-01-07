# PMPP Compositional GPU

Record Table

|flag|function|time (of vector with 1 element/100 elements/10000 elements/65536 elements)|
|---|---|---|
|NK|nop-kernel | 2.84161e-05 sec|
|MIK|min-kernel| 6.28596e-05 sec /6.48727e-05 sec/1.07273e-04 sec/3.56748e-04  sec|
|MAK|max-kernel| 3.81907e-05 sec /4.10672e-05 sec/ 8.46112e-05 sec/ 2.97976e-04 sec|
|SPK|scalarProduct-kernel| 7.21873e-05 sec/7.00473e-05/1.90116e-04 sec/3.18319e-04 sec|
|N|Nopper | |
|I|Increaser||
|QS|QuickSorter||
|RA|ReduceAdd||
|RM|ReduceMin||
|RO|Reorderer||
|SS|SelectionSorter||

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

