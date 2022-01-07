# PMPP Compositional GPU

Record Table

|flag|function|time|
|---|---|---|
|NK|nop-kernel | |
|MIK|min-kernel| |
|MAK|max-kernel| |
|SPK|scalarProduct-kernel| |
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

