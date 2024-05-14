# Report

## Task 0

以下为不同的编译器参数下的程序性能：

|编译参数|Elapsed time (seconds)|Performance|
|---|---|---|
|`-O0`|$0.9981$|$0.2689$|
|`-O1`|$0.3410$|$0.7872$|
|`-O2`|$0.3321$|$0.8084$|
|`-O3`|$0.0496$|$5.4093$|
|`-fast`|$0.0395$|$6.8003$|

查阅手册后，可得到每种参数对应的部分几个优化技术：

|编译参数|优化技术|
|---|---|
|`-O0`|不使用任何优化技术|
|`-O1`|使用global optimization，包括data-flow analysis, code motion, strength reduction, test replacement, split-lifetime analysis, instruction scheduling等|
|`-O2`|使用内联函数, Intra-file interprocedural optimization, onstant propagation, copy propagation, dead-code elimination, global register allocation, global instruction scheduling and control speculation, 循环展开等|
|`-O3`|使用 `-O2` 优化并使用更积极的循环转换，如Fusion, Block-Unroll-and-Jam, collapsing IF statements等|
|`-fast`|使用一些激进的优化使运行速率最大化，在Linux上会启用 `-ipo, -O3, -no-prec-div,-static` 等编译开关|

## Task 1

以下为设置不同的循环展开程度时程序的性能：

|`UNROLL_N`|Elapsed time (seconds)|Performance|
|---|---|---|
|`1`|$2.0508$|$15.9778$|
|`2`|$1.9659$|$16.6679$|
|`4`|$1.8872$|$17.3634$|
|`8`|$1.7693$|$18.5201$|
|`16`|$1.8256$|$17.9493$|

循环展开可以减少循环变量的比较次数和分支跳转次数。
