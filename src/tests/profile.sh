./configure  CPPFLAGS="-I/opt/local/OpenBLAS/include/ -I/opt/local/include/boost" LIBS="-L/opt/local/OpenBLAS/lib -lopenblas" && make && ./netreg_benchmarks && gprof netreg_benchmarks  gmon.out >! analysis.txt