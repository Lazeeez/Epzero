FROM ann-benchmarks

# Epzero picks its SIMD kernels (AVX2, AVX-512, NEON) at run time, so the published wheel is fast on
# any CPU. Pinned, so a run can be reproduced.
RUN pip install epzero==0.4.0
RUN python3 -c "import epzero; print('epzero', epzero.__version__, epzero.simd_level())"
