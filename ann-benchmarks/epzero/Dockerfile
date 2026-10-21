FROM ann-benchmarks

# Epzero picks its SIMD kernels (AVX2, AVX-512, NEON) at run time, so the published wheel is fast on
# any CPU. Until a release is on PyPI, build it from source instead:
#   RUN apt-get update && apt-get install -y git cmake g++ && \
#       pip install "git+https://github.com/Lazeeez/Epzero.git#subdirectory=engine"
RUN pip install epzero
RUN python3 -c "import epzero; print('epzero', epzero.__version__, epzero.simd_level())"
