FROM ann-benchmarks

# Oryx picks its SIMD kernels (AVX2, AVX-512, NEON) at run time, so the published wheel is fast on
# any CPU. Until a release is on PyPI, build it from source instead:
#   RUN apt-get update && apt-get install -y git cmake g++ && \
#       pip install "git+https://github.com/Lazeeez/Oryx.git#subdirectory=engine"
RUN pip install oryxdb
RUN python3 -c "import oryxdb; print('oryxdb', oryxdb.__version__, oryxdb.simd_level())"
