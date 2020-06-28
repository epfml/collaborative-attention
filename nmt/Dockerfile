FROM mlbench/mlbench-pytorch-base:pytorch-v1.5.0

# -------------------- Debug --------------------
RUN git clone https://github.com/NVIDIA/apex && cd apex && git checkout f3a960f && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" ./ && \
    cd ..

## Horovod
RUN HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_BROADCAST=MPI \
    HOROVOD_WITH_PYTORCH=1\
    HOROVOD_WITH_MPI=1\
    pip install --no-cache-dir horovod

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV MLBENCH_INSTALL_DIR /conda/lib/python3.7/site-packages/mlbench_core
ENV TRANSFORMER_MODULES models/pytorch/transformer/modules
ENV STRIDED_BATCHED_GEMM strided_batched_gemm.cpython-37m-x86_64-linux-gnu.so

RUN python $MLBENCH_INSTALL_DIR/install_cuda_extensions.py build && \
    mv build/lib.linux-x86_64-3.7/mlbench_core/$TRANSFORMER_MODULES/$STRIDED_BATCHED_GEMM $MLBENCH_INSTALL_DIR/$TRANSFORMER_MODULES/$STRIDED_BATCHED_GEMM && \
    rm -r build

RUN mkdir /codes
RUN mkdir /codes/utils

ADD ./main.py /codes/main.py
ADD ./collab_multihead_attention.py /codes/collab_multihead_attention.py
ADD ./utils/* /codes/utils/


ENV PYTHONPATH /codes

# Remove empty ld
RUN rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true
