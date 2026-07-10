FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install llama.cpp (for llama-server binary)
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /llama.cpp \
    && cd /llama.cpp \
    && cmake -B build \
    && cmake --build build --config Release -j$(nproc) --target llama-server \
    && cp build/bin/llama-server /usr/local/bin/ \
    && rm -rf /llama.cpp

# Copy the benchmark suite
COPY . .

# Install the package
RUN pip3 install -e .

ENTRYPOINT ["spec-bench"]
