#!/bin/bash
set -e

echo "🚀 Starting Local GGUF Quantization Pipeline..."

# 1. Clone llama.cpp if not exists
if [ ! -d "llama.cpp" ]; then
    echo "📦 Cloning llama.cpp (Shallow clone to avoid network drops)..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# 2. Install requirements
echo "📦 Installing Requirements..."
../venv/bin/pip install gguf sentencepiece

# 3. Convert HF to F16 GGUF
# echo "🔄 Converting HuggingFace model to F16 GGUF (This takes ~1 min)..."
# if [ ! -f "../models/merged_model/tokenizer.model" ]; then
#     echo "Downloading missing tokenizer.model..."
#     curl -sLo ../models/merged_model/tokenizer.model https://huggingface.co/microsoft/phi-3-mini-4k-instruct/resolve/main/tokenizer.model
# fi
# ../venv/bin/python3 convert_hf_to_gguf.py ../models/merged_model --outfile ../models/merged_model/model-f16.gguf --outtype f16

# 4. Compile quantization tool
echo "🔨 Compiling llama-quantize tool with CMake (This takes ~1-2 mins)..."
mkdir -p build
cd build
cmake ..
cmake --build . --config Release -j
cd ..

# 5. Quantize to Q4_K_M
echo "🗜️ Quantizing F16 to Q4_K_M (This takes ~1 min)..."
./build/bin/llama-quantize ../models/merged_model/model-f16.gguf ../models/merged_model/model-q4_k_m.gguf q4_k_m

echo "✅ Quantization Complete!"
echo "Your ultra-fast, 2.2GB model is ready at: models/merged_model/model-q4_k_m.gguf"
