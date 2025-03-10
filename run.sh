#!/bin/bash

/home/moguw/workspace/siriusx-infer/build/demo/llama2 \
--checkpoint_path /home/moguw/workspace/siriusx-infer/tmp/stories42M.bin \
--tokenizer_path /home/moguw/workspace/siriusx-infer/tmp/tokenizer.model \
--quantized false \
--prompt "long long ago," \
--use_cuda false # 如果没有CUDA就注释了