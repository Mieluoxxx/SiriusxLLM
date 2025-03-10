#!/bin/bash
###
 # @Author: Morgan Woods weiyiding0@gmail.com
 # @Date: 2025-03-09 17:41:04
 # @LastEditors: Morgan Woods weiyiding0@gmail.com
 # @LastEditTime: 2025-03-10 15:36:35
 # @FilePath: /SiriusxLLM/run.sh
 # @Description: 
### 

/home/moguw/workspace/SiriusxLLM/build/demo/llama2 \
--checkpoint_path /home/moguw/workspace/SiriusxLLM/tmp/stories42M.bin \
--tokenizer_path /home/moguw/workspace/SiriusxLLM/tmp/tokenizer.model \
--quantized false \
--prompt "long long ago, there was a king." \
--use_cuda true # 如果没有CUDA就注释了