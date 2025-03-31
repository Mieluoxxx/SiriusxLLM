#!/bin/bash
###
 # @Author: Morgan Woods weiyiding0@gmail.com
 # @Date: 2025-03-09 17:41:04
 # @LastEditors: Morgan Woods weiyiding0@gmail.com
 # @LastEditTime: 2025-03-18 10:11:22
 # @FilePath: /SiriusxLLM/run.sh
 # @Description: 
###

/home/moguw/workspace/SiriusxLLM/bin/llama2 \
--checkpoint_path /home/moguw/workspace/SiriusxLLM/tmp/stories42M.bin \
--tokenizer_path /home/moguw/workspace/SiriusxLLM/tmp/tinyllama.model \
--quantized false \
--prompt "One day, Lily" \
--use_cuda true # 如果没有CUDA就注释了