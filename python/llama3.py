'''
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-03-31 13:53:09
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-04-02 18:32:08
FilePath: /SiriusxLLM/python/llama3.py
Description: 
'''
# 此文件主要用来比对 c++/cuda 推理的正确性
from modelscope import AutoModelForCausalLM, AutoTokenizer

model = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    trust_remote_code=True,
).eval()
print(model)
inputs = tokenizer("hello", return_tensors="pt")
inputs = inputs.to(model.device)
pred = model.generate(
    **inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.0
)
test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(test)