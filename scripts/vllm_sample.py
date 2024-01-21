from vllm import LLM
llm = LLM("/root/model_assets/Llama-2-7b-chat-hf", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
print(output)