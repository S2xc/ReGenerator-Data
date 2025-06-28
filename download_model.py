from llama_cpp import Llama
model_path = "/Users/s2xdeb/Desktop/ReGd/ReGenerator_Datasets/google_model_llm/gemma-3-4b-it-Q4_K_M.gguf"
llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1)
print("Модель загружена офлайн!")


