import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import ChatGLM6BTokenizer, ChatGLM6BModel, ChatGLM6BForConditionalGeneration, ChatGLM6BConfig

print("loading")

tokenizer = AutoTokenizer.from_pretrained("/raid/sxx/GLM6B/hfg_ckpt/")
# tokenizer.save_pretrained("/raid/sxx/GLM6B/hfg_ckpt/")
# tokenizer.push_to_hub("THUDM/ChatGLM-6B", private=True, use_auth_token="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")

# model = ChatGLM6BForConditionalGeneration(ChatGLM6BConfig()).from_pretrained("/raid/sxx/GLM6B/hfg_ckpt/")
# state_dict = torch.load("/mnt/vepfs/workspace/sxx/checkpoints/qa-glm-6b-sft-v0.7.5/hgf/mp_rank_00_model_states.pt")
# model.load_state_dict(state_dict)
# model.save_pretrained("/mnt/vepfs/workspace/sxx/checkpoints/qa-glm-6b-sft-v0.7.5/hfg_ckpt/")

model = AutoModelForSeq2SeqLM.from_pretrained("/raid/sxx/GLM6B/hfg_ckpt/")
model.save_pretrained("/raid/sxx/GLM6B/hfg_ckpt/")
model = model.half().cuda()
model.eval()


# # Inference
print("inference")

input_ids = tokenizer(["清华大学[gMASK]"], return_tensors="pt", padding=True)
input_ids = input_ids.to('cuda')

outputs = model.generate(**input_ids, max_length=512, num_beams=2)
print(tokenizer.decode(outputs.tolist()))

response, history = model.chat(tokenizer=tokenizer, query="清华大学")
print(response)
response, history = model.chat(tokenizer=tokenizer, query="上海在哪里？", history=history)
print(response)
response, history = model.chat(tokenizer=tokenizer, query="北京大学在这里吗？", history=history)
print(response)
print(history)
