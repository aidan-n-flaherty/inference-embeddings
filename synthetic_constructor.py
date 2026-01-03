from llama_cpp import Llama, LlamaRAMCache, LlamaGrammar
import json
import re
import random
import pickle
import os
import traceback

import re
from datasets import load_dataset

if __name__ == "__main__":
	validation = True

	dataset = load_dataset("liar")

	llm = Llama(model_path='models_llm/openhermes-2.5-mistral-7b.Q6_K.gguf', chat_format="chatml", embedding=True, verbose=False, n_ctx=4096, n_gpu_layers=-1)

	def complete(text):
		prompt = f"""You are given a sentence: "{text}"

	Generate 5 sentences that directly support this statement, and 5 sentences that directly contradicts this statement, formatted as a Python list. Don't output anything else."""

		return llm.create_chat_completion(
			messages = [
				{"role": "user", "content": prompt}
			], max_tokens=512, stream=False, temperature=0.2)["choices"][0]["message"]["content"]

	data = []

	if os.path.exists(f"synthetic_data_{'validation' if validation else 'train'}.pkl"):
		with open(f"synthetic_data_{'validation' if validation else 'train'}.pkl", "rb") as f:
			data = pickle.load(f)

	i = len(data)

	for item in (dataset["train"] if not validation else dataset["validation"]):
		if any(item["statement"] == it["statement"] for it in data):
			continue

		output = complete(item["statement"])
		
		try:
			arr = [out.strip() for out in re.split(r'[\n\[\]]', output) if out.strip() != ""]
			arr = [out[:-1] if out[-1] == ',' else out for out in arr]
			arr = [out for out in arr if out != ""]
			arr = [out[1:] if out[0] == '"' else out for out in arr]
			arr = [out[:-1] if out[-1] == '"' else out for out in arr]
		except:
			print(output)
			print([out.strip() for out in re.split(r'[\n\[\]]', output) if out.strip() != ""])
			print(traceback.format_exc())
		
		if len(arr) != 10:
			continue

		data.append({
			"statement": item["statement"],
			"supported": arr[:5],
			"refuted": arr[5:]
		})

		i += 1
		print(f"Iteration {i}/{len(dataset['train'] if not validation else dataset['validation'])}")

		if i % 10 == 5:
			with open(f"synthetic_data_{'validation' if validation else 'train'}_tmp.pkl", "wb") as f:
				pickle.dump(data, f)
		
		if i % 10 == 0 or i == len(dataset["train"]):
			with open(f"synthetic_data_{'validation' if validation else 'train'}.pkl", "wb") as f:
				pickle.dump(data, f)