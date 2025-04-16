# Cached-Augemented-Generation-CAG-
# 🧠 Cache-Augmented Generation (CAG) with Mistral-7B

A practical demo of Cache-Augmented Generation using the `mistralai/Mistral-7B-Instruct-v0.1` model.  
This approach boosts LLM efficiency by **preloading knowledge** into the model's KV cache, avoiding real-time document retrieval.

> 📘 Article: [Medium Tutorial](https://medium.com/@ronantech/cache-augmented-generation-cag-in-llms-a-step-by-step-tutorial-6ac35d415eec)  
> 📓 Google Colab: [Open in Colab](https://colab.research.google.com/drive/1-0eKIu6cGAZ47ROKQaF6EU-mHtvJBILV?usp=sharing)

---

## 🚀 Features

- Uses Mistral-7B-Instruct for QA
- Preloads content (including Excel data) into the KV cache
- Avoids real-time document lookup
- Supports Google Colab for GPU-based inference
- Customizable knowledge base

---

## 🛠️ Setup

### ✅ Prerequisites

- Python 3.8+
- Hugging Face account with access to [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- PyTorch (with CUDA support if available)
- Transformers, Accelerate, and Optimum

### 🧪 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/cache-augmented-generation.git
cd cache-augmented-generation

pip install -r requirements.txt

Authentication (Hugging Face Token)
Before loading the model in Colab or Jupyter, run:

python
Copy
Edit
from huggingface_hub import login
login(token="your_huggingface_token")
Or set your token as an environment variable:

bash
Copy
Edit
export HF_TOKEN=your_token_here
💡 Usage
Run Demo (Locally or in Colab):
python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForCausalLM
from your_kv_utils import get_kv_cache

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)

# Example context + question
system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
<|user|>
Context:
{doc_text}
Question:
""".strip()

ronan_cache = get_kv_cache(model, tokenizer, system_prompt)
📦 Quantization (Optional)
To reduce memory usage (good for Colab):

bash
Copy
Edit
pip install optimum
