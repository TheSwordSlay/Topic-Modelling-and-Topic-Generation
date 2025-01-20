from torch import bfloat16
import transformers

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
)

from torch import cuda

model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()

# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)

system_prompt_qwen2 = """<|im_start|>system
You are a helpful, respectful and honest assistant for labeling topics.<|im_end|>"""

example_prompt_qwen2 = """<|im_start|>user
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|im_end|>"""

prompt_qwen2 = system_prompt_qwen2 + example_prompt_qwen2
res = generator(prompt_qwen2)
res_split = res[0]["generated_text"].replace(prompt_qwen2+"system", "")
def strip_first_line(s):
    lines = s.splitlines()
    lines.pop(0)
    return '\n'.join(lines)
label = strip_first_line(res_split)
print(label)