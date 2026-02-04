import transformers
import qwen_vl_utils
from transformers import AutoModelForVision2Seq
import cv2, numpy as np, subprocess
import multiprocessing as mp
import pdb, torch, sys


batch_size=int(sys.argv[1])
print('BATCH SIZE IS ', batch_size)
llm_path = 'data/checkpoints/llm_checkpoints/3view_sidebyside/checkpoint-3600'

def get_gpu_with_lowest_memory_util():
    # Run the nvidia-smi command to get the GPU status
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Parse the result and find the GPU with the lowest memory usage
    gpu_info = result.stdout.strip().split('\n')
    
    min_util_gpu = None
    min_util = float('inf')  # Initialize with a large number
    
    for gpu in gpu_info:
        index, memory_used, memory_total = map(int, gpu.split(', '))
        memory_util = memory_used / memory_total  # Memory utilization ratio
        
        if memory_util < min_util:
            min_util = memory_util
            min_util_gpu = index

    return min_util_gpu, min_util

# Find the GPU with the lowest memory utilization
LLM_GPU_ID, gpu_utilization = get_gpu_with_lowest_memory_util()

if LLM_GPU_ID is not None:
    print(f"GPU with the lowest memory utilization: GPU-{LLM_GPU_ID} with {gpu_utilization * 100:.2f}% usage")
else:
    raise Exception('cannot contain LLM ')
    print("No GPUs found.")

# Initialize LLM once
print(f"Initializing LLM on GPU {LLM_GPU_ID}...")
llm = AutoModelForVision2Seq.from_pretrained(
    llm_path,
    torch_dtype='bfloat16',
    device_map="auto",
    trust_remote_code=True,
    quantization_config=None,
)
llm = llm.eval()
processor = transformers.AutoProcessor.from_pretrained(llm_path)

TASK_DESCRIPTION = "Pick and place an object from the sink to the plate on the counter"
SYSTEM_PROMPT = f"""You are an expert roboticist tasked to compare a side-by-side of 2 images from a robot demonstration and determine which side shows more progress toward completing the task.
The robot task is: {TASK_DESCRIPTION}
You will be given a side-by-side of 2 images from the same demonstration, and you need to identify how much closer or behind in task completion is the right image compared to the left."""
problem = """Look at these two side-by-side images of a robot performing the task.

Left side image: Shows the robot at one point during the task.
Right side image: Shows the robot at another point during the task.

Task: Compare the two images and determine the relative progress difference.
- If the right image shows more progress toward task completion, respond with a positive number (1 to 100)
- If the right image shows less progress toward task completion, respond with a negative number (-1 to -100)
- If both images show equal progress, respond with 0

The number should represent how much more or less progress the right image shows compared to the left."""

extract_function = float

PROMPTS = {
    "system_prompt": SYSTEM_PROMPT,
    "problem": problem,
    'extract_function': extract_function,
}

overlay_image = 'data/checkpoints/dp_model/epoch=1100-val_loss=0.037/oct14/PnPSinkToCounter_mg_val_kbpckt_firsthalf_101517567__choose_sample_True_num_samples_5_ws_0-27/videos/env_24_step_1_sample_3_3view_last_frame_and_env_24_step_1_sample_4_3view_last_frame_overlay.png'
conversation = [
    {"role": "system", "content": [{"type": "text", "text": PROMPTS['system_prompt']}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": overlay_image},
            {"type": "text", "text": PROMPTS['problem']},
        ],
    },
]
batch_conversations = [list(conversation) for _ in range(batch_size)]
pdb.set_trace()
batch_texts = [
    processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    for conv in batch_conversations
]
batch_image_inputs, _ = zip(
    *[qwen_vl_utils.process_vision_info(conv) for conv in batch_conversations]
)
batch_image_inputs = list(batch_image_inputs)
inputs = processor(
    text=batch_texts,
    images=batch_image_inputs,
    padding=True,
    return_tensors="pt",
).to(llm.device)
pdb.set_trace()
with torch.no_grad():
    generated_ids = llm.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
batch_outputs = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print('DONE1', batch_outputs, 'with batchsize', str(len(batch_conversations)))

with torch.no_grad():
    generated_ids = llm.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
batch_outputs = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print('DONE2', batch_outputs)

with torch.no_grad():
    generated_ids = llm.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
batch_outputs = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print('DONE3', batch_outputs)

