# [Visual Test-time Scaling for GUI Agent Grounding](https://arxiv.org/abs/2505.00684)

<a href="https://arxiv.org/abs/2505.00684"><img src="https://img.shields.io/badge/arXiv-2505.00684-b31b1b.svg" height=20.5></a>

[Tiange Luo](https://tiangeluo.github.io/), [Lajanugen Logeswaran](https://lajanugen.github.io/)&dagger;, [Justin Johnson](https://web.eecs.umich.edu/~justincj)&dagger;, [Honglak Lee](https://web.eecs.umich.edu/~honglak/)&dagger;

We release our ScreenSpot-Pro code for both UI-TARS and Qwen2.5-VL. All hyperparameters and prompts are not carefully tuned. Due to company policy, the release of the WebVoyager-related code is no longer permitted.

## ScreenSpot-Pro

Please first download the data from ScreenSpot-Pro [Hugging Face](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/main) and put `images` and `annotations` folders under the same directory with code. Then, launch inference servers for different models (commands are listed below; the model names and ports have already been mapped inside the code). Finally, run `bash run_ss_pro_xxx.sh`.

You can use `summarize_results.py` to output ScreenSpot-Pro results categorically, following the order presented in our Table 1.
One Example:
```bash
python summarize_results.py results/qwen25vl_RegionFocus.json results/uitars_RegionFocus.json

# output: 
# results/qwen25vl_RegionFocus.json 76.0 & 26.2 & 51.8 & 75.8 & 30.8 & 56.9 & 72.1 & 28.1 & 61.3 & 86.8 & 37.3 & 65.4 & 86.4 & 60.4 & 80.4 & 74.8 & 38.2 & 58.2 & 78.5 & 34.3 & 61.6 1581
# results/uitars_RegionFocus.json ...
```

You can turn on `--debug` inside `eval_screenspot_pro_RegionFocus.py` to save intermediate RegionFocus step images, such as image-as-map stars for judgment, zoom-ins, and projecting zoomed-in predictions back onto the original input.

<details>
<summary>Command for launching UI-TARS-72B & -7B</summary>
  
Please first set up your `HUGGINGFACE_PATH` and `HF_TOKEN` in the below commands.

```bash
HUGGINGFACE_PATH='the local directory to cache Hugging Face models'
HF_TOKEN='your_HF_token'
docker run --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -p 8100:8100 \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  --env "TORCH_USE_CUDA_DSA=1" \
  --env "CUDA_LAUNCH_BLOCKING=1" \
  -v $HUGGINGFACE_PATH:/root/.cache/huggingface \
  vllm/vllm-openai:v0.6.6 \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --gpu_memory_utilization 0.9 \
  --model bytedance-research/UI-TARS-72B-DPO \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --limit-mm-per-prompt image=5 \
  --port 8100
```

```bash
HUGGINGFACE_PATH='the local directory to cache Hugging Face models'
HF_TOKEN='your_HF_token'
docker run --runtime nvidia --gpus '"device=0,1,2,3"' --ipc=host \
  -p 8200:8200 \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  --env "TORCH_USE_CUDA_DSA=1" \
  -v $HUGGINGFACE_PATH:/root/.cache/huggingface \
  vllm/vllm-openai:v0.6.6 \
  --max-model-len 16384 \
  --max-num-seqs 2048 \
  --gpu_memory_utilization 0.9 \
  --model bytedance-research/UI-TARS-7B-DPO \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt image=5 \
  --dtype bfloat16 \
  --port 8200
```
</details>


<details>
<summary>Command for launching Qwen2.5-VL-72B & -7B</summary>

Please first install https://github.com/QwenLM/Qwen-Agent. 

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --port 8300  --dtype bfloat16   --limit-mm-per-prompt '{"images": 5}'   --tensor-parallel-size 8

export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve Qwen/Qwen2.5-VL-7B-Instruct   --port 8400   --dtype bfloat16   --limit-mm-per-prompt '{"images": 5}'   --tensor-parallel-size 4
```

</details>


## Citation Information

If you find our code or paper useful, please consider citing:

```
@article{luo2025visual,
      title={Visual Test-time Scaling for GUI Agent Grounding},
      author={Luo, Tiange and Logeswaran, Lajanugen and Johnson, Justin and Lee, Honglak},
      journal={arXiv preprint arXiv:2505.00684},
      year={2025},
}
```
