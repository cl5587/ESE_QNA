from huggingface_hub import snapshot_download 

# snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf", repo_type="model", cache_dir="/scratch/network/cl5587/ESE_QNA/.cache")
snapshot_download(repo_id='cl5587/Llama-2-7b-chat-hf-bnb-4bit', repo_type="model", cache_dir="/scratch/network/cl5587/ESE_QNA/.cache")