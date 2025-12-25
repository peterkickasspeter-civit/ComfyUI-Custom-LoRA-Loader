# ComfyUI-Custom-LoRA-Loader

Credits to https://www.reddit.com/r/StableDiffusion/comments/1pthc20/block_edit_save_your_loras_in_comfyui_lora_loader/

This is my first custom node and I worked with ChatGPT and Gemini. You can clone it in your custom_nodes folder and restart your comfyui
Workflow: https://pastebin.com/TXB7uH0Q
There are 2 nodes here
LoRA Loader Custom (Stackable + CLIP)
This is where you load your LoRA and specify the weight and the steps you will use that weight, something like
Style LoRA:
2 : 0.8   # Steps 1-2: Get the style and composition
3 : 0.4   # Steps 3-5: Slow down and let Character LoRA take over
9 : 0.0   # Steps 6-14: Turn it off

Character LoRA:
4 : 0.6   # Steps 1-4: Lower weight to help the style LoRA with composition
2 : 0.85  # Steps 5-6: Ramp up so we have the likeness
7 : 0.9   # Steps 7-13: Max likeness steps
1 : 0     # Steps 14: OFF to get back some Z-Image skin texture
You can also use Json like [{"duration": 0.1, "strength": 0.5}, ...]
You can connect n number of loras (I only tested with a Style LoRA - Amateur Photography and Character LoRA's)
Apply Hooks To Conditioning (append)
Positive and Negative and the hooks from the lora loader connects to this and they go to your ksampler
