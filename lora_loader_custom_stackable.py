"""
LoRA Loader Custom (Stackable, CLIP-aware, scheduled) for ComfyUI

Updates:
- UI: 'strength' input is now a MULTILINE text box for easier editing.
- PARSER: Supports JSON input OR simple line-by-line format.
- PARSER: Supports comments (#) in the text box.
- FEATURE: Added 'total_steps' input to support absolute step scheduling.

Format Examples:
    1) Simple Steps (Set 'total_steps' to match your KSampler, e.g., 20):
        6 : 0.9     # First 6 steps
        2 : 0.85    # Next 2 steps
        1 : 0.0     # Last 1 step

    2) Percentage (Leave 'total_steps' at 0):
        0.2 : 0.9   # Start strong (20%)
        0.6 : 0.4   # Main section (60%)
        0.2 : 0.0   # Turn off (20%)
"""

import os
import torch
import json
import re

import folder_paths
import comfy.sd
import comfy.hooks
import comfy.utils


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _try_parse_float(s):
    try:
        return float(s)
    except Exception:
        return None

def _parse_duration_schedule(schedule_input):
    """
    Parses the input string into a list of (duration, strength) tuples.
    Supports:
      1. JSON List of Dicts: [{"duration": 0.1, "strength": 0.5}, ...]
      2. JSON List of Lists: [[0.1, 0.5], ...]
      3. Simple Text: "0.1:0.5, 0.2:0.8" or newline separated.
    """
    if schedule_input is None:
        return None

    s = str(schedule_input).strip()
    if not s:
        return None

    # --- 1. Try JSON Mode ---
    if s.startswith("[") or s.startswith("{"):
        try:
            data = json.loads(s)
            segments = []
            if isinstance(data, list):
                for item in data:
                    dur, str_val = None, None
                    # Handle [{"duration":0.1, "strength":0.9}]
                    if isinstance(item, dict):
                        dur = _try_parse_float(item.get("duration"))
                        str_val = _try_parse_float(item.get("strength"))
                    # Handle [[0.1, 0.9]]
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        dur = _try_parse_float(item[0])
                        str_val = _try_parse_float(item[1])
                    
                    if dur is not None and str_val is not None and dur > 0:
                        segments.append((dur, str_val))
            if segments:
                return segments
            print("[LoRA Schedule] JSON parsed but no valid segments found.")
        except json.JSONDecodeError:
            print("[LoRA Schedule] Input looked like JSON but failed to parse. Falling back to text parser.")

    # --- 2. Text/Line Mode ---
    # We replace newlines with commas to unify parsing, but strip comments first
    clean_lines = []
    for line in s.splitlines():
        # Remove comments (anything after #)
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.strip()
        if line:
            clean_lines.append(line)
    
    # Join with comma to handle both "line separated" and "comma separated"
    unified_str = ",".join(clean_lines)
    
    segments = []
    # Current format expected: "DURATION : STRENGTH"
    # We split by comma
    tokens = [t.strip() for t in unified_str.split(",") if t.strip()]
    
    for token in tokens:
        # constant float check (no colon)
        if ":" not in token:
            # If the whole input is just one number, it's not a schedule, it's a constant
            if len(tokens) == 1 and _try_parse_float(token) is not None:
                return None 
            continue
            
        parts = token.split(":", 1)
        if len(parts) != 2:
            continue
            
        dur = _try_parse_float(parts[0].strip())
        strength = _try_parse_float(parts[1].strip())
        
        if dur is not None and strength is not None and dur > 0:
            segments.append((dur, strength))

    return segments or None


# ---------------------------------------------------------------------------
# Keyframe construction
# ---------------------------------------------------------------------------

def _create_stepwise_keyframes_from_durations(segments, total_steps=0):
    if not segments:
        return None

    sum_durations = sum(d for d, _ in segments)
    
    # If total_steps is provided (>0), we use it as the denominator (Absolute Steps)
    # Otherwise we use the sum of the segments (Relative/Percentage)
    total = float(total_steps) if total_steps > 0 else sum_durations

    if total <= 0:
        return None

    eps = 1e-6
    group = comfy.hooks.HookKeyframeGroup()
    current = 0.0

    for idx, (dur, strength) in enumerate(segments):
        start_percent = max(0.0, min(1.0, current / total))
        guarantee_steps = 1 if idx == 0 else 0

        # Start of segment
        group.add(comfy.hooks.HookKeyframe(strength=strength, start_percent=start_percent, guarantee_steps=guarantee_steps))

        current += dur
        boundary = max(0.0, min(1.0, current / total))

        # End of segment (hold)
        group.add(comfy.hooks.HookKeyframe(strength=strength, start_percent=boundary, guarantee_steps=0))

        # Jump if next exists
        if idx < len(segments) - 1:
            next_strength = segments[idx + 1][1]
            group.add(
                comfy.hooks.HookKeyframe(
                    strength=next_strength,
                    start_percent=min(1.0, boundary + eps),
                    guarantee_steps=0,
                )
            )

    # Final anchor
    # If we are using absolute steps and the user didn't cover the whole range 
    # (e.g. defined 9 steps out of 20), this ensures the last value holds to the end.
    # Users should end with "1 : 0.0" if they want it to turn off.
    group.add(comfy.hooks.HookKeyframe(strength=segments[-1][1], start_percent=1.0, guarantee_steps=0))
    return group


def _hooks_to_tuple(h):
    if h is None: return ()
    if isinstance(h, list): return tuple(x for x in h if x is not None)
    if isinstance(h, tuple): return tuple(x for x in h if x is not None)
    if hasattr(h, "hooks"): return tuple(x for x in h.hooks if x is not None)
    return (h,)

def _merge_hooks(hooks_a, hooks_b):
    return _hooks_to_tuple(hooks_a) + _hooks_to_tuple(hooks_b)


# ---------------------------------------------------------------------------
# Conditioning Logic
# ---------------------------------------------------------------------------

def append_hooks_to_conditioning(conditioning, hooks):
    hook_tuple = _hooks_to_tuple(hooks)
    if not hook_tuple:
        return conditioning

    out = []
    for item in conditioning:
        try:
            cond, opts = item
            if not isinstance(opts, dict):
                out.append(item)
                continue

            new_opts = dict(opts)
            existing = new_opts.get("hooks", None)
            
            # Create HookGroup
            new_group = comfy.hooks.HookGroup()
            
            # Add existing
            if existing:
                if hasattr(existing, "hooks"):
                    for h in existing.hooks: new_group.add(h)
                elif isinstance(existing, (list, tuple)):
                    for h in existing: new_group.add(h)
                else:
                    new_group.add(existing)
            
            # Add new
            for h in hook_tuple:
                new_group.add(h)

            new_opts["hooks"] = new_group
            
            if isinstance(item, tuple):
                out.append((cond, new_opts))
            else:
                out.append([cond, new_opts])
        except Exception as e:
            print(f"[ApplyHooks] Error: {e}")
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class LoRALoaderCustomStackable:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file to load"}),
                "strength": (
                    "STRING",
                    {
                        "default": "1.0",
                        "multiline": True,
                        "tooltip": "Examples:\n0.2:0.9 (20% duration at 0.9 strength)\n\nJSON:\n[{\"duration\":0.2, \"strength\":0.9}]",
                    },
                ),
            },
            "optional": {
                "hooks": ("HOOKS",),
                "total_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Total sampling steps (from KSampler). If 0, 'strength' durations are treated as relative percentages. If > 0, they are treated as absolute steps."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    RETURN_NAMES = ("model", "clip", "hooks")
    FUNCTION = "apply"
    CATEGORY = "loaders/lora"

    def apply(self, model, clip, lora_name, strength, hooks=None, total_steps=0):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            print(f"[LoRA] Error: {lora_name} not found.")
            return (model, clip, _hooks_to_tuple(hooks))

        if lora_path.endswith(".safetensors"):
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        else:
            lora = torch.load(lora_path, map_location="cpu")

        segments = _parse_duration_schedule(strength)

        # --- SCHEDULE MODE ---
        if segments:
            hook_obj = comfy.hooks.create_hook_lora(lora=lora, strength_model=1.0, strength_clip=0.0)
            
            # Pass total_steps to the keyframe creator
            kf_group = _create_stepwise_keyframes_from_durations(segments, total_steps)

            if hook_obj and kf_group:
                hook_obj.set_keyframes_on_hooks(kf_group)
                model_out = model.clone()
                target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)
                model_out.register_all_hook_patches(hook_obj, target_dict)

                # Fix for CLIP: use max strength found in schedule
                clip_strength = max([s[1] for s in segments])
                model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip, lora, 0.0, clip_strength)

                mode_str = f"Absolute Steps (Total {total_steps})" if total_steps > 0 else "Relative %"
                print(f"[LoRA] Scheduled {lora_name} | Mode: {mode_str} | Segments: {len(segments)}")
                return (model_out, clip_out, _merge_hooks(hooks, hook_obj))

        # --- CONSTANT MODE ---
        const_strength = _try_parse_float(strength)
        if const_strength is None:
            # If parsing failed and it wasn't a valid schedule, default to 1.0 or warn
            print(f"[LoRA] Warning: Invalid strength '{strength}', defaulting to unapplied.")
            return (model, clip, _hooks_to_tuple(hooks))

        model_out, clip_out = comfy.sd.load_lora_for_models(model, clip, lora, const_strength, const_strength)
        print(f"[LoRA] Constant {lora_name} @ {const_strength}")
        return (model_out, clip_out, _hooks_to_tuple(hooks))


class ApplyHooksToConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "conditioning": ("CONDITIONING",), },
            "optional": { "hooks": ("HOOKS",), },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning"

    def apply(self, conditioning, hooks=None):
        return (append_hooks_to_conditioning(conditioning, hooks),)


NODE_CLASS_MAPPINGS = {
    "LoRALoaderCustomStackable": LoRALoaderCustomStackable,
    "ApplyHooksToConditioning": ApplyHooksToConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRALoaderCustomStackable": "LoRA Loader Custom (Stackable + CLIP)",
    "ApplyHooksToConditioning": "Apply Hooks To Conditioning (append)",
}