# fuzzer_core_relation.py

import os
import time
import csv
import random
import torch
import open_clip
import math

from image_utils_relation import (
    load_target_image, load_diffusion_model, fuzzing_status, mutator,
    mutate_single, execute, prompt_node
)
from llm_utils import prepare_model_and_tok_vl


class ImagePromptFuzzer:
    def __init__(self, args):
        self.args = args
        self.base_prompt = args.base_prompt
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        args.device = self.device
        print(f"Using device: {self.device}")

        # --- Load Models ---
        self.target_image = load_target_image(args.target_image_path)
        self.diffusion_pipeline = load_diffusion_model(args.target_model_path, self.device)
        print("Loading CLIP model for scoring...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=self.device
        )
        print("CLIP model loaded.")

        self.status = None

        if 'gpt' not in args.image_model_path:
            self.PROCESSOR, self.VL_MODEL = prepare_model_and_tok_vl(args)
        else:
            self.PROCESSOR, self.VL_MODEL = None, None

        # ================================
        # STAGED SCHEDULE CONFIG (NEW)
        # ================================
        # If args.base_only is True, we will stay base-only always.
        # If args.base_only is False, we run staged mode:
        #   first `warmup_steps` base-only, then switch to full set.
        self.staged_mode = not bool(getattr(self.args, "base_only", False))
        self.warmup_steps = int(getattr(self.args, "warmup_steps", 30))  # first 20 steps base-only
        # mutation steps counter (counts loop iterations that create new nodes)
        self.total_mutation_steps = 0

        # --- INIT ORDER FIX: make sure these exist before _set_phase_mutators() ---
        self.mutators_to_select = []          # placeholder; will be set in _set_phase_mutators
        self.ucb_exploration_factor = math.sqrt(2)
        self.ucb_stats = {} 

        # Initialize mutator set according to phase
        self._set_phase_mutators(init=True)

        # UCB stats (kept even if you pick randomly; handy to switch back)
        self.ucb_exploration_factor = math.sqrt(2)
        self.ucb_stats = {m: {'count': 0, 'total_reward': 0.0} for m in self.mutators_to_select}
        print(f"Mutators enabled: {', '.join([m.name for m in self.mutators_to_select])}")

        # === CSV Log header ===
        log_dir = args.record_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"fuzz_results_{args.target_image_path[-7:-4]}.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query", "full_prompt", "relation", "style",
                "clip_similarity", "mutation_type", "generation", "best_score_so_far"
            ])

        print("=" * 50)
        print("Fuzzer initialized and ready to run.")
        print(f"Using Base Prompt: '{self.base_prompt}'")
        print("=" * 50)

    # ------------------------------
    # Phase switching helper (NEW)
    # ------------------------------
    def _set_phase_mutators(self, init=False):
        """Switch mutator set + args.base_only flag depending on current phase."""
        in_warmup = (self.staged_mode and self.total_mutation_steps < self.warmup_steps) 
        if in_warmup or bool(getattr(self.args, "base_only", False)):
            self.mutators_to_select = [mutator.ENRICH_BASE_INLINE, mutator.FIX_GRAMMAR, mutator.PARAPHRASE_BASE]
            # force base-only behavior in image_utils
            phase_name = "BASE-ONLY"
        else:
            self.mutators_to_select = [
                mutator.GEN_DESC_STYLE,
                mutator.MODIFY_STYLE,
                mutator.MODIFY_DESC,
                mutator.PARAPHRASE_BASE,
                mutator.ENRICH_BASE_INLINE,
                mutator.FIX_GRAMMAR,
            ]
            phase_name = "FULL"

        if not init:
            print(f"[PHASE SWITCH] Using {phase_name} mutators: {', '.join([m.name for m in self.mutators_to_select])}")

        # Ensure UCB stats has keys for all current mutators
        if not hasattr(self, "ucb_stats") or self.ucb_stats is None:
            self.ucb_stats = {}
        for m in self.mutators_to_select:
            if m not in getattr(self, "ucb_stats", {}):
                self.ucb_stats[m] = {'count': 0, 'total_reward': 0.0}

    # UCB1 (unchanged; optional use)
    def select_mutator_ucb(self):
        for m in self.mutators_to_select:
            if self.ucb_stats[m]['count'] == 0:
                return m
        best_mutator = None
        max_ucb_score = -float('inf')
        for m in self.mutators_to_select:
            count = self.ucb_stats[m]['count']
            avg = self.ucb_stats[m]['total_reward'] / count
            exploration = self.ucb_exploration_factor * math.sqrt(math.log(max(1, self.total_mutation_steps)) / count)
            score = avg + exploration
            if score > max_ucb_score:
                max_ucb_score = score
                best_mutator = m
        return best_mutator

    def update_ucb_stats(self, chosen_mutator, reward):
        if chosen_mutator in self.ucb_stats:
            self.ucb_stats[chosen_mutator]['count'] += 1
            self.ucb_stats[chosen_mutator]['total_reward'] += reward

    def run(self):
        start_time = time.time()

        print("\nGenerating initial population...")
        initial_nodes = []

        # ① include raw base prompt as an initial node
        base_score, _, _ = execute(
            self.base_prompt, self.target_image, self.diffusion_pipeline,
            self.clip_model, self.clip_preprocess, self.device, self.args
        )
        base_node = prompt_node(
            text=self.base_prompt,
            relation="",
            style="",
            base_prompt=self.base_prompt,
            response=base_score
        )
        initial_nodes.append(base_node)
        print(f"Added raw base prompt node (Score: {base_score:.4f})")

        self.status = fuzzing_status(initial_nodes=initial_nodes, max_query=self.args.max_query)
        for node in initial_nodes:
            self._log_result(node, "initial")

        # ========= main loop =========
        while not self.status.stop_condition():
            self._set_phase_mutators()  # may switch phase and mutator set
            parent_node = self.status.seed_selection_strategy()
            # Use UCB or random selection
            # mutation_to_apply = self.select_mutator_ucb()
            mutation_to_apply = random.choice(self.mutators_to_select)

            print(f"\n--- Query {self.status.query + 1}/{self.args.max_query} ---")
            print(f"  Parent (Score: {parent_node.response:.4f}): base prompt='{parent_node.base_prompt}', relation='{parent_node.relation}', style='{parent_node.style}'")
            print(f"  Applying Mutation: {mutation_to_apply.name.upper()}")
            full_prompt, base, relation, style = mutate_single(
                base_prompt=parent_node.base_prompt,
                mutation_type=mutation_to_apply,
                parent_node=parent_node,
                args=self.args,
                target_image=self.target_image,
                PROCESSOR=self.PROCESSOR,
                VL_MODEL=self.VL_MODEL
            )

            new_score, _, _ = execute(
                full_prompt, self.target_image, self.diffusion_pipeline,
                self.clip_model, self.clip_preprocess, self.device, self.args
            )

            # accounting and stats
            self.total_mutation_steps += 1
            self.update_ucb_stats(mutation_to_apply, new_score)

            print(f"  New Result (Score: {new_score:.4f}): base prompt='{base}', relation='{relation}', style='{style}'")

            new_node = prompt_node(
                text=full_prompt,
                relation=relation,
                style=style,
                base_prompt=base,
                response=new_score,
                parent=parent_node,
                mutation=mutation_to_apply.name
            )
            self.status.update_with_node(new_node)
            self._log_result(new_node, mutation_to_apply.name)

        end_time = time.time()
        best_seed = self.status.get_best_seed()
        print("\n" + "=" * 50)
        print("Fuzzing finished!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Best score found: {best_seed.response:.4f}")
        print(f"Best prompt: '{best_seed.text}'")
        print(f"Results saved to: {self.csv_path}")
        print("=" * 50)

    def _log_result(self, node, mutation_type):
        best_score_so_far = self.status.get_best_seed().response
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.status.query, node.text, node.relation, node.style,
                f"{node.response:.4f}", mutation_type, node.generation,
                f"{best_score_so_far:.4f}"
            ])
