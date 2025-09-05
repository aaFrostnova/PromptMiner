# fuzzer_core.py

import os
import time
import csv
import random
import torch
import open_clip

# Import all utilities and data structures
# Make sure your utils_image.py has the new mutator enum and prompt_node class
from image_utils import (
    load_target_image, load_diffusion_model, fuzzing_status, mutator,
    mutate_single, execute, prompt_node
)
from llm_utils import prepare_model_and_tok_vl

class ImagePromptFuzzer:
    def __init__(self, args):
        self.args = args
        # Store the base prompt from the arguments
        self.base_prompt = args.base_prompt
        self.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        args.device = self.device  # Ensure args has the device info
        print(f"Using device: {self.device}")

        # --- Load Models ---
        self.target_image = load_target_image(args.target_image_path)
        self.diffusion_pipeline = load_diffusion_model(args.target_model_path, self.device)
        
        print("Loading CLIP model for scoring...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=self.device
        )
        print("CLIP model loaded.")

        # --- Fuzzing State ---
        self.status = None # Will be initialized in run()

        if 'gpt' not in args.image_model_path: 
            self.PROCESSOR, self.VL_MODEL = prepare_model_and_tok_vl(args)
        else:
            self.PROCESSOR, self.VL_MODEL = None, None
        # --- Logging ---
        # Updated logging to include detail and position
        log_dir = args.record_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"fuzz_results_{args.target_image_path[-7:-4]}.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "full_prompt", "detail", "position", "clip_similarity", "mutation_type", "generation", "best_score_so_far"])
        
        print("=" * 50)
        print("Fuzzer initialized and ready to run.")
        print(f"Using Base Prompt: '{self.base_prompt}'")
        print("=" * 50)

    def run(self):
        start_time = time.time()
        
        # --- Step 1: Create and evaluate an initial population from the base prompt ---
        print("\nGenerating initial population...")
        initial_nodes = []
        # Generate 5 initial variations to seed the MCTS tree
        for i in range(5):
            print(f"Generating initial seed {i+1}/5...")
            # Use the new mutate_single that returns structured data
            full_prompt, detail, position = mutate_single(
                base_prompt=self.base_prompt,
                mutation_type=mutator.GENERATE_NEW,
                args=self.args,
                target_image=self.target_image, # No parent_node needed for initial generation
                PROCESSOR=self.PROCESSOR,
                VL_MODEL=self.VL_MODEL 
            )
            
            # Evaluate the newly generated full prompt
            score, _, _ = execute(
                full_prompt, self.target_image, self.diffusion_pipeline,
                self.clip_model, self.clip_preprocess, self.device, self.args
            )

            # Create a prompt_node that stores all the new info
            node = prompt_node(text=full_prompt, detail=detail, position=position, response=score)
            initial_nodes.append(node)
        
        print("\nInitial population generated and evaluated.")
        
        # --- Step 2: Initialize fuzzing status with the scored population ---
        self.status = fuzzing_status(initial_nodes=initial_nodes, max_query=self.args.max_query)
        # Log each of the initial nodes
        for node in initial_nodes:
            self._log_result(node, "initial")

        # --- Step 3: Main fuzzing loop ---
        while not self.status.stop_condition():
            # --- Select a parent prompt using MCTS strategy ---
            parent_node = self.status.seed_selection_strategy()
            
            # --- Choose a refinement strategy (avoid GENERATE_NEW in the main loop) ---
            mutation_to_apply = random.choice([mutator.REFINE_DETAIL, mutator.CHANGE_POSITION])
            
            print(f"\n--- Query {self.status.query + 1}/{self.args.max_query} ---")
            print(f"  Parent (Score: {parent_node.response:.4f}): detail='{parent_node.detail}', position='{parent_node.position}'")
            print(f"  Applying Mutation: {mutation_to_apply.name.upper()}")
            # --- Mutate the prompt by refining the parent node ---
            full_prompt, detail, position = mutate_single(
                base_prompt=self.base_prompt,
                mutation_type=mutation_to_apply,
                parent_node=parent_node,
                args=self.args,
                target_image=self.target_image,
                PROCESSOR=self.PROCESSOR,
                VL_MODEL=self.VL_MODEL 
            )
            
            # --- Execute and evaluate the new prompt ---
            new_score, _, _ = execute(
                full_prompt, self.target_image, self.diffusion_pipeline,
                self.clip_model, self.clip_preprocess, self.device, self.args
            )
            
            print(f"  New Result (Score: {new_score:.4f}): detail='{detail}', position='{position}'")

            # --- Create a new node with all info and update the MCTS tree ---
            new_node = prompt_node(
                text=full_prompt, detail=detail, position=position,
                response=new_score, parent=parent_node, mutation=mutation_to_apply.name
            )
            self.status.update_with_node(new_node)
            
            # Log the new result
            self._log_result(new_node, mutation_to_apply.name)

        end_time = time.time()
        best_seed = self.status.get_best_seed()
        print("\n" + "="*50)
        print("Fuzzing finished!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Best score found: {best_seed.response:.4f}")
        print(f"Best prompt: '{best_seed.text}'")
        print(f"Results saved to: {self.csv_path}")
        print("="*50)

    def _log_result(self, node, mutation_type):
        """Logs the structured prompt data to the CSV file."""
        best_score_so_far = self.status.get_best_seed().response
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write all the relevant columns
            writer.writerow([
                self.status.query, node.text, node.detail, node.position,
                f"{node.response:.4f}", mutation_type, node.generation,
                f"{best_score_so_far:.4f}"
            ])