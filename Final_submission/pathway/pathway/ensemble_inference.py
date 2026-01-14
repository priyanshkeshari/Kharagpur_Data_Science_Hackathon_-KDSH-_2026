"""
Pathway-based Real-Time Ensemble Model Inference
For KDSH 2026 Competition - Character Statement Consistency Detection
"""

import pathway as pw
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from typing import Dict, List, Any

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for model paths and ensemble weights"""
    DEBERTA_PATH = "/path/to/deberta-v3-base-nli/checkpoint-40"
    QWEN_V2_PATH = "/path/to/qwen2.5-7b-books-lora-cls/checkpoint-10000"
    QWEN_V1_PATH = "/path/to/qwen2.5-7b-books-lora-cls/checkpoint-5000"
    MC_CONSTRAINTS_PATH = "/path/to/monte_cristo_constraints_updated_2.jsonl"
    CA_CONSTRAINTS_PATH = "/path/to/castaways_constraints_filled.jsonl"
    
    # Ensemble weights (optimized from training)
    ENSEMBLE_WEIGHTS = np.array([0.98402039, 0.00670148, 0.00927813])
    THRESHOLD = 0.5
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 512
    MAX_CONSTRAINTS = 6


# ==================== CONSTRAINT PROCESSING ====================
class ConstraintProcessor:
    """Handles loading and processing of character constraints"""
    
    def __init__(self, mc_path: str, ca_path: str):
        self.constraints = self._load_all_constraints(mc_path, ca_path)
    
    def _load_constraints(self, path: str) -> Dict:
        """Load constraints from JSONL file"""
        mapping = {}
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = (obj["book_name"], obj["character"])
                mapping[key] = obj.get("constraints", [])
        return mapping
    
    def _load_all_constraints(self, mc_path: str, ca_path: str) -> Dict:
        """Load and merge all constraints"""
        mc_constraints = self._load_constraints(mc_path)
        ca_constraints = self._load_constraints(ca_path)
        return {**mc_constraints, **ca_constraints}
    
    def constraint_to_sentence(self, book: str, char: str, c: Dict) -> str:
        """Convert constraint dictionary to natural language sentence"""
        dim, val = c["dimension"], c["value"]
        
        templates = {
            "health_state": f"In {book}, {char} is {val}.",
            "family_role": f"In {book}, {char} has family role: {val}.",
            "role": f"In {book}, {char} is described as {val}.",
            "geographic_expertise": f"{char} is familiar with {val}.",
            "criminal_history": f"{char} has criminal history: {val}."
        }
        
        return templates.get(dim, f"{dim}: {val}.")
    
    def build_context(self, book: str, char: str, max_cons: int = 6) -> str:
        """Build context string from character constraints"""
        cons = self.constraints.get((book, char), [])
        sentences = [
            self.constraint_to_sentence(book, char, c) 
            for c in cons[:max_cons]
        ]
        return " ".join(sentences)


# ==================== MODEL WRAPPER ====================
class EnsembleModelWrapper:
    """Wrapper for ensemble of DeBERTa and Qwen models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self._load_models()
    
    def _load_models(self):
        """Load all models and tokenizers"""
        print("Loading DeBERTa model...")
        self.deb_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.deb_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.DEBERTA_PATH, 
            num_labels=2,
            local_files_only=True
        ).to(self.device).eval()
        
        print("Loading Qwen base model and adapters...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            num_labels=2,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load both adapters
        self.qwen_model = PeftModel.from_pretrained(
            base_model, 
            self.config.QWEN_V2_PATH, 
            adapter_name="v2"
        )
        self.qwen_model.load_adapter(self.config.QWEN_V1_PATH, adapter_name="v1")
        self.qwen_model.eval()
        
        print("All models loaded successfully!")
    
    def _get_single_prob(self, text: str, tokenizer, model, adapter_name=None) -> float:
        """Get probability from a single model"""
        if adapter_name:
            model.set_adapter(adapter_name)
        
        inputs = tokenizer(
            text,
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
            padding="max_length"
        ).to(self.device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return probs[1]  # Probability of "contradict" class
    
    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Run ensemble prediction on a single example"""
        # Format text for models
        text = f"Premise: {premise}\nHypothesis: {hypothesis}"
        
        # Get predictions from each model
        p_deb = self._get_single_prob(text, self.deb_tokenizer, self.deb_model)
        p_qw2 = self._get_single_prob(text, self.qwen_tokenizer, self.qwen_model, "v2")
        p_qw1 = self._get_single_prob(text, self.qwen_tokenizer, self.qwen_model, "v1")
        
        # Ensemble prediction
        pred_vector = np.array([p_deb, p_qw2, p_qw1])
        final_prob = np.dot(pred_vector, self.config.ENSEMBLE_WEIGHTS)
        
        # Classification
        prediction = int(final_prob > self.config.THRESHOLD)
        label = "contradict" if prediction == 1 else "consistent"
        
        return {
            "label": label,
            "probability": float(final_prob),
            "model_probs": {
                "deberta": float(p_deb),
                "qwen_v2": float(p_qw2),
                "qwen_v1": float(p_qw1)
            }
        }


# ==================== PATHWAY PIPELINE ====================
def build_pathway_pipeline(
    input_path: str,
    output_path: str,
    config: Config,
    constraint_processor: ConstraintProcessor,
    model_wrapper: EnsembleModelWrapper
):
    """Build Pathway streaming pipeline for real-time inference"""
    
    # Define input schema
    class InputSchema(pw.Schema):
        id: int
        book_name: str
        char: str
        content: str
    
    # Read input data stream
    input_table = pw.io.csv.read(
        input_path,
        schema=InputSchema,
        mode="streaming"
    )
    
    # Define UDF for context building
    @pw.udf
    def add_context(book_name: str, char: str) -> str:
        return constraint_processor.build_context(book_name, char)
    
    # Define UDF for model prediction
    @pw.udf
    def predict_consistency(context: str, content: str) -> str:
        result = model_wrapper.predict(context, content)
        return result["label"]
    
    @pw.udf
    def predict_probability(context: str, content: str) -> float:
        result = model_wrapper.predict(context, content)
        return result["probability"]
    
    # Build the processing pipeline
    enriched_table = input_table.select(
        id=input_table.id,
        book_name=input_table.book_name,
        char=input_table.char,
        content=input_table.content,
        context=add_context(input_table.book_name, input_table.char)
    )
    
    # Add predictions
    predictions = enriched_table.select(
        id=enriched_table.id,
        label=predict_consistency(enriched_table.context, enriched_table.content),
        probability=predict_probability(enriched_table.context, enriched_table.content)
    )
    
    # Write results
    pw.io.csv.write(predictions, output_path)
    
    return predictions


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Load constraint processor
    print("Loading constraints...")
    constraint_processor = ConstraintProcessor(
        config.MC_CONSTRAINTS_PATH,
        config.CA_CONSTRAINTS_PATH
    )
    
    # Initialize model wrapper
    print("Initializing models...")
    model_wrapper = EnsembleModelWrapper(config)
    
    # Build and run Pathway pipeline
    print("Building Pathway pipeline...")
    input_path = "test.csv"  # Input CSV path
    output_path = "submission.csv"  # Output CSV path
    
    pipeline = build_pathway_pipeline(
        input_path=input_path,
        output_path=output_path,
        config=config,
        constraint_processor=constraint_processor,
        model_wrapper=model_wrapper
    )
    
    # Run the pipeline
    print("Running inference pipeline...")
    pw.run()
    
    print("✅ Pipeline completed! Results saved to submission.csv")


if __name__ == "__main__":
    main()