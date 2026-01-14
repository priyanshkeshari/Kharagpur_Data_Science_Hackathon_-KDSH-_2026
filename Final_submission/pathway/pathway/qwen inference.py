import pathway as pw
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import PeftModel
from datasets import Dataset
import gc

# ==================== Configuration ====================
class Config:
    """Centralized configuration for inference pipeline"""
    
    # Paths
    TEST_PATH = "/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset/train.csv"
    MC_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters-2/Jsonl_file_chars/monte_cristo/monte_cristo_constraints_updated_2.jsonl"
    CA_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters-2/Jsonl_file_chars/castaways/castaways_constraints_filled.jsonl"
    ADAPTER_PATH = "/kaggle/input/kdsh26-qwen2-5-7b-instruct-fine-t-model-checkpoint/qwen2.5-7b-books-lora-cls/kaggle/working/qwen2.5-7b-books-lora-cls/checkpoint-5000"
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    
    # Output
    OUTPUT_CSV = "submission.csv"
    FULL_OUTPUT_CSV = "kdsh26_qwen2.5_7b_base_causllm_rationale_inference.csv"
    
    # Inference parameters
    MAX_LENGTH = 512
    EVAL_BATCH_SIZE = 16
    MAX_NEW_TOKENS = 60
    TEMPERATURE = 0.1

# ==================== Constraint Loading ====================
def load_constraints_from_jsonl(path):
    """Load character constraints from JSONL file"""
    mapping = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["book_name"], obj["character"])
            mapping[key] = obj.get("constraints", [])
    return mapping

# Load constraints globally
print("Loading constraints...")
mc_constraints = load_constraints_from_jsonl(Config.MC_CONS_PATH)
ca_constraints = load_constraints_from_jsonl(Config.CA_CONS_PATH)
ALL_CONSTRAINTS = {**mc_constraints, **ca_constraints}

# ==================== Pathway UDFs ====================
@pw.udf
def constraint_to_sentence_udf(book: str, char: str, dimension: str, value: str) -> str:
    """Convert single constraint to natural language"""
    if dimension == "health_state":
        return f"In {book}, {char} is {value}."
    elif dimension == "family_role":
        return f"In {book}, {char} has family role: {value}."
    elif dimension == "role":
        return f"In {book}, {char} is described as {value}."
    elif dimension == "geographic_expertise":
        return f"{char} is familiar with {value}."
    elif dimension == "criminal_history":
        return f"{char} has criminal history: {value}."
    else:
        return f"{dimension}: {value}."

@pw.udf
def build_context_udf(book_name: str, char: str, max_cons: int = 6) -> str:
    """Build context with fuzzy matching for character names"""
    # Try exact match first
    cons = ALL_CONSTRAINTS.get((book_name, char), [])
    
    # If empty, try fuzzy match
    if not cons:
        possible_matches = [
            k for k in ALL_CONSTRAINTS.keys() 
            if k[0] == book_name and (char in k[1] or k[1] in char)
        ]
        if possible_matches:
            cons = ALL_CONSTRAINTS[possible_matches[0]]
    
    if not cons:
        return ""
    
    # Format sentences
    sentences = []
    for c in cons[:max_cons]:
        dim = c["dimension"]
        val = c["value"]
        
        if dim == "health_state":
            sent = f"In {book_name}, {char} is {val}."
        elif dim == "family_role":
            sent = f"In {book_name}, {char} has family role: {val}."
        elif dim == "role":
            sent = f"In {book_name}, {char} is described as {val}."
        elif dim == "geographic_expertise":
            sent = f"{char} is familiar with {val}."
        elif dim == "criminal_history":
            sent = f"{char} has criminal history: {val}."
        else:
            sent = f"{dim}: {val}."
        
        sentences.append(sent)
    
    return " ".join(sentences)

@pw.udf
def format_full_text_udf(context: str, content: str) -> str:
    """Format as Premise-Hypothesis"""
    return f"Premise: {context}\nHypothesis: {content}"

# ==================== Pathway Data Pipeline ====================
class PathwayInferencePipeline:
    """Pathway-based inference pipeline with rationale generation"""
    
    def __init__(self):
        self.config = Config()
        
    def create_inference_table(self):
        """Create Pathway table with preprocessing"""
        
        print("Loading Test Data...")
        test_table = pw.io.csv.read(
            self.config.TEST_PATH,
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=int),
                "book_name": pw.column_definition(dtype=str),
                "char": pw.column_definition(dtype=str),
                "caption": pw.column_definition(dtype=str, optional=True),
                "content": pw.column_definition(dtype=str),
                "label": pw.column_definition(dtype=str),
            }),
            mode="static"
        )
        
        # Add context with fuzzy matching
        test_with_context = test_table.select(
            id=test_table.id,
            book_name=test_table.book_name,
            char=test_table.char,
            caption=test_table.caption,
            content=test_table.content,
            label=test_table.label,
            context=build_context_udf(test_table.book_name, test_table.char)
        )
        
        # Format full text
        test_formatted = test_with_context.select(
            id=pw.this.id,
            book_name=pw.this.book_name,
            char=pw.this.char,
            caption=pw.this.caption,
            content=pw.this.content,
            label=pw.this.label,
            context=pw.this.context,
            full_text=format_full_text_udf(pw.this.context, pw.this.content)
        )
        
        return test_formatted
    
    def export_to_dataframe(self, pw_table, output_path="/tmp/test_processed.csv"):
        """Export Pathway table to pandas DataFrame"""
        
        # Write table
        pw.io.csv.write(pw_table, output_path)
        
        # Execute computation
        pw.run()
        
        # Load as DataFrame
        df = pd.read_csv(output_path)
        
        print(f"Test data shape: {df.shape}")
        return df

# ==================== Classification Inference ====================
class ClassificationInference:
    """Handle classification inference with LoRA adapter"""
    
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        
    def setup(self):
        """Initialize tokenizer and model with adapter"""
        print("--- Loading Classifier ---")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.BASE_MODEL, 
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.BASE_MODEL,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
            ignore_mismatched_sizes=True
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, self.config.ADAPTER_PATH)
        self.model.eval()
        
        return self.model, self.tokenizer
    
    def predict(self, test_df):
        """Run classification predictions"""
        print("--- Predicting Labels ---")
        
        def tokenize(examples):
            return self.tokenizer(
                examples["full_text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.MAX_LENGTH
            )
        
        # Create HuggingFace Dataset
        hf_ds = Dataset.from_pandas(test_df[["full_text"]]).map(
            tokenize, 
            batched=True
        ).remove_columns(["full_text"])
        
        # Run inference
        trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=TrainingArguments(
                output_dir="/tmp",
                per_device_eval_batch_size=self.config.EVAL_BATCH_SIZE,
                report_to="none"
            )
        )
        
        logits = trainer.predict(hf_ds).predictions
        pred_ids = np.argmax(logits, axis=-1)
        
        # Map to labels
        id2label = {0: "consistent", 1: "contradict"}
        labels = [id2label[i] for i in pred_ids]
        
        print(f"Predictions complete. Sample: {labels[:5]}")
        
        return labels
    
    def cleanup(self):
        """Free VRAM"""
        print("--- Classifier Unloaded ---")
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

# ==================== Rationale Generation ====================
class RationaleGenerator:
    """Generate explanations using CausalLM"""
    
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        
    def setup(self):
        """Load CausalLM for rationale generation"""
        print("--- Loading Reasoning Model ---")
        
        # Use same tokenizer from classification
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.BASE_MODEL,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load CausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        
        return self.model, self.tokenizer
    
    def generate_reasoning(self, row):
        """Generate strict reasoning for a single row"""
        prompt = f"""### Instruction
You are a strict logical auditor. Your task is to verify the relationship between the Context and the Hypothesis.

**Context (True Facts):** "{row['context']}"
**Hypothesis (Claim):** "{row['content']}"
**Verified Relationship:** {row['label'].upper()}

**Task:**
1. Ignore all outside knowledge. Use ONLY the 'Context' above.
2. If the relationship is CONTRADICT: Quote the specific part of the Context that proves the Hypothesis is wrong.
3. If the relationship is CONSISTENT: Quote the specific part of the Context that supports the Hypothesis.

### Output
Provide a single, short sentence explaining the {row['label']} relationship based ONLY on the context provided.
"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                do_sample=False
            )
        
        response = self.tokenizer.decode(
            generated_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_batch(self, df):
        """Generate reasoning for entire dataframe"""
        print("--- Generating Explanations ---")
        
        reasonings = []
        for idx, row in df.iterrows():
            reasoning = self.generate_reasoning(row)
            reasonings.append(reasoning)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(df)} rows")
        
        return reasonings

# ==================== Complete Pipeline ====================
class PathwayQwenInferencePipeline:
    """Complete end-to-end Pathway inference with rationale generation"""
    
    def __init__(self):
        self.config = Config()
        self.data_pipeline = PathwayInferencePipeline()
        self.classifier = ClassificationInference()
        self.rationale_gen = RationaleGenerator()
        
    def run(self):
        """Execute complete inference pipeline"""
        
        # Step 1: Prepare data with Pathway
        print("\n=== Data Preparation with Pathway ===")
        test_table = self.data_pipeline.create_inference_table()
        test_df = self.data_pipeline.export_to_dataframe(
            test_table,
            "/tmp/test_pathway_inference.csv"
        )
        
        # Step 2: Classification
        print("\n=== Classification Inference ===")
        self.classifier.setup()
        predictions = self.classifier.predict(test_df)
        test_df["label"] = predictions
        self.classifier.cleanup()
        
        # Step 3: Rationale Generation
        print("\n=== Rationale Generation ===")
        self.rationale_gen.setup()
        reasonings = self.rationale_gen.generate_batch(test_df)
        test_df["reasoning"] = reasonings
        
        # Step 4: Save results
        print("\n=== Saving Results ===")
        
        # Submission file (id, label only)
        submission = test_df[["id", "label"]]
        submission.to_csv(self.config.OUTPUT_CSV, index=False)
        print(f"Submission saved to {self.config.OUTPUT_CSV}")
        print(submission.head())
        
        # Full output with reasoning
        test_df.to_csv(self.config.FULL_OUTPUT_CSV, index=False)
        print(f"Full output saved to {self.config.FULL_OUTPUT_CSV}")
        
        # Display sample reasoning
        print("\n=== Sample Reasoning ===")
        for idx in range(min(5, len(test_df))):
            row = test_df.iloc[idx]
            print(f"\n[Label]: {row['label']}")
            print(f"[Content]: {row['content'][:100]}...")
            print(f"[Reasoning]: {row['reasoning']}")
        
        return test_df

# ==================== Evaluation ====================
def evaluate_predictions(test_df, ground_truth_path):
    """Evaluate predictions against ground truth"""
    from sklearn.metrics import (
        accuracy_score, 
        f1_score, 
        classification_report, 
        confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_path)
    
    y_true = ground_truth["label"]
    y_pred = test_df["label"]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    print(f"\nAccuracy: {acc:.4f} ({np.sum(y_true == y_pred)}/{len(y_true)})")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["consistent", "contradict"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["consistent", "contradict"],
        yticklabels=["consistent", "contradict"]
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create and run complete pipeline
    pipeline = PathwayQwenInferencePipeline()
    results_df = pipeline.run()
    
    print("\n" + "="*50)
    print("Inference Pipeline Completed Successfully!")
    print("="*50)
    
    # Optional: Evaluate if ground truth available
    # evaluate_predictions(
    #     results_df, 
    #     Config.TEST_PATH
    # )