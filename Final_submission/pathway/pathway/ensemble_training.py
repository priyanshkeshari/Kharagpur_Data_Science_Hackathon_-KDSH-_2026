import pathway as pw
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==================== Configuration ====================
class Config:
    """Centralized configuration for ensemble inference"""
    
    # Paths
    TEST_CSV = "/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset/train.csv"
    MC_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters-2/Jsonl_file_chars/monte_cristo/monte_cristo_constraints_updated_2.jsonl"
    CA_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters-2/Jsonl_file_chars/castaways/castaways_constraints_filled.jsonl"
    
    # Model paths
    DEBERTA_PATH = "/kaggle/input/kdsh26-deberta-v3-base-fine-tune-model/deberta-v3-base-nli/checkpoint-40"
    QWEN_V2_PATH = "/kaggle/input/newapproach/qwen2.5-7b-books-lora-cls/checkpoint-10000"
    QWEN_V1_PATH = "/kaggle/input/kdsh26-qwen2-5-7b-instruct-fine-t-model-checkpoint/qwen2.5-7b-books-lora-cls/kaggle/working/qwen2.5-7b-books-lora-cls/checkpoint-5000"
    
    # Output
    OUTPUT_ENSEMBLE = "submission.csv"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== Constraint Management ====================
def load_constraints_from_jsonl(path):
    """Load character constraints from JSONL file"""
    mapping = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj["book_name"], obj["character"])
            mapping[key] = obj.get("constraints", [])
    return mapping

# Load constraints globally
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
    """Build context from character constraints"""
    cons = ALL_CONSTRAINTS.get((book_name, char), [])
    if not cons:
        return ""
    
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

# ==================== Pathway Data Pipeline ====================
class PathwayEnsembleDataPipeline:
    """Pathway-based data preprocessing for ensemble inference"""
    
    def __init__(self):
        self.config = Config()
        
    def create_inference_table(self):
        """Create Pathway table with preprocessing"""
        
        print("Loading test data...")
        test_table = pw.io.csv.read(
            self.config.TEST_CSV,
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
        
        # Add context
        test_with_context = test_table.select(
            id=test_table.id,
            book_name=test_table.book_name,
            char=test_table.char,
            content=test_table.content,
            label=test_table.label,
            context=build_context_udf(test_table.book_name, test_table.char)
        )
        
        return test_with_context
    
    def export_to_dataframe(self, pw_table, output_path="/tmp/ensemble_data.csv"):
        """Export Pathway table to pandas DataFrame"""
        
        # Write table
        pw.io.csv.write(pw_table, output_path)
        
        # Execute computation
        pw.run()
        
        # Load as DataFrame
        df = pd.read_csv(output_path)
        
        print(f"Test data shape: {df.shape}")
        return df

# ==================== Model Management ====================
class EnsembleModels:
    """Manage multiple models for ensemble inference"""
    
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Model components
        self.deb_tk = None
        self.deb_mdl = None
        self.q_tk = None
        self.qwen_mdl = None
        
    def load_models(self):
        """Load DeBERTa and Qwen models with adapters"""
        
        print("--- Loading DeBERTa ---")
        self.deb_tk = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.deb_mdl = AutoModelForSequenceClassification.from_pretrained(
            self.config.DEBERTA_PATH,
            num_labels=2,
            local_files_only=True
        ).to(self.device).eval()
        
        print("--- Loading Qwen Base and Adapters ---")
        self.q_tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            num_labels=2,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load Qwen v2 as primary adapter
        self.qwen_mdl = PeftModel.from_pretrained(
            base_model,
            self.config.QWEN_V2_PATH,
            adapter_name="v2"
        )
        
        # Load Qwen v1 as additional adapter
        self.qwen_mdl.load_adapter(self.config.QWEN_V1_PATH, adapter_name="v1")
        self.qwen_mdl.eval()
        
        return (self.deb_tk, self.deb_mdl), self.q_tk, self.qwen_mdl
    
    def get_probabilities(self, df, tokenizer, model, model_type="qwen", adapter_name=None):
        """Get prediction probabilities from a model"""
        
        probs_list = []
        
        # Find appropriate columns
        cols = df.columns.tolist()
        ctx_col = next((c for c in ['context', 'context_for_inference', 'full_text'] if c in cols), None)
        stmt_col = next((c for c in ['content', 'statement'] if c in cols), None)
        
        if not ctx_col or not stmt_col:
            raise KeyError(f"Could not find required columns. Available: {cols}")
        
        print(f"Using '{ctx_col}' as Premise and '{stmt_col}' as Hypothesis.")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference {model_type}"):
            premise = str(row[ctx_col])
            hypothesis = str(row[stmt_col])
            
            # Use same template as training
            if adapter_name:
                model.set_adapter(adapter_name)
            
            text = f"Premise: {premise}\nHypothesis: {hypothesis}"
            
            inputs = tokenizer(
                text,
                max_length=512,
                truncation=True,
                return_tensors="pt",
                padding="max_length"
            ).to(self.device)
            
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # class 1 = contradict
            probs_list.append(probs[1])
        
        return np.array(probs_list)

# ==================== Ensemble Weights ====================
class EnsembleWeights:
    """Pre-defined ensemble weights (from validation/training)"""
    
    # These weights should be determined from validation set during training
    # Default: equal weighting
    DEFAULT_WEIGHTS = np.array([0.33, 0.33, 0.34])  # DeBERTa, Qwen v2, Qwen v1
    
    # You can update these based on validation performance
    OPTIMIZED_WEIGHTS = np.array([0.97, 0.02, 0.01])  # Example from training
    
    @staticmethod
    def get_weights(use_optimized=True):
        """Get ensemble weights"""
        if use_optimized:
            weights = EnsembleWeights.OPTIMIZED_WEIGHTS
        else:
            weights = EnsembleWeights.DEFAULT_WEIGHTS
        
        # Normalize
        weights = weights / np.sum(weights)
        
        print(f"Using ensemble weights: {weights}")
        return weights

# ==================== Complete Pipeline ====================
class PathwayEnsemblePipeline:
    """Complete end-to-end ensemble inference with Pathway"""
    
    def __init__(self):
        self.config = Config()
        self.data_pipeline = PathwayEnsembleDataPipeline()
        self.models = EnsembleModels()
        
    def run(self):
        """Execute complete ensemble pipeline"""
        
        # Step 1: Prepare data with Pathway
        print("\n=== Data Preparation with Pathway ===")
        test_table = self.data_pipeline.create_inference_table()
        test_df = self.data_pipeline.export_to_dataframe(
            test_table,
            "/tmp/ensemble_pathway.csv"
        )
        
        # Step 2: Load models
        print("\n=== Loading Models ===")
        (deb_tk, deb_mdl), q_tk, qwen_mdl = self.models.load_models()
        
        # Step 3: Get predictions from all models
        print("\n=== Running Ensemble Inference ===")
        
        # DeBERTa predictions
        p_deb = self.models.get_probabilities(
            test_df, deb_tk, deb_mdl, "deberta"
        )
        
        # Qwen v2 predictions
        p_qw2 = self.models.get_probabilities(
            test_df, q_tk, qwen_mdl, "qwen_v2", adapter_name="v2"
        )
        
        # Qwen v1 predictions
        p_qw1 = self.models.get_probabilities(
            test_df, q_tk, qwen_mdl, "qwen_v1", adapter_name="v1"
        )
        
        # Create prediction matrix
        pred_matrix = np.vstack([p_deb, p_qw2, p_qw1]).T
        
        # Step 4: Get ensemble weights (pre-determined from validation)
        print("\n=== Applying Ensemble Weights ===")
        ensemble_weights = EnsembleWeights.get_weights(use_optimized=True)
        
        # Step 5: Generate final predictions
        print("\n=== Generating Final Predictions ===")
        final_probs = np.dot(pred_matrix, ensemble_weights)
        final_preds = (final_probs > 0.5).astype(int)
        
        # Step 6: Save results
        print("\n=== Saving Results ===")
        test_df['label'] = [
            "consistent" if p == 0 else "contradict"
            for p in final_preds
        ]
        
        submission = test_df[['id', 'label']]
        submission.to_csv(self.config.OUTPUT_ENSEMBLE, index=False)
        print(f"Submission saved to {self.config.OUTPUT_ENSEMBLE}")
        print(submission.head())
        
        # Distribution of predictions
        pred_dist = submission['label'].value_counts()
        print(f"\nPrediction Distribution:")
        print(pred_dist)
        
        return test_df, ensemble_weights

# ==================== Evaluation ====================
def evaluate_ensemble(test_df, ground_truth_path):
    """Evaluate ensemble predictions with visualizations"""
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
    
    return acc, f1_macro, f1_weighted

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create and run pipeline
    pipeline = PathwayEnsemblePipeline()
    results_df, optimal_weights, accuracy = pipeline.run()
    
    print("\n" + "="*50)
    print("Ensemble Pipeline Completed Successfully!")
    print("="*50)
    print(f"\nOptimal Model Weights:")
    print(f"  DeBERTa: {optimal_weights[0]:.4f}")
    print(f"  Qwen v2: {optimal_weights[1]:.4f}")
    print(f"  Qwen v1: {optimal_weights[2]:.4f}")
    print(f"\nFinal Ensemble Accuracy: {accuracy:.4f}")
    
    # Optional: Evaluate with visualizations
    # evaluate_ensemble(
    #     results_df,
    #     Config.TEST_CSV
    # )