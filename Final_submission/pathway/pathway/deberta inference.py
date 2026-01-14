import pathway as pw
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

# Model configuration
MODEL_DIR = "/kaggle/input/kdsh26-deberta-v3-base-fine-tune-model/deberta-v3-base-nli/checkpoint-40"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Model loaded on {device}")
print(f"Label mapping: {model.config.id2label}")

# Load constraints from JSONL files
def load_constraints_dict(path):
    """Load constraints from JSONL file into a dictionary"""
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

MC_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/monte_cristo/monte_cristo_constraints_updated.jsonl"
CA_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/castaways/castaways_constraints_filled.jsonl"

mc_constraints = load_constraints_dict(MC_CONS_PATH)
ca_constraints = load_constraints_dict(CA_CONS_PATH)
all_constraints = {**mc_constraints, **ca_constraints}

# Helper functions
def constraint_to_sentence(book, char, c):
    """Convert constraint dict to natural language sentence"""
    dim = c["dimension"]
    val = c["value"]
    
    if dim == "health_state":
        return f"In {book}, {char} is {val}."
    elif dim == "family_role":
        return f"In {book}, {char} has family role: {val}."
    elif dim == "role":
        return f"In {book}, {char} is described as {val}."
    elif dim == "geographic_expertise":
        return f"{char} is familiar with {val}."
    elif dim == "criminal_history":
        return f"{char} has criminal history: {val}."
    else:
        return f"{dim}: {val}."

def build_context(book, char, max_cons=6):
    """Build context from character constraints"""
    cons = all_constraints.get((book, char), [])
    if not cons:
        return ""
    sents = [constraint_to_sentence(book, char, c) for c in cons[:max_cons]]
    return " ".join(sents)

# Define Pathway UDFs (User Defined Functions)
@pw.udf
def add_context(book_name: str, char: str) -> str:
    """Pathway UDF to add context based on book and character"""
    return build_context(book_name, char)

@pw.udf
def format_inference_text(context: str, content: str) -> str:
    """Pathway UDF to format text for NLI inference"""
    ctx = context if context else ""
    return f"Premise: {ctx}\nHypothesis: {content}"

@pw.udf
def predict_label(text: str) -> str:
    """Pathway UDF to perform NLI inference"""
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        logits = model(**enc).logits
        pred_id = logits.argmax(dim=-1).item()
    
    # Map id to label
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return id2label[pred_id]

# Define Pathway streaming pipeline
class NLIInferencePipeline:
    """Pathway pipeline for real-time NLI inference"""
    
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
    
    def run(self):
        """Execute the Pathway pipeline"""
        
        # Read input data as a streaming table
        input_table = pw.io.csv.read(
            self.input_path,
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=int),
                "book_name": pw.column_definition(dtype=str),
                "char": pw.column_definition(dtype=str),
                "content": pw.column_definition(dtype=str),
            }),
            mode="static"  # Use "streaming" for real-time processing
        )
        
        # Apply transformations using Pathway
        result_table = input_table.select(
            id=input_table.id,
            book_name=input_table.book_name,
            char=input_table.char,
            content=input_table.content,
            # Add context based on book and character
            context=add_context(input_table.book_name, input_table.char)
        ).select(
            id=pw.this.id,
            # Format text for inference
            inference_text=format_inference_text(pw.this.context, pw.this.content)
        ).select(
            id=pw.this.id,
            # Perform NLI prediction
            label=predict_label(pw.this.inference_text)
        )
        
        # Write output
        pw.io.csv.write(result_table, self.output_path)
        
        # Run the computation
        pw.run()
        
        print(f"Pipeline completed! Results written to {self.output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize and run pipeline
    TEST_PATH = "/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset/train.csv"
    OUTPUT_PATH = "submission_pathway.csv"
    
    pipeline = NLIInferencePipeline(TEST_PATH, OUTPUT_PATH)
    pipeline.run()
    
    print("Inference complete!")