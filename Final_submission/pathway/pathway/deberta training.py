import pathway as pw
import json
from pathlib import Path
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import os

# ==================== Configuration ====================
class Config:
    """Centralized configuration for DeBERTa fine-tuning"""
    
    # Paths
    DATA_DIR = "/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset"
    TRAIN_PATH = f"{DATA_DIR}/train.csv"
    TEST_PATH = f"{DATA_DIR}/test.csv"
    
    MC_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/monte_cristo/monte_cristo_constraints_updated.jsonl"
    CA_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/castaways/castaways_constraints_filled.jsonl"
    
    # Model
    MODEL_NAME = "microsoft/deberta-v3-base"
    OUTPUT_DIR = "./deberta-v3-base-nli"
    SAVE_DIR = "./kaggle/working/deberta-v3-base-castaways-monte"
    
    # Training hyperparameters
    NUM_EPOCHS = 8
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_EVAL = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 512
    VAL_SPLIT_SIZE = 0.2
    LOGGING_STEPS = 10

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# ==================== Constraint Loading ====================
def load_constraints_from_jsonl(path):
    """Load character constraints from JSONL file into dictionary"""
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

# Load constraints globally for UDF access
mc_constraints = load_constraints_from_jsonl(Config.MC_CONS_PATH)
ca_constraints = load_constraints_from_jsonl(Config.CA_CONS_PATH)
ALL_CONSTRAINTS = {**mc_constraints, **ca_constraints}

# ==================== Pathway UDFs ====================
@pw.udf
def constraint_to_sentence_udf(book: str, char: str, dimension: str, value: str) -> str:
    """Convert a single constraint to natural language sentence"""
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

@pw.udf
def format_premise_hypothesis(context: str, content: str) -> str:
    """Format as Premise-Hypothesis for NLI task"""
    ctx = context if context else ""
    return f"Premise: {ctx}\nHypothesis: {content}"

@pw.udf
def encode_label_udf(label: str) -> int:
    """Convert label string to integer ID"""
    label2id = {"consistent": 0, "contradict": 1}
    return label2id[label]

# ==================== Pathway Data Pipeline ====================
class PathwayDebertaDataPipeline:
    """Pathway-based data processing pipeline for DeBERTa NLI fine-tuning"""
    
    def __init__(self):
        self.config = Config()
        
    def create_training_table(self):
        """Create Pathway table with training data preprocessing"""
        
        # Read training CSV as Pathway table
        train_table = pw.io.csv.read(
            self.config.TRAIN_PATH,
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=int),
                "book_name": pw.column_definition(dtype=str),
                "char": pw.column_definition(dtype=str),
                "caption": pw.column_definition(dtype=str, optional=True),
                "content": pw.column_definition(dtype=str),
                "label": pw.column_definition(dtype=str),
            }),
            mode="static"  # Use "streaming" for real-time processing
        )
        
        # Add context from constraints
        train_with_context = train_table.select(
            id=train_table.id,
            book_name=train_table.book_name,
            char=train_table.char,
            caption=train_table.caption,
            content=train_table.content,
            label=train_table.label,
            context=build_context_udf(train_table.book_name, train_table.char)
        )
        
        # Format for NLI task
        train_formatted = train_with_context.select(
            id=pw.this.id,
            book_name=pw.this.book_name,
            char=pw.this.char,
            context=pw.this.context,
            content=pw.this.content,
            label=pw.this.label,
            text=format_premise_hypothesis(pw.this.context, pw.this.content),
            label_id=encode_label_udf(pw.this.label)
        )
        
        return train_formatted
    
    def create_test_table(self):
        """Create Pathway table with test data preprocessing"""
        
        # Read test CSV as Pathway table
        test_table = pw.io.csv.read(
            self.config.TEST_PATH,
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=int),
                "book_name": pw.column_definition(dtype=str),
                "char": pw.column_definition(dtype=str),
                "caption": pw.column_definition(dtype=str, optional=True),
                "content": pw.column_definition(dtype=str),
            }),
            mode="static"
        )
        
        # Add context
        test_with_context = test_table.select(
            id=test_table.id,
            book_name=test_table.book_name,
            char=test_table.char,
            content=test_table.content,
            context=build_context_udf(test_table.book_name, test_table.char)
        )
        
        # Format for NLI
        test_formatted = test_with_context.select(
            id=pw.this.id,
            book_name=pw.this.book_name,
            char=pw.this.char,
            context=pw.this.context,
            content=pw.this.content,
            text=format_premise_hypothesis(pw.this.context, pw.this.content)
        )
        
        return test_formatted
    
    def export_to_dataframe(self, pw_table, output_path="/tmp/pw_export.csv"):
        """Export Pathway table to pandas DataFrame via CSV"""
        
        # Write Pathway table to CSV
        pw.io.csv.write(pw_table, output_path)
        
        # Execute Pathway computation
        pw.run()
        
        # Load as pandas DataFrame
        import pandas as pd
        df = pd.read_csv(output_path)
        
        return df

# ==================== Model Wrapper ====================
class DebertaNLIModel:
    """DeBERTa model wrapper for sequence classification"""
    
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        self.label2id = {"consistent": 0, "contradict": 1}
        self.id2label = {0: "consistent", 1: "contradict"}
        
    def setup(self):
        """Initialize tokenizer and model"""
        print(f"Loading tokenizer from {self.config.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        print(f"Loading model from {self.config.MODEL_NAME}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        
        return self.model, self.tokenizer
    
    def tokenize_dataset(self, df, is_train=True):
        """Tokenize text data for model input"""
        texts = df["text"].fillna("").tolist()
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_LENGTH,
        )
        
        if is_train:
            encodings["labels"] = df["label_id"].tolist()
        
        return encodings

# ==================== Training Pipeline ====================
class PathwayDebertaTrainingPipeline:
    """Complete end-to-end Pathway-based training pipeline for DeBERTa"""
    
    def __init__(self):
        self.config = Config()
        self.data_pipeline = PathwayDebertaDataPipeline()
        self.model_wrapper = DebertaNLIModel()
        
    def prepare_data(self):
        """Process data using Pathway pipelines"""
        print("Creating Pathway data processing pipelines...")
        
        # Create Pathway tables with transformations
        train_table = self.data_pipeline.create_training_table()
        test_table = self.data_pipeline.create_test_table()
        
        # Export to pandas DataFrames
        print("Exporting Pathway tables to DataFrames...")
        train_df = self.data_pipeline.export_to_dataframe(
            train_table, 
            "/tmp/train_pathway.csv"
        )
        test_df = self.data_pipeline.export_to_dataframe(
            test_table, 
            "/tmp/test_pathway.csv"
        )
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print("\nLabel distribution:")
        print(train_df["label"].value_counts())
        
        return train_df, test_df
    
    def create_train_val_split(self, train_df):
        """Create stratified train/validation split"""
        from sklearn.model_selection import train_test_split
        
        train_split, val_split = train_test_split(
            train_df,
            test_size=self.config.VAL_SPLIT_SIZE,
            stratify=train_df["label"],
            random_state=42,
        )
        
        print(f"Train split: {len(train_split)} samples")
        print(f"Val split: {len(val_split)} samples")
        
        return train_split, val_split
    
    def prepare_datasets(self, train_split, val_split, test_df):
        """Tokenize and create HuggingFace Datasets"""
        
        # Tokenize
        train_enc = self.model_wrapper.tokenize_dataset(train_split, is_train=True)
        val_enc = self.model_wrapper.tokenize_dataset(val_split, is_train=True)
        test_enc = self.model_wrapper.tokenize_dataset(test_df, is_train=False)
        
        # Convert to HuggingFace Dataset objects
        train_ds = Dataset.from_dict(train_enc)
        val_ds = Dataset.from_dict(val_enc)
        test_ds = Dataset.from_dict(test_enc)
        
        return train_ds, val_ds, test_ds
    
    def setup_metrics(self):
        """Setup evaluation metrics"""
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy.compute(
                    predictions=preds, 
                    references=labels
                )["accuracy"],
                "f1_macro": f1.compute(
                    predictions=preds, 
                    references=labels, 
                    average="macro"
                )["f1"],
            }
        
        return compute_metrics
    
    def train(self):
        """Execute complete training pipeline"""
        
        # Setup model
        print("Setting up model...")
        self.model_wrapper.setup()
        
        # Prepare data with Pathway
        print("\n=== Data Preparation with Pathway ===")
        train_df, test_df = self.prepare_data()
        
        # Create train/val split
        print("\n=== Creating Train/Val Split ===")
        train_split, val_split = self.create_train_val_split(train_df)
        
        # Tokenize and create datasets
        print("\n=== Tokenizing Data ===")
        train_ds, val_ds, test_ds = self.prepare_datasets(
            train_split, val_split, test_df
        )
        
        # Setup metrics
        compute_metrics = self.setup_metrics()
        
        # Training arguments
        print("\n=== Configuring Training ===")
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=self.config.BATCH_SIZE_EVAL,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_steps=self.config.LOGGING_STEPS,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.model_wrapper.tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("\n=== Starting Training ===")
        trainer.train()
        
        # Save model
        print(f"\n=== Saving Model to {self.config.SAVE_DIR} ===")
        trainer.save_model(self.config.SAVE_DIR)
        self.model_wrapper.tokenizer.save_pretrained(self.config.SAVE_DIR)
        
        # Save label mapping
        label_map_path = Path(self.config.SAVE_DIR) / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump({
                "label2id": self.model_wrapper.label2id,
                "id2label": self.model_wrapper.id2label
            }, f)
        
        print(f"\nModel saved to: {self.config.SAVE_DIR}")
        print("Training complete!")
        
        return trainer, test_ds

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create and run complete pipeline
    pipeline = PathwayDebertaTrainingPipeline()
    trainer, test_ds = pipeline.train()
    
    print("\n" + "="*50)
    print("Training Pipeline Completed Successfully!")
    print("="*50)
    print(f"\nModel location: {Config.SAVE_DIR}")
    print(f"Test dataset ready with {len(test_ds)} samples")