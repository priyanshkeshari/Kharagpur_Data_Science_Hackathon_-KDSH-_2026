import pathway as pw
import json
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
import evaluate

# ==================== Configuration ====================
class Config:
    DATA_DIR = "/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset"
    TRAIN_PATH = "/kaggle/input/kdsh26-train-augmented-dataset/train_augmented_50k_balanced.csv"
    TEST_PATH = f"{DATA_DIR}/test.csv"
    
    MC_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/monte_cristo/monte_cristo_constraints_updated.jsonl"
    CA_CONS_PATH = "/kaggle/input/kdsh26-jsonl-file-characters/castaways/castaways_constraints_filled.jsonl"
    
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    OUTPUT_DIR = "./qwen2.5-7b-books-lora-cls"
    
    # Training hyperparameters
    NUM_EPOCHS = 3
    BATCH_SIZE_TRAIN = 1
    BATCH_SIZE_EVAL = 2
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 5e-5
    WARMUP_RATIO = 0.1
    MAX_LENGTH = 512

# ==================== Constraint Management ====================
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

# Load all constraints globally
mc_constraints = load_constraints_from_jsonl(Config.MC_CONS_PATH)
ca_constraints = load_constraints_from_jsonl(Config.CA_CONS_PATH)
ALL_CONSTRAINTS = {**mc_constraints, **ca_constraints}

# ==================== Pathway UDFs ====================
@pw.udf
def constraint_to_sentence(book: str, char: str, constraint_json: str) -> str:
    """Convert constraint dictionary to natural language sentence"""
    c = json.loads(constraint_json)
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

@pw.udf
def build_context_from_constraints(book_name: str, char: str, max_cons: int = 6) -> str:
    """Build context string from character constraints"""
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
def format_nli_text(context: str, content: str) -> str:
    """Format text for NLI task in premise-hypothesis format"""
    ctx = context if context else ""
    return f"Premise: {ctx}\nHypothesis: {content}"

@pw.udf
def encode_label(label: str) -> int:
    """Encode label string to integer"""
    label2id = {"consistent": 0, "contradict": 1}
    return label2id[label]

# ==================== Model Setup ====================
class QwenNLIModel:
    """Qwen2.5 model wrapper with LoRA for sequence classification"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label2id = {"consistent": 0, "contradict": 1}
        self.id2label = {0: "consistent", 1: "contradict"}
        
    def setup_model(self):
        """Initialize tokenizer and model with 4-bit quantization + LoRA"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model with 4-bit quantization...")
        # 4-bit quantization config (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load base model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        
        # Configure label mappings
        self.model.config.label2id = self.label2id
        self.model.config.id2label = self.id2label
        
        print("Attaching LoRA adapters...")
        # Attach LoRA for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer

# ==================== Pathway Data Pipeline ====================
class PathwayNLIDataPipeline:
    """Pathway-based data processing pipeline for NLI fine-tuning"""
    
    def __init__(self, model_wrapper: QwenNLIModel):
        self.model_wrapper = model_wrapper
        self.tokenizer = model_wrapper.tokenizer
        
    def create_training_table(self, train_path: str):
        """Create Pathway table for training data with preprocessing"""
        
        # Read CSV as Pathway table
        train_table = pw.io.csv.read(
            train_path,
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
        
        # Add context using constraints
        train_with_context = train_table.select(
            id=train_table.id,
            book_name=train_table.book_name,
            char=train_table.char,
            content=train_table.content,
            label=train_table.label,
            context=build_context_from_constraints(train_table.book_name, train_table.char)
        )
        
        # Format for NLI
        train_formatted = train_with_context.select(
            id=pw.this.id,
            text=format_nli_text(pw.this.context, pw.this.content),
            label=pw.this.label,
            label_id=encode_label(pw.this.label)
        )
        
        return train_formatted
    
    def create_test_table(self, test_path: str):
        """Create Pathway table for test data"""
        
        test_table = pw.io.csv.read(
            test_path,
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
            context=build_context_from_constraints(test_table.book_name, test_table.char)
        )
        
        # Format for NLI
        test_formatted = test_with_context.select(
            id=pw.this.id,
            text=format_nli_text(pw.this.context, pw.this.content)
        )
        
        return test_formatted
    
    def table_to_dataset(self, pw_table, output_path="/tmp/pw_data.csv"):
        """Convert Pathway table to HuggingFace Dataset via CSV export"""
        
        # Write Pathway table to CSV
        pw.io.csv.write(pw_table, output_path)
        pw.run()
        
        # Load as pandas then convert to Dataset
        import pandas as pd
        df = pd.read_csv(output_path)
        
        return df

# ==================== Training Components ====================
class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss"""
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

class StepLoggingCallback(TrainerCallback):
    """Callback to log training progress at each step"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            loss = state.log_history[-1].get('loss', 'N/A')
            if loss != 'N/A':
                print(f"[Step {state.global_step}] Loss: {loss:.4f}")

# ==================== Main Training Pipeline ====================
class PathwayQwenTrainingPipeline:
    """Complete Pathway-based training pipeline for Qwen2.5 NLI"""
    
    def __init__(self):
        self.config = Config()
        self.model_wrapper = QwenNLIModel()
        self.data_pipeline = None
        
    def setup(self):
        """Initialize model and data pipeline"""
        print("Setting up model...")
        self.model_wrapper.setup_model()
        self.data_pipeline = PathwayNLIDataPipeline(self.model_wrapper)
        
    def prepare_data(self):
        """Process data using Pathway pipelines"""
        print("Creating Pathway data tables...")
        
        # Create Pathway tables with transformations
        train_table = self.data_pipeline.create_training_table(Config.TRAIN_PATH)
        test_table = self.data_pipeline.create_test_table(Config.TEST_PATH)
        
        # Convert to pandas DataFrames for HuggingFace integration
        train_df = self.data_pipeline.table_to_dataset(train_table, "/tmp/train_processed.csv")
        test_df = self.data_pipeline.table_to_dataset(test_table, "/tmp/test_processed.csv")
        
        return train_df, test_df
    
    def tokenize_data(self, train_df, test_df, val_split=0.2):
        """Tokenize and create train/val splits"""
        from sklearn.model_selection import train_test_split
        
        tokenizer = self.model_wrapper.tokenizer
        
        def encode_batch(df, is_train=True):
            texts = df["text"].fillna("").tolist()
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.MAX_LENGTH,
            )
            if is_train:
                enc["labels"] = df["label_id"].tolist()
            return enc
        
        # Train/val split
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_split,
            stratify=train_df["label"],
            random_state=42,
        )
        
        # Encode
        train_enc = encode_batch(train_split, is_train=True)
        val_enc = encode_batch(val_split, is_train=True)
        test_enc = encode_batch(test_df, is_train=False)
        
        # Convert to HuggingFace Datasets
        train_ds = Dataset.from_dict(train_enc)
        val_ds = Dataset.from_dict(val_enc)
        test_ds = Dataset.from_dict(test_enc)
        
        return train_ds, val_ds, test_ds, train_split
    
    def compute_class_weights(self, train_split):
        """Calculate class weights for imbalanced data"""
        label2id = self.model_wrapper.label2id
        train_label_ids = train_split["label"].map(label2id).values
        
        class_counts = np.bincount(train_label_ids)
        num_classes = len(class_counts)
        
        # Inverse-frequency weights
        weights = class_counts.sum() / (num_classes * class_counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        if torch.cuda.is_available():
            class_weights = class_weights.to("cuda")
        
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights}")
        
        return class_weights
    
    def train(self):
        """Execute complete training pipeline"""
        
        # Setup
        self.setup()
        
        # Prepare data with Pathway
        train_df, test_df = self.prepare_data()
        
        # Tokenize and split
        train_ds, val_ds, test_ds, train_split = self.tokenize_data(train_df, test_df)
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_split)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=Config.BATCH_SIZE_EVAL,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=0.01,
            warmup_ratio=Config.WARMUP_RATIO,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            bf16=True,
            gradient_checkpointing=True,
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            disable_tqdm=False,
            report_to="none",
        )
        
        # Metrics
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            return {
                "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
                "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            }
        
        # Initialize trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.model_wrapper.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[StepLoggingCallback()],
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save
        print(f"Saving model to {Config.OUTPUT_DIR}...")
        trainer.model.save_pretrained(Config.OUTPUT_DIR)
        self.model_wrapper.tokenizer.save_pretrained(Config.OUTPUT_DIR)
        
        # Save label mapping
        label_map_path = Path(Config.OUTPUT_DIR) / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump({
                "label2id": self.model_wrapper.label2id,
                "id2label": self.model_wrapper.id2label
            }, f)
        
        print("Training complete!")
        
        return trainer

# ==================== Execution ====================
if __name__ == "__main__":
    # Create and run pipeline
    pipeline = PathwayQwenTrainingPipeline()
    trainer = pipeline.train()
    
    print("\nTraining finished successfully!")
    print(f"Model saved to: {Config.OUTPUT_DIR}")