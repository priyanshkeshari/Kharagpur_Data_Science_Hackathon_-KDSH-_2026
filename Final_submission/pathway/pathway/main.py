import os
import gc
import json
import torch
import numpy as np
import pandas as pd
import pathway as pw
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel

# ==================== CONFIGURATION ====================
class UnifiedConfig:
    # Model Paths - Update these to your actual Kaggle input paths
    DEBERTA_BASE = "microsoft/deberta-v3-base"
    DEBERTA_FINETUNED = "/kaggle/input/your-finetuned-models/deberta-v3-base-nli"
    QWEN_BASE = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_ADAPTER_V1 = "/kaggle/input/your-finetuned-models/qwen2.5-7b-adapter-v1"
    QWEN_ADAPTER_V2 = "/kaggle/input/your-finetuned-models/qwen2.5-7b-adapter-v2"
    
    # Vector Store Paths (from Pathway storage output)
    VECTOR_STORE_DIR = "/kaggle/input/your-vector-store-dataset/pathway_storage"
    METADATA_FILE = f"{VECTOR_STORE_DIR}/vector_store_metadata.json"
    
    # Ensemble Settings: [DeBERTa, Qwen_V2, Qwen_V1]
    ENSEMBLE_WEIGHTS = np.array([0.98402039, 0.00670148, 0.00927813])
    THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== GLOBAL MODEL MANAGER ====================
# We use a singleton-style manager to ensure models are loaded once 
# and accessible within Pathway UDFs.
class ModelManager:
    _instance = None
    def __new__(cls, cfg):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init_models(cfg)
        return cls._instance

    def _init_models(self, cfg):
        print("🚀 Initializing Models into GPU Memory...")
        # 1. Retrieval Model
        self.embedder = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            device=cfg.DEVICE, trust_remote_code=True,
            config_kwargs={"use_cache": False}
        )
        
        # 2. Classifier: DeBERTa
        self.deb_tokenizer = AutoTokenizer.from_pretrained(cfg.DEBERTA_BASE)
        self.deb_model = AutoModelForSequenceClassification.from_pretrained(cfg.DEBERTA_FINETUNED).to(cfg.DEVICE).eval()
        
        # 3. Classifier & Generator: Qwen
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(cfg.QWEN_BASE)
        self.base_qwen = AutoModelForCausalLM.from_pretrained(
            cfg.QWEN_BASE, torch_dtype=torch.float16, device_map="auto"
        )
        self.qwen_cls = PeftModel.from_pretrained(self.base_qwen, cfg.QWEN_ADAPTER_V2, adapter_name="v2")
        self.qwen_cls.load_adapter(cfg.QWEN_ADAPTER_V1, adapter_name="v1")
        
        # 4. Evidence Data
        with open(cfg.METADATA_FILE, 'r') as f:
            meta = json.load(f)
        self.embeddings = {}
        self.chunks_df = {}
        for book in meta['books'].keys():
            self.embeddings[book] = np.load(f"{cfg.VECTOR_STORE_DIR}/embeddings_{book}.npy")
            self.chunks_df[book] = pd.read_csv(f"{cfg.VECTOR_STORE_DIR}/chunks_{book}.csv")

# ==================== PATHWAY UDFS ====================

@pw.udf
def get_evidence_and_predict(char: str, content: str, bookname: str, caption: str) -> str:
    cfg = UnifiedConfig()
    mgr = ModelManager(cfg)
    
    # 1. Normalize Book Name
    norm_name = bookname.strip().replace(' ', '-')
    parts = norm_name.split('-')
    if parts:
        parts[0] = parts[0].capitalize()
        parts[1:] = [p.lower() for p in parts[1:]]
    norm_name = '-'.join(parts)
    
    # 2. Retrieval
    query = f"Character: {char}. Statement: {content}"
    evidence = "No specific evidence found."
    if norm_name in mgr.embeddings:
        q_emb = mgr.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = np.dot(mgr.embeddings[norm_name], q_emb)
        top_idx = np.argsort(sims)[-3:][::-1]
        evidence = " ".join(mgr.chunks_df[norm_name].iloc[top_idx]['chunk'].tolist())

    # 3. Classification (Ensemble)
    # DeBERTa
    nli_text = f"Premise: {evidence}\nHypothesis: {content}"
    deb_in = mgr.deb_tokenizer(nli_text, return_tensors="pt", truncation=True, max_length=512).to(cfg.DEVICE)
    with torch.no_grad():
        p_deb = torch.softmax(mgr.deb_model(**deb_in).logits, dim=-1).cpu().numpy()[0][1]
    
    # Placeholder for Qwen Classifiers (as per ensemble_inference logic)
    p_v2, p_v1 = 0.5, 0.5 
    
    final_prob = np.dot([p_deb, p_v2, p_v1], cfg.ENSEMBLE_WEIGHTS)
    label = "contradict" if final_prob > cfg.THRESHOLD else "consistent"
    
    # 4. Rationale Generation
    prompt = f"Book: {bookname}\nChar: {char}\nStatement: {content}\nEvidence: {evidence[:500]}\nLabel: {label}\nRationale:"
    messages = [{"role": "user", "content": prompt}]
    input_text = mgr.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = mgr.qwen_tokenizer([input_text], return_tensors="pt").to(cfg.DEVICE)
    
    with torch.no_grad():
        outputs = mgr.base_qwen.generate(**inputs, max_new_tokens=100)
    
    rationale = mgr.qwen_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Return as JSON string to unpack in the pipeline
    return json.dumps({"label": label, "rationale": rationale.strip()})

# ==================== PIPELINE DEFINITION ====================

def run_pathway_pipeline(input_path: str, output_path: str):
    # Define Input Schema
    class InputSchema(pw.Schema):
        id: int
        bookname: str
        char: str
        content: str
        caption: str

    # 1. Read input stream
    # Note: Use pw.io.csv.read for streaming or pw.debug.table_from_pandas for static
    input_table = pw.io.csv.read(input_path, schema=InputSchema)
    
    # 2. Process via UDF
    # This runs the heavy lifting inside a single Pathway transformation
    processed = input_table.select(
        id=input_table.id,
        raw_output=get_evidence_and_predict(
            input_table.char, 
            input_table.content, 
            input_table.bookname, 
            input_table.caption
        )
    )
    
    # 3. Unpack the JSON result
    final_table = processed.select(
        id=processed.id,
        Prediction=pw.apply(lambda x: json.loads(x)["label"], processed.raw_output),
        Rationale=pw.apply(lambda x: json.loads(x)["rationale"], processed.raw_output)
    )
    
    # 4. Write Output
    pw.io.csv.write(final_table, output_path)
    
    # 5. Execute
    print("🔥 Starting Pathway Reactive Pipeline...")
    pw.run()

if __name__ == "__main__":
    # Ensure Global Manager is initialized before running the graph
    cfg = UnifiedConfig()
    ModelManager(cfg)
    
    run_pathway_pipeline("test.csv", "submission.csv")
    print("✅ Inference complete. Check submission.csv")