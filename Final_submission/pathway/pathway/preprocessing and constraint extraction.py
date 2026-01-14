import pathway as pw
import json
import math
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== Configuration ====================
class Config:
    """Centralized configuration for book preprocessing"""
    
    BOOK_NAME = "In Search of the Castaways"
    BOOK_PATH = Path("/kaggle/input/kharagpur-data-science-hackathon-kdsh-2026-dataset/Books/In search of the castaways.txt")
    NAME_COUNTS_CSV = Path("/kaggle/input/kdsh26-name-counts-csv/In_Search_of_the_Castaways_name_counts.csv")
    OUT_PATH = Path("/kaggle/working/castaways_constraints.jsonl")
    LOG_PATH = Path("/kaggle/working/progress_castaways.log")
    
    CHUNK_SIZE = 4000
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MAX_NEW_TOKENS = 512
    
    # Dimensions to extract
    DIMENSIONS = [
        "birthplace",
        "role",
        "geographic_expertise",
        "tribe_or_nation",
        "languages",
        "moral_arc",
        "criminal_history",
        "core_motivation",
    ]
    
    # Character alias mapping
    ALIAS_MAP = {
        "Lord Glenarvan": "Lord Glenarvan",
        "Glenarvan": "Lord Glenarvan",
        "Glenarvan's": "Lord Glenarvan",
        "Captain Grant": "Captain Grant",
        "Grant": "Captain Grant",
        "Robert Grant": "Robert Grant",
        "Robert": "Robert Grant",
        "Mary Grant": "Mary Grant",
        "Mary": "Mary Grant",
        "Lady Helena": "Lady Helena Glenarvan",
        "Lady Helena Glenarvan": "Lady Helena Glenarvan",
        "Helena": "Lady Helena Glenarvan",
        "Jacques Paganel": "Jacques Paganel",
        "Paganel": "Jacques Paganel",
        "John Mangles": "John Mangles",
        "Mangles": "John Mangles",
        "Major MacNabbs": "Major MacNabbs",
        "MacNabb": "Major MacNabbs",
        "MacNabbs": "Major MacNabbs",
        "Wilson": "Wilson",
        "Mulready": "Mulready",
        "Austin": "Austin",
        "Thalcave": "Thalcave",
        "Kai-Koumou": "Kai-Koumou",
        "Tom Ayrton": "Tom Ayrton / Ben Joyce",
        "Tom": "Tom Ayrton / Ben Joyce",
        "Ayrton": "Tom Ayrton / Ben Joyce",
        "Ben Joyce": "Tom Ayrton / Ben Joyce",
        "Ben": "Tom Ayrton / Ben Joyce",
        "Joyce": "Tom Ayrton / Ben Joyce",
    }

# ==================== Logging ====================
def log_progress(msg: str, log_path: Path = Config.LOG_PATH):
    """Print and append progress messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ==================== Text Processing ====================
class BookProcessor:
    """Handle book loading and chunking"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_book(self) -> str:
        """Load book text from file"""
        return self.config.BOOK_PATH.read_text(encoding="utf-8", errors="ignore")
    
    def chunk_book(self, text: str):
        """Split book into chunks"""
        chunk_size = self.config.CHUNK_SIZE
        n = math.ceil(len(text) / chunk_size)
        
        chunks = []
        for i in range(n):
            chunk_text = text[i*chunk_size:(i+1)*chunk_size]
            chunks.append({
                "chunk_id": i,
                "text": chunk_text
            })
        
        return chunks
    
    def detect_present_characters(self, chunk_text: str) -> List[str]:
        """Detect canonical characters present in chunk"""
        found = set()
        
        for surface, canon in self.config.ALIAS_MAP.items():
            # Disambiguate generic tokens
            if surface in {"Tom", "Ben"}:
                if not (re.search(r"\bAyrton\b", chunk_text) or 
                        re.search(r"\bJoyce\b", chunk_text)):
                    continue
            
            pattern = rf"\b{re.escape(surface)}\b"
            if re.search(pattern, chunk_text):
                found.add(canon)
        
        return list(found)

# ==================== Pathway UDFs ====================
@pw.udf
def detect_characters_udf(chunk_text: str) -> str:
    """Detect characters in chunk (returns JSON list)"""
    processor = BookProcessor(Config())
    characters = processor.detect_present_characters(chunk_text)
    return json.dumps(characters)

@pw.udf
def extract_chunk_id_udf(row_index: int) -> int:
    """Extract chunk ID from row index"""
    return row_index

# ==================== LLM Constraint Extraction ====================
class LLMConstraintExtractor:
    """Extract character constraints using LLM"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def setup_model(self):
        """Initialize LLM"""
        log_progress(f"Loading LLM: {self.config.MODEL_NAME}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        log_progress("LLM loaded successfully")
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for LLM"""
        dims_str = ", ".join(self.config.DIMENSIONS)
        
        return f"""
You are extracting factual constraints about ONE character from
Jules Verne's "In Search of the Castaways".

Focus on dimensions:
- role (noble, geographer, sailor, criminal, guide, Maori chief, etc.)
- geographic_expertise (Pampas, Patagonia, seas, Australia, New Zealand)
- tribe_or_nation (Patagonian, Maori, Scottish, French, etc.)
- languages (Spanish, English, French, indigenous languages)
- moral_arc (e.g., criminal_to_redeemed, loyal_helper)
- criminal_history (for Tom Ayrton / Ben Joyce)
- core_motivation (loyalty, duty, search for Captain Grant)

Output STRICT JSON with schema:

{{
  "character": str,  # canonical name
  "book_name": "In Search of the Castaways",
  "constraints": [
    {{
      "dimension": str,    # one of: {dims_str},
      "value": str,
      "polarity": "positive" or "negative",
      "evidence_text": str,
      "chapter_id": str
    }}
  ]
}}

Include only facts clearly stated or strongly implied in the excerpt.
"""
    
    def get_user_prompt(self, character: str, chapter_id: int, text_chunk: str) -> str:
        """Generate user prompt for specific character and chunk"""
        return f"""
Book: In Search of the Castaways
Canonical character: {character}
Chunk id: {chapter_id}

Excerpt:
\"\"\"{text_chunk}\"\"\"
"""
    
    def generate_json_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate LLM response"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.model.device)
        
        eos_id = self.model.config.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        
        gen_ids = output_ids[0, input_ids.shape[-1]:]
        out_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return out_text
    
    def extract_first_json(self, text: str) -> Dict:
        """Extract first JSON object from text"""
        match = re.search(r"\{.*\}", text, flags=re.S)
        candidate = match.group(0) if match else text
        candidate = candidate.strip().strip("`")
        
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    
    def extract_constraints(self, canonical_char: str, chapter_id: int, text_chunk: str) -> Dict:
        """Extract constraints for a character from a chunk"""
        user_prompt = self.get_user_prompt(canonical_char, chapter_id, text_chunk)
        system_prompt = self.get_system_prompt()
        
        raw_text = self.generate_json_response(system_prompt, user_prompt)
        obj = self.extract_first_json(raw_text)
        
        if not isinstance(obj, dict):
            obj = {}
        
        obj.setdefault("character", canonical_char)
        obj.setdefault("book_name", self.config.BOOK_NAME)
        obj.setdefault("constraints", [])
        
        if not isinstance(obj["constraints"], list):
            obj["constraints"] = []
        
        # Clean constraints
        cleaned_constraints = []
        for c in obj["constraints"]:
            if not isinstance(c, dict):
                continue
            
            dim = c.get("dimension")
            val = c.get("value")
            pol = c.get("polarity", "positive")
            evid = c.get("evidence_text", "")
            chap = c.get("chapter_id", str(chapter_id))
            
            if not dim or not val:
                continue
            
            cleaned_constraints.append({
                "dimension": dim,
                "value": val,
                "polarity": pol if pol in ("positive", "negative") else "positive",
                "evidence_text": evid,
                "chapter_id": str(chap),
            })
        
        obj["constraints"] = cleaned_constraints
        return obj

# ==================== Pathway Pipeline ====================
class PathwayPreprocessingPipeline:
    """Pathway-based book preprocessing pipeline"""
    
    def __init__(self):
        self.config = Config()
        self.book_processor = BookProcessor(self.config)
        self.llm_extractor = LLMConstraintExtractor(self.config)
        
    def create_chunks_table(self, text: str):
        """Create Pathway table from book chunks"""
        
        # Get chunks
        chunks = self.book_processor.chunk_book(text)
        
        # Convert to DataFrame for Pathway
        chunks_df = pd.DataFrame(chunks)
        
        # Create Pathway table from DataFrame
        # Note: Pathway doesn't directly support in-memory DataFrames,
        # so we write to CSV first
        temp_path = "/tmp/book_chunks.csv"
        chunks_df.to_csv(temp_path, index=False)
        
        chunks_table = pw.io.csv.read(
            temp_path,
            schema=pw.schema_builder({
                "chunk_id": pw.column_definition(dtype=int),
                "text": pw.column_definition(dtype=str),
            }),
            mode="static"
        )
        
        # Detect characters in each chunk
        chunks_with_chars = chunks_table.select(
            chunk_id=chunks_table.chunk_id,
            text=chunks_table.text,
            characters_json=detect_characters_udf(chunks_table.text)
        )
        
        return chunks_with_chars
    
    def process_chunks(self, chunks_table):
        """Process chunks to extract constraints"""
        
        # Export to DataFrame for processing
        temp_output = "/tmp/chunks_processed.csv"
        pw.io.csv.write(chunks_table, temp_output)
        pw.run()
        
        # Load processed chunks
        chunks_df = pd.read_csv(temp_output)
        
        return chunks_df
    
    def extract_all_constraints(self, chunks_df: pd.DataFrame):
        """Extract constraints from all chunks"""
        
        canonical_chars = sorted(set(self.config.ALIAS_MAP.values()))
        char_to_objs = {c: [] for c in canonical_chars}
        
        total_chunks = len(chunks_df)
        log_progress(f"Starting {self.config.BOOK_NAME} preprocessing: {total_chunks} chunks")
        
        for idx, row in chunks_df.iterrows():
            chunk_id = row['chunk_id']
            chunk_text = row['text']
            characters_json = row['characters_json']
            
            # Parse characters
            present = json.loads(characters_json)
            
            if not present:
                if idx % 25 == 0:
                    log_progress(f"Chunk {idx+1}/{total_chunks}: no target characters found")
                continue
            
            log_progress(
                f"Chunk {idx+1}/{total_chunks} (id={chunk_id}): "
                f"{len(present)} characters present: {present}"
            )
            
            for ci, canon in enumerate(present, start=1):
                try:
                    obj = self.llm_extractor.extract_constraints(canon, chunk_id, chunk_text)
                    char_to_objs[canon].append(obj)
                    log_progress(
                        f"  ↳ Processed character {ci}/{len(present)} in chunk {idx+1}: {canon}"
                    )
                except Exception as e:
                    log_progress(f"  ✗ Error for {canon} chunk {chunk_id}: {e}")
        
        return char_to_objs
    
    def merge_constraints(self, book_name: str, character: str, objs: List[Dict]) -> Dict:
        """Merge constraints from multiple chunks"""
        merged = []
        seen = set()
        
        for obj in objs:
            for c in obj.get("constraints", []):
                key = (c["dimension"], c["value"].strip().lower(), c["polarity"])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(c)
        
        return {
            "book_name": book_name,
            "character": character,
            "constraints": merged,
        }
    
    def run(self):
        """Execute complete preprocessing pipeline"""
        
        # Step 1: Setup LLM
        print("\n=== Loading LLM ===")
        self.llm_extractor.setup_model()
        
        # Step 2: Load and chunk book with Pathway
        print("\n=== Processing Book with Pathway ===")
        text = self.book_processor.load_book()
        log_progress(f"Loaded book: {len(text)} characters")
        
        chunks_table = self.create_chunks_table(text)
        chunks_df = self.process_chunks(chunks_table)
        
        # Step 3: Extract constraints
        print("\n=== Extracting Constraints ===")
        char_to_objs = self.extract_all_constraints(chunks_df)
        
        # Step 4: Merge and save
        print("\n=== Merging and Saving Results ===")
        all_outputs = []
        for canon, objs in char_to_objs.items():
            merged = self.merge_constraints(self.config.BOOK_NAME, canon, objs)
            all_outputs.append(merged)
            log_progress(
                f"Merged constraints for {canon}: {len(merged['constraints'])} constraints"
            )
        
        # Save to JSONL
        with self.config.OUT_PATH.open("w", encoding="utf-8") as f:
            for obj in all_outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
        log_progress(f"Saved constraints to {self.config.OUT_PATH}")
        
        return all_outputs

# ==================== Main Execution ====================
if __name__ == "__main__":
    pipeline = PathwayPreprocessingPipeline()
    results = pipeline.run()
    
    print("\n" + "="*50)
    print("Book Preprocessing Completed Successfully!")
    print("="*50)
    print(f"\nExtracted constraints for {len(results)} characters")
    print(f"Output saved to: {Config.OUT_PATH}")