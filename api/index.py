import os
import re
import ast
import requests
import uuid
import time
import threading
import concurrent.futures
import json
import base64
import io
from flask import Flask, request, jsonify, render_template_string, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import requests
import concurrent.futures

# Performance: Global Connection Pooling
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=50)
http_session.mount('https://', adapter)
http_session.mount('http://', adapter)

try: 
    from PyPDF2 import PdfReader
    from PIL import Image
    import hashlib
except ImportError: 
    PdfReader = None
    Image = None

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32MB Limit

import math

# --------------------------
# SEMANTIC PERSISTENT MEMORY
# --------------------------
MEMORY_FILE = "lyra_memory.json"
memory_lock = threading.Lock()

def load_persistent_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"user_context": "", "semantic_bank": []}
    return {"user_context": "", "semantic_bank": []}

def save_persistent_memory(data):
    with memory_lock:
        try:
            with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        except Exception as e: print(f"Memory Save Error: {e}")

memory = load_persistent_memory()
if "semantic_bank" not in memory: memory["semantic_bank"] = []
if "document_store" not in memory: memory["document_store"] = []
if "config" not in memory: memory["config"] = {"response_style": "standard", "personality": "helpful", "speed_priority": True}
if "proposals" not in memory: memory["proposals"] = []
# --------------------------
# UNIFIED OS CORE STATE
# --------------------------
GLOBAL_OS_STATE = {
    "active_budget": 50000.0, # Target default INR
    "planned_expenses": 0.0,
    "current_trip": {
        "destination": None,
        "distance_km": 0,
        "travel_time_hrs": 0,
        "total_estimated_cost": 0.0,
        "itinerary": []
    },
    "market_sentiment": "Neutral",
    "last_calculation": {
        "expression": "",
        "result": "0"
    },
    "active_parameters": {},
    "last_sync": time.time()
}

def sync_system_state(updates):
    global GLOBAL_OS_STATE
    for key, value in updates.items():
        if key in GLOBAL_OS_STATE:
            if isinstance(value, dict) and isinstance(GLOBAL_OS_STATE[key], dict):
                GLOBAL_OS_STATE[key].update(value)
            else:
                GLOBAL_OS_STATE[key] = value
    GLOBAL_OS_STATE["last_sync"] = time.time()
    return GLOBAL_OS_STATE

query_cache = {}
memory = load_persistent_memory()

def analyze_self():
    print("DEBUG [Evolution]: Starting self-analysis...")
    feedback_pool = [m["text"] for m in memory["semantic_bank"][-20:]]
    context = "\n".join(feedback_pool)
    prompt = f"Analyze this conversation log and your current config: {memory['config']}.\nIdentify 3 specific ways to improve your performance or behavior based on user interaction patterns. Return ONLY a JSON list of objects: [{{\"action\": \"description\", \"value\": \"change\"}}]"
    
    try:
        res = mistral_call(prompt, [], "Lyra Metacognition Engine")
        # Clean JSON and parse
        import re
        match = re.search(r'\[.*\]', res, re.DOTALL)
        if match:
            new_proposals = json.loads(match.group(0))
            memory["proposals"] = new_proposals[:3]
            save_persistent_memory(memory)
            return True
    except Exception as e:
        print(f"Evolution Error: {e}")
    return False

def apply_change(action, value):
    # Safe backup
    try:
        import shutil
        shutil.copy(MEMORY_FILE, MEMORY_FILE + ".bak")
        
        # Mapping logic
        if action == "response_style": memory["config"]["response_style"] = value
        elif action == "personality": memory["config"]["personality"] = value
        elif action == "speed": memory["config"]["speed_priority"] = (value == "true")
        
        save_persistent_memory(memory)
        return True
    except: return False

# --------------------------
# VECTOR MATH & EMBEDDINGS
# --------------------------
def get_embedding(text):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None
    url = f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key={api_key}"
    try:
        payload = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
        resp = requests.post(url, json=payload, timeout=5)
        return resp.json().get('embedding', {}).get('values')
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def cosine_similarity(v1, v2):
    if not v1 or not v2: return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if not mag1 or not mag2: return 0.0
    return dot / (mag1 * mag2)

def retrieve_relevant_context(query, bank_key="semantic_bank", limit=3, threshold=0.65):
    q_emb = get_embedding(query)
    if not q_emb or not memory.get(bank_key): return []
    
    scores = []
    for entry in memory[bank_key]:
        score = cosine_similarity(q_emb, entry.get("embedding"))
        scores.append((score, entry.get("text")))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top = [s[1] for s in scores if s[0] > threshold][:limit]
    if top: print(f"DEBUG: RAG ({bank_key}) retrieved {len(top)} slices. Top Similarity: {scores[0][0]:.4f}")
    return top

def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_file_rag(file_data, filename):
    ext = filename.split('.')[-1].lower()
    text_content = ""
    
    if ext in ['jpg', 'jpeg', 'png']:
        print(f"DEBUG: Performing RAG Vision Ingestion on {filename}...")
        text_content = analyze_image(file_data, filename)
    else:
        text_content = extract_text_from_file(file_data, filename)

    if not text_content or len(text_content.strip()) < 5: return

    chunks = chunk_text(text_content)
    print(f"DEBUG: Chunking {filename} into {len(chunks)} fragments...")
    
    source_tag = "[DOCUMENT SOURCE]" if ext not in ['jpg', 'jpeg', 'png'] else "[IMAGE SOURCE]"
    
    with memory_lock:
        for chunk in chunks:
            emb = get_embedding(chunk)
            if not emb: continue
            memory["document_store"].append({"text": f"{source_tag} {filename}: {chunk}", "embedding": emb})
            if len(memory["document_store"]) > 200: memory["document_store"].pop(0)
    save_persistent_memory(memory)

VISION_CACHE = {}
VISION_JOBS = {}

def optimize_image_data(file_data):
    if not Image: return file_data
    try:
        img = Image.open(io.BytesIO(file_data))
        # Ensure RGB (converts RGBA/CMYK/etc)
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((512, 512))
        out = io.BytesIO()
        img.save(out, format="JPEG", optimize=True, quality=85)
        return out.getvalue()
    except: return file_data

# --------------------------
# HIGH-PRECISION MATH ENGINE
# --------------------------
CALC_VARS = {}

def normalize_math_expr(expr):
    """Auto-fixes common mathematical notation errors."""
    import re
    e = expr.strip()
    # Fix 2(3) -> 2*(3)
    e = re.sub(r'(\d)\s*\(', r'\1*(', e)
    # Fix (3)2 -> (3)*2
    e = re.sub(r'\)\s*(\d)', r')*\1', e)
    # Fix 2x -> 2*x (if x is pi or e)
    e = re.sub(r'(\d)\s*(pi|e|sin|cos|tan|sqrt)', r'\1*\2', e)
    return e

def generate_math_steps(expr, result):
    """Creates a logical step-by-step breakdown for simple arithmetic."""
    try:
        # Detect operators
        ops = []
        if "(" in expr: ops.append("Solving parentheses first.")
        if "**" in expr or "^" in expr: ops.append("Evaluating exponents/roots.")
        if "*" in expr or "/" in expr: ops.append("Processing multiplication/division.")
        if "+" in expr or "-" in expr: ops.append("Finalizing addition/subtraction.")
        
        steps = f"**Logic Path**:\n- " + "\n- ".join(ops)
        return f"**Answer**: {result}\n\n{steps}\n\n`{expr} = {result}`"
    except: return f"**Answer**: {result}"

def scientific_eval(expr):
    """Highly optimized and secure scientific evaluator."""
    import math
    global CALC_VARS
    
    # 1. Handle Variable Assignment (e.g., a = 5)
    if "=" in expr and not ("==" in expr or ">=" in expr or "<=" in expr or "!=" in expr):
        try:
            parts = expr.split("=")
            var_name = parts[0].strip()
            var_val_expr = parts[1].strip()
            # Safety check on var name
            if not var_name.isidentifier(): return "Error: Invalid Var Name"
            res = scientific_eval(var_val_expr)
            CALC_VARS[var_name] = res
            return f"{var_name} set to {res}"
        except: return "Error: Assignment Failed"

    # 2. Normalize and Map
    e = normalize_math_expr(expr)
    clean_expr = e.replace("^", "**").replace("sqrt", "math.sqrt").replace("sin", "math.sin").replace("cos", "math.cos").replace("tan", "math.tan")
    
    # 3. Environment Context
    safe_dict = {
        "math": math, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, **CALC_VARS
    }
    
    # Allowed character whitelist for security
    if any(m in clean_expr for m in ["import", "os", "sys", "open", "eval", "exec", "getattr", "setattr"]): return None
    
    try:
        res = eval(clean_expr, {"__builtins__": None}, safe_dict)
        return res
    except: return None

def parse_spoken_math(text):
    """High-precision regex parser for spoken mathematical queries."""
    import re
    t = text.lower()
    mapping = [
        (r'square root of\s*(\d+)', r'sqrt(\1)'),
        (r'(\d+)\s+whole square', r'(\1)^2'),
        (r'(\d+)\s+power\s+(\d+)', r'\1^\2'),
        (r'(\d+)\s+raised to\s+(\d+)', r'\1^\2'),
        (r'sine of\s*(\d+)', r'sin(\1)'),
        (r'cosine of\s*(\d+)', r'cos(\1)'),
        (r'plus', '+'), (r'minus', '-'), (r'into', '*'), (r'multiplied by', '*'), (r'divided by', '/'), (r'over', '/')
    ]
    for pattern, replacement in mapping:
        t = re.sub(pattern, replacement, t)
    return t.strip()

def ai_math_solvent(query):
    """Orchestrates between Instant Math Engine and AI Word Problem Solver."""
    # 1. Attempt Instant Local Solve
    res = scientific_eval(query)
    if res is not None and not isinstance(res, str):
        return generate_math_steps(query, res)
    elif isinstance(res, str) and res.startswith("Error"):
        return res
        
    # 2. Fallback to AI for Word Problems
    sys = "You are the Logic Catalyst. Decompose word problems into pure math and solve. Format: [EXPR] ... [EXPL] ..."
    raw = fast_orchestrator(query, [], sys)
    
    if "[EXPR]" in raw:
        parts = raw.split("[EXPL]")
        expr = parts[0].replace("[EXPR]", "").strip()
        expl = parts[1].strip() if len(parts) > 1 else ""
        res_val = scientific_eval(expr)
        if res_val is not None:
            return f"**Answer**: {res_val}\n\n**Neural Breakdown**:\n{expl}\n\n`{expr} = {res_val}`"
    
    return fast_orchestrator(query, [], "Expert solve.")

# --------------------------
# MULTI-AGENT SWARM (MAS)
# --------------------------
def planner_agent(goal):
    prompt = f"Goal: {goal}\nBreak this into 3-5 distinct steps. Return as a numbered list. Example: 1. Search X | 2. Creative Y | 3. Logic Z"
    res = groq_call(prompt, [], "🧠 Lead Planner Agent")
    return [s.strip() for s in res.split('\n') if s.strip() and s[0].isdigit()][:5]

def creative_agent(task):
    prompt = f"Perform this imaginative task: {task}. Be expressive, visionary, and creative."
    return fast_orchestrator(prompt, [], "🎨 Creative Specialist")

def logic_agent(task):
    prompt = f"Analyze this task with precision: {task}. Be concise, logical, and evidence-based."
    return fast_orchestrator(prompt, [], "📊 Logic Specialist")

def executor_agent(goal, results):
    prompt = f"Goal: {goal}\nReview these cross-agent reports:\n{results}\nSynthesize into a premium, formatted final deliverable. Remove redundancies."
    return fast_orchestrator(prompt, [], "⚙️ System Executor")

def run_agent_swarm(goal):
    print(f"DEBUG [Swarm]: Planning goal: {goal}")
    steps = planner_agent(goal)
    results_map = {}
    
    def process_swarm_step(step):
        # SKILL-BASED ROUTING
        if any(w in step.lower() for w in ["search", "find", "latest"]):
            return f"**[SEARCH]**: {serp_search(step)[:300]}"
        elif any(w in step.lower() for w in ["image", "draw", "art", "design", "write", "creative"]):
            return f"**[CREATIVE]**: {creative_agent(step)}"
        else:
            return f"**[LOGIC]**: {logic_agent(step)}"

    # PARALLEL EXECUTION
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as swarm:
        futures = {swarm.submit(process_swarm_step, s): s for s in steps}
        for future in concurrent.futures.as_completed(futures, timeout=12):
            try:
                task_text = futures[future]
                res = future.result()
                final_results.append(f"### {task_text}\n{res}\n")
            except: continue
            
    swarm_report = "\n".join(final_results)
    return executor_agent(goal, swarm_report)

# --------------------------
# REASONING & VISION SYNTHESIS
# --------------------------
def reason_about_image(structured_data, query, mode_prompt=""):
    system = "You are an Elite Image Reasoning Engine. Use the provided structured image data to answer the user query logically and accurately. Reason step-by-step."
    prompt = f"IMAGE DATA:\n{structured_data}\n\nUSER QUERY: {query}"
    return fast_orchestrator(prompt, [], system)

def analyze_image(file_data, filename, deep=True):
    h = hashlib.md5(file_data).hexdigest()
    if h in VISION_CACHE and not deep:
        print(f"DEBUG [Vision]: Cache hit for {filename}")
        return VISION_CACHE[h]

    start = time.time()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "Error: API Key missing."

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite-preview-02-05:generateContent?key={api_key}"
    
    try:
        proc_data = optimize_image_data(file_data)
        b64 = base64.b64encode(proc_data).decode('utf-8')
        
        # STRUCTURED EXTRACTION PROMPT
        prompt = """Analyze this image and provide a structured breakdown. Return only a mini-summary followed by specific labels.
Format:
SUMMARY: [One sentence overview]
OBJECTS: [list]
SCENE: [description]
TEXT: [any visible text]
ACTIONS: [interactions detected]"""

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                ]
            }]
        }
        
        resp = http_session.post(url, json=payload, timeout=12)
        resp.raise_for_status()
        raw_analysis = resp.json()['candidates'][0]['content']['parts'][0]['text']
        
        VISION_CACHE[h] = raw_analysis
        print(f"DEBUG [Vision]: Analysis complete in {time.time()-start:.2f}s")
        return raw_analysis
        
    except Exception as e:
        print(f"ERROR [Vision]: {e}")
        return "Visual metadata extraction failed."

def add_to_semantic_memory(text):
    if len(text.strip()) < 15: return
    emb = get_embedding(text)
    if not emb: return
    memory["semantic_bank"].append({"text": text, "embedding": emb})
    if len(memory["semantic_bank"]) > 50: memory["semantic_bank"].pop(0)
    save_persistent_memory(memory)

SYSTEM_PROMPT = """You are Lyra, an intelligent AI creation suite assistant.

Your personality
* Clear and confident:
* Slightly conversational, not robotic
* Helpful and practical
* Never overly verbose
* Speak naturally like a smart assistant

When speaking:
* Keep responses structured
* Use simple language when possible
* Avoid unnecessary filler words"""

# --------------------------
# DAEMON PRE-WARMING SYSTEM 
# --------------------------
def preload_models():
    print("DEBUG: Executing pre-warm routine for API endpoints...")
    try:
        keys = [os.getenv(k).strip() for k in ["GROQ_API_KEY", "GROQ_API_KEY_2"] if os.getenv(k)]
        if keys: requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"}, json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user", "content":"ping"}], "max_tokens": 1}, timeout=1.5)
    except: pass
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key: requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model":"mistral-large-latest","messages":[{"role":"user","content":"ping"}], "max_tokens": 1}, timeout=1.5)
    except: pass
    print("DEBUG: Pre-warming complete.")

threading.Thread(target=preload_models, daemon=True).start()





# --------------------------
# LYRA AI BACKEND ARCHITECTURE
# --------------------------

@app.route('/update_memory', methods=['POST'])
def update_memory():
    data = request.get_json()
    memory["user_context"] = data.get("user_context", "")
    save_persistent_memory(memory)
    return jsonify({"status": "ok"})

# --------------------------
# LYRA AI BACKEND ARCHITECTURE
# --------------------------

UNIVERSAL_STYLING = """
[OUTPUT PROTOCOL: MANDATORY STRUCTURE]
1. SUMMARY: A 1-sentence synthesis of the answer.
2. DETAILS: Structured analysis (use bullet points, maximum 5).
3. FINAL CONCLUSION: The definitive resolution or calculation.

Rules: 
- Zero conversational fluff (No "Sure", "I'd be happy to").
- Use Bold for key terms.
- Use Code Blocks for math/logic.
"""

def get_mode_prompt(mode, is_voice):
    base = UNIVERSAL_STYLING
    if mode == "builder": base += "\nMode: SENIOR ARCHITECT. Prioritize implementation steps and code integrity."
    elif mode == "student": base += "\nMode: LOGICAL INSTRUCTOR. Prioritize analogies and first-principles explanation."
    elif mode == "idea": base += "\nMode: CREATIVE STRATEGIST. Prioritize variety, blue-sky thinking, and feasibility."
    
    # VOICE INTELLIGENCE OVERRIDE
    if is_voice:
        base = "Respond in short, natural spoken sentences. Keep it conversational. Max 2 sentences."
    return base

def score_response(query, response):
    if not response: return 0.0
    query_words = set(query.lower().split())
    resp_words = set(response.lower().split())
    overlap = len(query_words.intersection(resp_words)) / max(1, len(query_words))
    relevance = min(1.0, overlap * 2) 
    clarity = 0.5
    if '\n' in response or '-' in response or '*' in response: clarity += 0.3
    return min(1.0, (relevance + clarity) / 2.0)

# --------------------------
# AI INTELLIGENCE ORCHESTRATOR
# --------------------------
def detect_os_intent(query):
    """Categorizes intent for smart tool routing."""
    q = query.lower()
    if any(k in q for k in ["code", "script", "function", "python", "javascript", "html", "css", "program", "debug"]): return "CODING"
    if any(k in q for k in ["plan", "schedule", "task", "calendar", "day", "todo", "appointment", "arrange"]): return "PLANNER"
    if any(k in q for k in ["stock", "price", "market", "finance", "invest", "buy", "sell", "dividend", "crypto"]): return "FINANCE"
    if any(k in q for k in ["calculate", "math", "equation", "solve", "multiply", "divide", "plus", "minus", "root", "sin", "cos"]): return "CALCULATOR"
    if any(k in q for k in ["search", "find", "who is", "latest", "news", "google", "current"]): return "SEARCH"
    return "CHITCHAT"

def fast_orchestrator(message, history, mode_prompt, branded_mode="LYT G1"):
    """Refined multi-model routing & fallback system for LYT G1."""
    # Step 1: Detect intent and perform potential Search
    intent = detect_os_intent(message)
    
    # SYSTEM STATE INJECTION
    os_context = f"\n[LYRA OS STATE]: Budget ₹{GLOBAL_OS_STATE['active_budget']} | Planned ₹{GLOBAL_OS_STATE['planned_expenses']} | Market: {GLOBAL_OS_STATE['market_sentiment']}"
    if GLOBAL_OS_STATE["current_trip"]["destination"]:
        os_context += f" | Next Trip: {GLOBAL_OS_STATE['current_trip']['destination']} ({GLOBAL_OS_STATE['current_trip']['distance_km']}km)"
    
    mode_prompt += os_context
    
    if intent == "SEARCH":
        sd = serp_search(message)
        if sd: message = f"Realtime Web Intelligence (High Priority):\n{sd}\n\nQ: {message}"
        
    intent_instruct = f"\n[INTENT DETECTED]: {intent}. Optimize output for {intent.lower()} logic."
    mode_prompt += intent_instruct
    
    # Step 2: Inject Memory Awareness
    recent_context = retrieve_relevant_context(message, limit=2)
    if recent_context:
        mode_prompt += f"\n[RELEVANT MEMORY]: {' '.join(recent_context)}"

    style = memory["config"].get("response_style", "smart")
    
    # Internal Mapping
    # LYT G1 = Smart mode (Parallel synthesis)
    # Orchestrator AI = System logic prioritized
    
    if branded_mode == "Orchestrator AI":
        mode_prompt += "\n[SYSTEM ROLE]: Executive Controller. Prioritize tool outputs and system state updates."

    # Model Map
    logical_models = [groq_call, cerebras_call, sambanova_call]
    creative_models = [mistral_call, together_call, gemini_call]

    # Step 3: Execution Strategy
    try:
        if style == "fast":
            # FAST MODE: Precision & Speed (Cerebras / Groq)
            return cerebras_call(message, history, mode_prompt)
        elif style == "creative":
            # CREATIVE MODE: Visionary & Fluid (Mistral)
            return mistral_call(message, history, mode_prompt)
        else:
            # SMART MODE (LYT G1): Parallel Synthesis
            # Uses best-of-n logic with Groq/SambaNova
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(m, message, history, mode_prompt): m.__name__ for m in logical_models[:2]}
                for future in concurrent.futures.as_completed(futures, timeout=10.0):
                    result = future.result()
                    if result and len(result) > 10: return result
    except Exception as e:
        print(f"Orchestration Error: {e}. Executing Fallback Layer...")
    
    # Universal Fallback Layer
    for model_fn in [gemini_call, together_call, openrouter_call]:
        try:
            res = model_fn(message, history, mode_prompt)
            if res: return res
        except: continue
        
    return "The Lyra intelligence node is currently experiencing architectural load. Please wait while we stabilize your stream."

def construct_payload(message, history, mode_prompt):
    base_sys = f"{SYSTEM_PROMPT}\n\n[USER SYSTEM INSTRUCTION]: {mode_prompt}"
    return [{"role": "system", "content": base_sys}] + history + [{"role": "user", "content": message}]

def groq_call(message, history=[], mode_prompt=""):
    keys = [os.getenv(k).strip() for k in ["GROQ_API_KEY", "GROQ_API_KEY_2"] if os.getenv(k)]
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {"model": "llama-3.3-70b-versatile", "messages": construct_payload(message, history, mode_prompt)}
    for key in keys:
        try:
            resp = http_session.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except: continue
    raise Exception("Fail")

def mistral_call(message, history=[], mode_prompt=""):
    api_key = os.getenv("MISTRAL_API_KEY")
    url = "https://api.mistral.ai/v1/chat/completions"
    # Pixtral Large Update
    payload = {"model": "pixtral-large-2411", "messages": construct_payload(message, history, mode_prompt)}
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def gemini_call(message, history=[], mode_prompt=""):
    api_key = os.getenv("GEMINI_API_KEY")
    # Flash Lite Model Update
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite-preview-02-05:generateContent?key={api_key}"
    contents = []
    for msg in history: contents.append({"role": "user" if msg["role"]=="user" else "model", "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    base_sys = f"{SYSTEM_PROMPT}\n\n[USER SYSTEM INSTRUCTION]: {mode_prompt}"
    payload = {"system_instruction": {"parts": [{"text": base_sys}]}, "contents": contents}
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def openrouter_call(message, history=[], mode_prompt=""):
    keys = [os.getenv(k).strip() for k in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"] if os.getenv(k)]
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # 5 Models Rotation
    models = [
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-20b:free",
        "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    ]
    
    for key in keys:
        for model in models:
            try:
                payload = {"model": model, "messages": construct_payload(message, history, mode_prompt)}
                resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=10)
                if resp.ok: return resp.json()["choices"][0]["message"]["content"]
            except: continue
    raise Exception("OpenRouter Nodes Exhausted")

def sambanova_call(message, history=[], mode_prompt=""):
    # SambaNova 6 Keys -> 6 Specific Models
    sn_configs = [
        {"key": os.getenv("SAMBANOVA_API_KEY"), "model": "Llama-4-Maverick-17B-128E-Instruct"},
        {"key": os.getenv("SAMBANOVA_API_KEY_2"), "model": "gpt-oss-120b"},
        {"key": os.getenv("SAMBANOVA_API_KEY_3"), "model": "DeepSeek-V3.1"},
        {"key": os.getenv("SAMBANOVA_API_KEY_4"), "model": "gemma-3-12b-it"},
        {"key": os.getenv("SAMBANOVA_API_KEY_5"), "model": "DeepSeek-V3.2"},
        {"key": os.getenv("SAMBANOVA_API_KEY_6"), "model": "MiniMax-M2.5"}
    ]
    url = "https://api.sambanova.ai/v1/chat/completions"
    for config in sn_configs:
        if not config["key"]: continue
        try:
            payload = {"model": config["model"], "messages": construct_payload(message, history, mode_prompt)}
            resp = requests.post(url, headers={"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"}, json=payload, timeout=12)
            if resp.ok: return resp.json()["choices"][0]["message"]["content"]
        except: continue
    raise Exception("SambaNova Pipeline Failure")

def together_call(message, history=[], mode_prompt=""):
    api_key = os.getenv("TOGETHER_API_KEY")
    url = "https://api.together.xyz/v1/chat/completions"
    # 2 Models Rotation
    models = ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "google/gemma-2-27b-it"]
    for model in models:
        try:
            payload = {"model": model, "messages": construct_payload(message, history, mode_prompt)}
            resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=10)
            if resp.ok: return resp.json()["choices"][0]["message"]["content"]
        except: continue
    raise Exception("Together Nodes Failure")

def cerebras_call(message, history=[], mode_prompt=""):
    api_key = os.getenv("CEREBRAS_API_KEY")
    url = "https://api.cerebras.ai/v1/chat/completions"
    payload = {"model": "qwen-3-235b-a22b-instruct-2507", "messages": construct_payload(message, history, mode_prompt)}
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=10)
    return resp.json()["choices"][0]["message"]["content"]

def huggingface_call(message, history=[], mode_prompt=""):
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
    payload = {"model": "mistralai/Mistral-7B-Instruct-v0.3", "messages": construct_payload(message, history, mode_prompt), "max_tokens": 1024}
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=10)
    return resp.json()["choices"][0]["message"]["content"]

# --------------------------
# SEARCH & TOOLS SYSTEM
# --------------------------
def serp_search(query):
    api_key = os.getenv("SERP_API_KEY")
    if not api_key: return ddgs_search(query) # Fallback
    
    url = "https://serpapi.com/search"
    params = { "q": query, "api_key": api_key, "num": 5 }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        results = []
        if "organic_results" in data:
            for r in data["organic_results"]: results.append(f"{r.get('title')}: {r.get('snippet')}")
        return "\n".join(results)
    except: return ddgs_search(query)

def ddgs_search(query):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5): results.append(r["body"])
        return "\n".join(results)
    except: return ""

# --------------------------
# FILE EXTRACTION SYSTEM
# --------------------------
def extract_text_from_file(file_data, file_name):
    ext = file_name.split('.')[-1].lower()
    if ext == 'txt':
        return file_data.decode('utf-8', errors='ignore')
    elif ext == 'pdf':
        try:
            import io
            from PyPDF2 import PdfReader
            pdf = PdfReader(io.BytesIO(file_data))
            text = ""
            for page in pdf.pages: text += page.extract_text()
            return text
        except: return "[System: PDF Library Missing or File Corrupt]"
    elif ext in ['jpg', 'jpeg', 'png']:
        return f"[System: Image Attachment: {file_name} - Vision Analysis Required (Base64 Context Provided)]"
    return "[System: Unsupported File Format]"

# --------------------------
# IMAGE MODALITY SYSTEM
# --------------------------
IMAGE_CACHE = {}

def is_image_request(message):
    triggers = ["generate image", "create image", "draw ", "image of", "make a picture", "generate a picture", "create a picture"]
    if message.startswith("[SYSTEM:"): return True
    return any(t in message.lower() for t in triggers)

def optimize_image_prompt(message, quality="standard"):
    sys_prompt = "You are a prompt engineer for stable diffusion. Convert this simple input into a high-quality, highly detailed image generation prompt. Add descriptors like 'cinematic lighting, 4k, ultra detailed, sharp focus, masterpiece'. Return ONLY the prompt string, no quotes."
    try: 
        p = mistral_call(message, [], sys_prompt)
        if quality == "hd": p += " -- Massive detail override, 8k resolution, photorealistic masterpiece level quality."
        return p
    except: 
        p = f"{message}, high quality, detailed, masterpiece, cinematic lighting, 4k, sharp focus"
        if quality == "hd": p += ", 8k, hyper-detailed"
        return p

import base64
import uuid
import concurrent.futures

def edit_image(img_id, prompt, instruction, quality="standard"):
    sys = f"You are a prompt editor. Rewrite this image prompt: '{prompt}' to explicitly follow this edit instruction: '{instruction}'. Keep it highly detailed. Return ONLY the new prompt string."
    try: new_prompt = mistral_call(sys, [], "You are a prompt engineer.")
    except: new_prompt = f"{prompt}, {instruction}"
    
    return generate_image(new_prompt, quality)

def generate_variations(img_id, prompt, quality="standard"):
    # SD3 doesn't easily support simple variations without strict payload params, 
    # we simulate powerful iterations via threading multiple distinct generation calls
    def gen(_): return generate_image(prompt + f", highly creative alternative variant", quality)
    res = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(gen, i) for i in range(2)]
        for f in concurrent.futures.as_completed(futures):
            rtn = f.result()
            if "image" in rtn: res.append(rtn["image"])
    
    if len(res) > 0: return {"images": res, "prompt": prompt}
    return {"error": "Couldn't generate variations."}

def generate_image(prompt, quality="standard"):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key: return {"error": "NVIDIA_API_KEY missing."}
    target_prompt = optimize_image_prompt(prompt, quality)
    # NVIDIA Screenshot Update: Stable Diffusion 3 Medium
    configs = [
        {"url": "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium", "type": "genai"},
        {"url": "https://ai.api.nvidia.com/v1/images/generations", "type": "openai", "model": "stabilityai/sd3-medium"},
        {"url": "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl", "type": "genai"}
    ]
    
    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json" }
    
    last_err = ""
    for cfg in configs:
        url = cfg["url"]
        try:
            print(f"DEBUG [Nvidia]: Trying {cfg['type']} node: {url}")
            
            if cfg["type"] == "openai":
                payload = {
                    "model": cfg["model"],
                    "prompt": target_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                }
            else:
                payload = { "prompt": target_prompt, "width": 1024, "height": 1024 }
            
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if not resp.ok: 
                last_err = f"{resp.status_code}: {resp.text}"
                continue
                
            data = resp.json()
            img_str = None
            
            if cfg["type"] == "openai":
                img_str = data["data"][0]["b64_json"]
            else:
                if "image" in data: img_str = data['image']
                elif "artifacts" in data and len(data["artifacts"]) > 0: img_str = data["artifacts"][0]["base64"]
            
            if img_str:
                img_id = str(uuid.uuid4())
                IMAGE_CACHE[img_id] = img_str
                print(f"DEBUG [Nvidia]: SUCCESS via {url}")
                return {"image": f"data:image/png;base64,{img_str}", "prompt": target_prompt, "img_id": img_id}
                
        except Exception as e:
            last_err = str(e)
            continue

    print(f"ERROR [Nvidia]: Multi-protocol failure. Final trace: {last_err}")
    
    # --------------------------
    # EMERGENCY FALLBACK: HUGGING FACE
    # --------------------------
    print("WARNING [Image]: NVIDIA Nodes exhausted. Switching to Hugging Face Fallback...")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_key:
        try:
            hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            hf_headers = {"Authorization": f"Bearer {hf_key}"}
            hf_payload = {"inputs": target_prompt}
            
            hf_resp = requests.post(hf_url, headers=hf_headers, json=hf_payload, timeout=40)
            if hf_resp.ok:
                img_data = base64.b64encode(hf_resp.content).decode('utf-8')
                img_id = str(uuid.uuid4())
                IMAGE_CACHE[img_id] = img_data
                print("DEBUG [Image]: SUCCESS via Hugging Face Fallback.")
                return {"image": f"data:image/png;base64,{img_data}", "prompt": target_prompt, "img_id": img_id}
        except Exception as e:
            print(f"ERROR [Image]: Fallback failed: {e}")

    return {"error": "All visual generation nodes (NVIDIA & HuggingFace) are currently offline. Please try again in 5 minutes."}

# --------------------------
# FLASK ROUTING
# --------------------------

@app.route('/favicon.<ext>')
def serve_favicon(ext):
    for f in ['favicon.png', 'favicon.jpg']:
        if os.path.exists(os.path.join(app.root_path, f)): return send_file(os.path.join(app.root_path, f))
    return "Not Found", 404

 @app.route('/upload_image', methods=['POST'])
def upload_image_standalone():
    if 'image' not in request.files: return jsonify({"error": "No image part"}), 400
    f = request.files['image']
    if f.filename == '': return jsonify({"error": "No filename"}), 400
    
    try:
        file_data = f.read()
        if len(file_data) > 8*1024*1024: return jsonify({"error": "File too large (>8MB)"}), 400
        
        job_id = str(uuid.uuid4())
        VISION_JOBS[job_id] = {"status": "analyzing", "filename": f.filename}
        session_id = request.headers.get('X-Session-ID', 'default')

        def run_vision_pipeline():
            # STAGE 1: Extract Structured Metadata
            metadata = analyze_image(file_data, f.filename)
            VISION_JOBS[job_id]["status"] = "synthesizing"
            
            # STAGE 2: Deep Reasoning Logic
            reasoning = reason_about_image(metadata, "Explain what this image is and its significance.", "Vision Mode")
            
            # STORE FOR FOLLOW-UPS
            if session_id not in memory: memory[session_id] = []
            if "vision_context" not in memory: memory["vision_context"] = {}
            memory["vision_context"][session_id] = metadata

            VISION_JOBS[job_id] = {
                "status": "complete", 
                "result": reasoning, 
                "metadata": metadata, 
                "filename": f.filename
            }
            ingest_file_rag(file_data, f.filename)

        threading.Thread(target=run_vision_pipeline, daemon=True).start()
        return jsonify({"job_id": job_id, "filename": f.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/vision_status/<job_id>')
def vision_status(job_id):
    job = VISION_JOBS.get(job_id)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

# --------------------------
# EVOLUTION ROUTES
# --------------------------
@app.route('/evolution/proposals')
def get_proposals():
    return jsonify({"proposals": memory.get("proposals", [])})

@app.route('/evolution/analyze', methods=['POST'])
def trigger_analysis():
    analyze_self()
    return jsonify({"status": "analyzing", "proposals": memory.get("proposals", [])})

@app.route('/evolution/approve', methods=['POST'])
def approve_proposal():
    data = request.get_json()
    action = data.get('action')
    value = data.get('value')
    index = data.get('index')
    
    if apply_change(action, value):
        proposals = memory.get("proposals", [])
        if 0 <= index < len(proposals):
            proposals.pop(index)
            memory["proposals"] = proposals
            save_persistent_memory(memory)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

# --------------------------
# UTILITY ORCHESTRATOR
# --------------------------
def run_utility_orchestrator(message):
    u_type = "GENERAL"
    if "[UTIL: CALCULATOR]" in message: u_type = "CALCULATOR"
    elif "[UTIL: MAPS]" in message: u_type = "MAPS"
    elif "[UTIL: FINANCE]" in message: u_type = "FINANCE"
    elif "[UTIL: PLANNER]" in message: u_type = "PLANNER"
    
    clean_msg = message.replace(f"[UTIL: {u_type}]", "").strip()
    
    if u_type == "CALCULATOR":
        # Simple Math Bypass (Zero AI Latency)
        if re.match(r'^[0-9+\-*/().% ]+$', clean_msg):
            res = safe_eval(clean_msg)
            if res is not None:
                return f"**Answer**: {res}\n\n**Mechanism**: Native Precision Solver"
        
        # Complex/Word Problem Logic
        return ai_math_solvent(clean_msg)
    elif u_type == "MAPS":
        sd = serp_search(f"location data and maps for: {clean_msg}")
        sys = "You are a Geospatial Intelligence Engine. Describe locations, travel times, and map data based on the provided search results."
        return fast_orchestrator(f"SEARCH DATA: {sd}\n\nUSER: {clean_msg}", [], sys)
    elif u_type == "FINANCE":
        sd = serp_search(f"stock price and market news: {clean_msg}")
        sys = "You are a Financial Quantitative Analyst. Provide market insights, stock prices, and financial analysis."
        return fast_orchestrator(f"MARKET DATA: {sd}\n\nUSER: {clean_msg}", [], sys)
    elif u_type == "PLANNER":
        return run_agent_swarm(clean_msg)
    return "[System: Utility Misfire]"

# --------------------------
# UNIFIED INTELLIGENCE SYSTEM
# --------------------------

def get_unified_os_context():
    """Synthesizes data from across all Lyra apps into a unified cognitive bank."""
    try:
        # 1. Planner Logic
        planner_data = load_planner_data()
        all_tasks = planner_data.get('tasks', []) if isinstance(planner_data, dict) else []
        today_tasks = [t['task'] for t in all_tasks if isinstance(t, dict) and t.get('status') == 'pending']
        
        # 2. Finance Logic
        finance_data = load_finance_data()
        watchlist = finance_data.get('watchlist', [])
        invested = finance_data.get('portfolio', {}).get('invested', 0)
        
        # 3. User Context
        user_context = memory.get("user_context", "No specific habits stored.")

        os_context = f"""
        [LYRA OS CURRENT INTELLIGENCE STATE]
        
        - ACTIVE TASKS: {', '.join(today_tasks[:5]) if today_tasks else 'No pending tasks.'}
        - MARKET WATCHLIST: {', '.join(watchlist[:5]) if watchlist else 'No active stocks.'}
        - PORTFOLIO CAPITAL: ${invested}
        - KNOWN USER HABITS: {user_context}
        
        [SYSTEM CAPABILITIES]
        - You can modify the Planner if user asks to schedule something.
        - You can perform Financial analysis for any stock in the watchlist.
        - You can use the Calculator engine for complex math.
        """
        return os_context
    except Exception as e:
        print(f"OS Context Sync Error: {e}")
        return "[SYSTEM: Cross-App Sync Interrupted]"

def detect_os_intent(query):
    """Routes queries to specific OS tools if intent is detected."""
    query = query.lower()
    if any(w in query for w in ["plan", "schedule", "task", "calendar", "day", "todo"]):
        return "PLANNER"
    if any(w in query for w in ["stock", "price", "market", "finance", "invest", "buy", "sell"]):
        return "FINANCE"
    if any(w in query for w in ["calculate", "math", "equation", "solve", "plus", "minus"]):
        return "CALCULATOR"
    return "GENERAL"
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if f.filename == '': return jsonify({"error": "No selected file"}), 400
    
    # Check max images/files if needed
    file_data = f.read()
    
    # INGEST INTO MULTIMODAL RAG PIPELINE
    threading.Thread(target=ingest_file_rag, args=(file_data, f.filename), daemon=True).start()
    return jsonify({"filename": f.filename, "status": "indexed", "info": f"{f.filename} is now part of my semantic knowledge base."})

@app.route('/restore_session', methods=['POST'])
def restore_session():
    data = request.get_json()
    sid = data.get('session_id')
    history = data.get('history', [])
    if sid:
        memory[sid] = history
        return jsonify({"status": "restablished"})
    return jsonify({"status": "failed"}), 400

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    raw_message = data.get('message', '')
    provider = data.get('provider', 'smart').lower()
    mode = data.get('mode', 'builder').lower()
    is_voice = data.get('is_voice', False)
    image_quality = data.get('image_quality', 'standard')
    session_id = data.get('session_id')
    
    if not session_id or session_id not in memory:
        session_id = str(uuid.uuid4())
        memory[session_id] = []
        
    history = memory.get(session_id, [])[-10:]
    
    # RAG & PERSISTENT MEMORY INJECTION
    user_prefs = memory.get("user_context", "")
    semantic_history = retrieve_relevant_context(raw_message, "semantic_bank", limit=2)
    document_context = retrieve_relevant_context(raw_message, "document_store", limit=4, threshold=0.7)
    
    hist_str = "\n".join([f"- {m}" for m in semantic_history])
    doc_str = "\n".join([f"- {m}" for m in document_context])
    
    # UNIFIED OS CONTEXT INJECTION
    os_intelligence = get_unified_os_context()
    
    rag_feedback = ""
    if document_context:
        is_visual = any("[IMAGE SOURCE]" in m for m in document_context)
        rag_feedback = "[SOURCE: INFRASTRUCTURE DOCUMENTS DETECTED]"
        if is_visual: rag_feedback = "[SOURCE: VISUAL DATA & DOCUMENTS DETECTED. Mention 'Based on the provided images' if applicable.]"

    mode_prompt = f"{get_mode_prompt(mode, is_voice)}\n{rag_feedback}\n\n{os_intelligence}\n\n[USER PERSONALITY]: {user_prefs}\n\n[PAST CONVERSATION CONTEXT]:\n{hist_str}\n\n[DOCUMENTS & IMAGES CONTEXT]:\n{doc_str}"
    
    # MULTIMODAL VISION CONTEXT INJECTION (Multi-turn Support)
    vision_context = memory.get("vision_context", {}).get(session_id, "")
    if vision_context and any(w in raw_message.lower() for w in ["it", "this image", "image", "what is", "happening", "describe", "explain"]):
        final_text = reason_about_image(vision_context, raw_message, mode_prompt)
        provider_name = "Lyra Vision Reasoning"
    elif is_image_request(raw_message):
        
        # Parse Specific Action Protocols
        if raw_message.startswith("[SYSTEM: VARIATION]"):
            parts = raw_message.split(" | PRMPT:")
            img_id = parts[0].replace("[SYSTEM: VARIATION] ID:", "").strip()
            prompt = parts[1].strip() if len(parts) > 1 else ""
            img_res = generate_variations(img_id, prompt, image_quality)
            if "error" in img_res: return jsonify({"type": "error", "data": img_res["error"]})
            memory[session_id].append({"role": "user", "content": "Create variations of the image."})
            memory[session_id].append({"role": "assistant", "content": f"[GENERATED IMAGE GRID]"})
            return jsonify({"type": "image_grid", "data": img_res["images"], "prompt": img_res["prompt"]})
            
        elif raw_message.startswith("[SYSTEM: EDIT]"):
            parts = raw_message.split(" | PRMPT:")
            p2 = parts[1].split(" | INST:") if len(parts) > 1 else ["",""]
            img_id = parts[0].replace("[SYSTEM: EDIT] ID:", "").strip()
            prompt = p2[0].strip()
            instruction = p2[1].strip() if len(p2) > 1 else ""
            img_res = edit_image(img_id, prompt, instruction, image_quality)
            
        elif raw_message.startswith("[SYSTEM: STYLE]"):
            parts = raw_message.split(" | PRMPT:")
            p2 = parts[1].split(" | STY:") if len(parts) > 1 else ["",""]
            img_id = parts[0].replace("[SYSTEM: STYLE] ID:", "").strip()
            prompt = p2[0].strip()
            style = p2[1].strip() if len(p2) > 1 else ""
            img_res = generate_image(prompt + f", painted in a strict highly detailed {style} style art direction", image_quality)
            
        else:
            img_res = generate_image(raw_message, image_quality)
            
        if "error" in img_res: 
            return jsonify({"type": "error", "data": img_res["error"]})
        
        memory[session_id].append({"role": "user", "content": raw_message})
        memory[session_id].append({"role": "assistant", "content": f"[GENERATED IMAGE: {img_res['prompt']}]"})
        return jsonify({"type": "image", "data": img_res["image"], "prompt": img_res["prompt"], "img_id": img_res["img_id"]})

    cache_key = f"{provider}_{mode}_{is_voice}_{raw_message}"
    if cache_key in query_cache:
        def stream_cached():
            for word in query_cache[cache_key].split(" "):
                yield word + " "
                time.sleep(0.01) 
        return Response(stream_with_context(stream_cached()), mimetype='text/plain')

    try:
        wf_instruction = get_workflow_instruction(raw_message)
        if wf_instruction: mode_prompt = f"{mode_prompt}\n\n[WORKFLOW OVERRIDE]: {wf_instruction}"

        if "[UTIL:" in raw_message:
            final_text = run_utility_orchestrator(raw_message)
        elif provider == "smart":
            # LYT G1 Core
            final_text = fast_orchestrator(raw_message, history, mode_prompt, branded_mode="LYT G1")
        elif provider == "agent":
            # Agentic Autopilot
            final_text = run_agent_swarm(raw_message)
        elif provider == "orchestrator":
            # Orchestrator AI Control
            final_text = fast_orchestrator(raw_message, history, mode_prompt, branded_mode="Orchestrator AI")
        else:
            # Silent Fallback
            final_text = fast_orchestrator(raw_message, history, mode_prompt)

        # CACHE FINAL RESULT
        query_cache[cache_key] = final_text
        if len(query_cache) > 200: query_cache.pop(next(iter(query_cache))) 
        
        # PERSIST SEMANTICALLY
        add_to_semantic_memory(f"Query: {raw_message} | Logic: {final_text[:200]}")
        memory["chat_count"] = memory.get("chat_count", 0) + 1
        if memory["chat_count"] % 15 == 0:
            threading.Thread(target=analyze_self, daemon=True).start()
            
        if session_id not in memory: memory[session_id] = []
        memory[session_id].append({"role": "user", "content": raw_message})
        memory[session_id].append({"role": "assistant", "content": final_text})
        save_persistent_memory(memory)
        
        format_text = final_text

        def generate():
            # STREAMING ENGINE: Improved logic for "sentient" flow
            words = format_text.split(" ")
            for i in range(0, len(words)):
                chunk = words[i] + " "
                yield chunk
                if len(chunk) > 10: time.sleep(0.015)
                else: time.sleep(0.008)
                
        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        print(f"API Error Caught: {e}")
        def generate_err():
            err_msg = "⚠ Something failed structurally. Generating fallback metrics."
            for chunk in err_msg.split(" "):
                yield chunk + " "
                time.sleep(0.01)
        return Response(stream_with_context(generate_err()), mimetype='text/plain')

@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    msg_id = data.get('id')
    vote = data.get('vote') # 1 for up, -1 for down
    add_to_semantic_memory(f"[SYSTEM FEEDBACK]: User gave {vote} to message {msg_id}. Adjust future reasoning patterns accordingly.")
    return jsonify({"status": "captured"})

def get_workflow_instruction(message):
    if "[WORKFLOW: IDEA]" in message:
        return "Act as an Expert Visionary. Generate 3 distinct concept directions. For each, provide: 1. Core Value 2. Market Strategy 3. Key Obstacle."
    if "[WORKFLOW: CONTENT]" in message:
        return "Act as a Pro Content Architect. Create a high-converting draft. Include relevant SEO keywords, a hook, and a clear call to action."
    if "[WORKFLOW: BUILDER]" in message:
        return "Act as a Senior Software Engineer. Provide clean, modular code with comments. Explain your logic step-by-step and mention potential edge cases."
    if "[ITERATE: IMPROVE]" in message:
        return "Take the previous output and improve the professional quality, clarity, and depth. Make it elite."
    if "[ITERATE: SIMPLIFY]" in message:
        return "Make the previous output shorter and easier to understand for a 5th grader. Remove jargon."
    if "[ITERATE: MORE CREATIVE]" in message:
        return "Inject more personality, unique analogies, and creative flair into the previous response."
    return ""

PLANNER_DATA_FILE = "planner.json"

def load_planner_data():
    if not os.path.exists(PLANNER_DATA_FILE): return {"tasks": [], "history": []}
    try:
        with open(PLANNER_DATA_FILE, "r") as f: return json.load(f)
    except: return {"tasks": [], "history": []}

def save_planner_data(data):
    with open(PLANNER_DATA_FILE, "w") as f: json.dump(data, f, indent=4)

@app.route('/planner/data')
def get_planner_data():
    return jsonify(load_planner_data())

CHATS_FILE = "chats.json"

def load_all_chats():
    if not os.path.exists(CHATS_FILE): return []
    try:
        with open(CHATS_FILE, "r") as f: return json.load(f)
    except: return []

def save_all_chats(chats):
    with open(CHATS_FILE, "w") as f: json.dump(chats, f, indent=4)

@app.route('/chats', methods=['GET'])
def get_chats():
    return jsonify(load_all_chats())

@app.route('/chats/create', methods=['POST'])
def create_chat():
    data = request.get_json()
    new_chat = {
        "id": data.get("id", str(uuid.uuid4())),
        "title": "New Chat",
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time(),
        "is_archived": False
    }
    chats = load_all_chats()
    chats.append(new_chat)
    save_all_chats(chats)
    return jsonify(new_chat)

@app.route('/chats/update', methods=['POST'])
def update_chat():
    data = request.get_json()
    chat_id = data.get("id")
    title = data.get("title")
    messages = data.get("messages")
    is_archived = data.get("is_archived")
    
    chats = load_all_chats()
    for c in chats:
        if c["id"] == chat_id:
            if title is not None: c["title"] = title
            if messages is not None: c["messages"] = messages
            if is_archived is not None: c["is_archived"] = is_archived
            c["updated_at"] = time.time()
            break
    save_all_chats(chats)
    return jsonify({"status": "ok"})

@app.route('/chats/delete', methods=['POST'])
def delete_chat():
    data = request.get_json()
    chat_id = data.get("id")
    chats = [c for c in load_all_chats() if c["id"] != chat_id]
    save_all_chats(chats)
    return jsonify({"status": "ok"})

@app.route('/generate_title', methods=['POST'])
def generate_title():
    data = request.get_json()
    first_msg = data.get("message", "")
    prompt = f"Generate a short, semantic 3-5 word title for a conversation starting with: '{first_msg}'. Return ONLY the title string, no quotes."
    try:
        title = mistral_call(prompt, [], "Lyra Naming Engine")
        return jsonify({"title": title.strip()})
    except:
        return jsonify({"title": "New Conversation"})

@app.route('/planner/save', methods=['POST'])
def save_planner():
    save_planner_data(request.get_json())
    return jsonify({"status": "saved"})


@app.route('/planner/hotels')
def discover_hotels():
    city = request.args.get('city', 'Goa')
    # Synthetic Discovery Engine (Mocking high-fidelity API response)
    hotels = [
        {"name": f"The {city} Grand", "price": "₹4,500/night", "rating": "4.8★", "img": "fa-hotel"},
        {"name": f"Azure {city} Resort", "price": "₹7,200/night", "rating": "4.9★", "img": "fa-water"},
        {"name": "Budget Oasis", "price": "₹1,200/night", "rating": "4.2★", "img": "fa-bed"}
    ]
    return jsonify(hotels)

@app.route('/planner/events')
def discover_events():
    city = request.args.get('city', 'Goa')
    events = [
        {"title": f"{city} Music Fest", "time": "8 PM Tonight", "type": "Concert", "cost": "₹999"},
        {"title": "Beach Yoga Summit", "time": "6 AM Tomorrow", "type": "Wellness", "cost": "Free"},
        {"title": "Night Life Crawl", "time": "10 PM", "type": "Clubbing", "cost": "₹1,500"}
    ]
    return jsonify(events)

@app.route('/planner/generate', methods=['POST'])
def generate_planner():
    data = request.get_json()
    goal = data.get('goal', 'Plan my day')
    user_context = memory.get("user_context", "Prefers nocturnal studying and has gym sessions.")
    
    prompt = f"""
    Act as an elite productivity architect. 
    User Goal: {goal}
    Memory Context: {user_context}
    
    Break this into a structured daily schedule in JSON format.
    Output MUST be a JSON list of objects: [ {{"time": "8 AM", "task": "Wake up", "category": "health"}}, ... ]
    Only return raw JSON. No markdown.
    """
    
    try:
        raw_json = groq_call(prompt, [], "PLANNER_ORCHESTRATOR")
        # Clean potential markdown
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        schedule = json.loads(clean_json)
        return jsonify({"schedule": schedule})
    except Exception as e:
        print(f"Planner Gen Error: {e}")
        return jsonify({"error": "Failed to architect schedule", "raw": str(e)}), 500

FINANCE_DATA_FILE = "finance.json"

def load_finance_data():
    if not os.path.exists(FINANCE_DATA_FILE): return {"watchlist": ["RELIANCE", "TCS", "AAPL", "NVDA"], "portfolio": {"invested": 100000, "units": {}}}
    try:
        with open(FINANCE_DATA_FILE, "r") as f: return json.load(f)
    except: return {"watchlist": ["RELIANCE", "TCS", "AAPL", "NVDA"], "portfolio": {"invested": 100000, "units": {}}}

def save_finance_data(data):
    with open(FINANCE_DATA_FILE, "w") as f: json.dump(data, f, indent=4)

@app.route('/finance/data')
def get_finance_data():
    return jsonify(load_finance_data())

@app.route('/finance/save', methods=['POST'])
def save_finance():
    save_finance_data(request.get_json())
    return jsonify({"status": "saved"})

@app.route('/stock/<symbol>')
def get_stock_data(symbol):
    symbol = symbol.upper()
    import random
    # Highly sophisticated mock data generator
    # We simulate a 7-day price history with realistic volatility
    base_prices = {"RELIANCE": 2950, "TCS": 3800, "AAPL": 185, "NVDA": 850, "ZOMATO": 180, "GOOGL": 140}
    base = base_prices.get(symbol, random.randint(100, 5000))
    
    history = []
    curr = base
    for _ in range(7):
        curr = curr * (1 + random.uniform(-0.02, 0.025))
        history.append(round(curr, 2))
    
    change = round(((history[-1] - history[-2]) / history[-2]) * 100, 2)
    
    return jsonify({
        "symbol": symbol,
        "price": history[-1],
        "change": change,
        "history": history,
        "name": symbol + " INC."
    })

@app.route('/finance/ai', methods=['POST'])
def finance_ai_analysis():
    data = request.get_json()
    symbol = data.get('stock')
    price = data.get('price')
    trend = data.get('trend')
    
    prompt = f"Analyze {symbol} at {price}. Trend: {trend}. Keep it simple, educational, and professional."
    res = fast_orchestrator(prompt, [], "Lyra Market Analyst")
    return jsonify({"analysis": res})

# --------------------------
# LYRA ECOSYSTEM SYNC LAYER
# --------------------------

@app.route('/system/state')
def get_system_state():
    return jsonify(GLOBAL_OS_STATE)

@app.route('/system/sync', methods=['POST'])
def update_system_state():
    updates = request.get_json()
    new_state = sync_system_state(updates)
    return jsonify(new_state)

# --------------------------
# GEOSPATIAL INTELLIGENCE ENGINE
# --------------------------
MAJOR_CITIES = {
    "GOA": (15.2993, 74.1240),
    "MUMBAI": (19.0760, 72.8777),
    "BANGALORE": (12.9716, 77.5946),
    "CHENNAI": (13.0827, 80.2707),
    "DELHI": (28.6139, 77.2090),
    "PONDICHERRY": (11.9416, 79.8083),
    "HYDERABAD": (17.3850, 78.4867)
}

def calculate_geo_distance(origin, target):
    import math
    o = origin.upper()
    t = target.upper()
    if o in MAJOR_CITIES and t in MAJOR_CITIES:
        lat1, lon1 = MAJOR_CITIES[o]
        lat2, lon2 = MAJOR_CITIES[t]
        R = 6371 # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return round(R * c)
    return 0

@app.route('/planner/trip', methods=['POST'])
def plan_trip():
    data = request.get_json()
    goal = data.get('goal', 'Plan my trip')
    
    # Intelligence Extraction: Target City
    target_city = "GOA"
    for city in MAJOR_CITIES.keys():
        if city.lower() in goal.lower():
            target_city = city
            break
            
    distance = calculate_geo_distance("CHENNAI", target_city)
    budget = GLOBAL_OS_STATE["active_budget"]
    
    sys = f"""
    Act as Lyra's Travel Architect. 
    Constraint: Total budget is ₹{budget}.
    Origin: CHENNAI | Target: {target_city} | Est. Distance: {distance}km.
    Rules:
    1. Provide ESTIMATED price ranges (e.g. ₹3,000 - ₹3,500).
    2. Format as [ITINERARY] followed by a JSON list.
    3. Include real-world distance steps.
    Itinerary JSON Schema: [ {{"day": 1, "steps": [ {{"time": "10 AM", "activity": "...", "cost": 100, "distance": 10}} ] }} ]
    """
    
    res = fast_orchestrator(goal, [], sys)
    
    if "[ITINERARY]" in res:
        try:
            itinerary_json = res.split("[ITINERARY]")[1].strip()
            # DEFENSE: Strip markdown code blocks
            itinerary_json = itinerary_json.replace("```json", "").replace("```", "").strip()
            trip_data = json.loads(itinerary_json)
            
            # Unified Cost Engine
            total_trip_cost = 0
            total_distance = 0
            for day in trip_data:
                for step in day.get("steps", []):
                    # Robust Cost Cleaning
                    cost_val = str(step.get("cost", "0")).replace("₹", "").replace(",", "").split("-")[0].strip()
                    try: total_trip_cost += int(float(cost_val))
                    except: pass
                    
                    dist_val = str(step.get("distance", "0")).replace("km", "").strip()
                    try: total_distance += int(float(dist_val))
                    except: pass
            
            # System Synchronization
            sync_system_state({
                "planned_expenses": total_trip_cost,
                "current_trip": {
                    "destination": goal,
                    "distance_km": total_distance,
                    "total_estimated_cost": total_trip_cost,
                    "itinerary": trip_data
                }
            })
            
            return jsonify({
                "status": "success", 
                "data": trip_data, 
                "summary": {
                    "cost": total_trip_cost,
                    "distance": total_distance,
                    "budget_safe": total_trip_cost <= budget
                }
            })
        except Exception as e:
            print(f"Sync Error: {e}")
            pass
            
    return jsonify({"status": "fallback", "text": res})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
