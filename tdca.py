import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    pipeline,
    AutoModelForCausalLM
)
import os
import re
import concurrent.futures
from functools import lru_cache
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import deque
import mmap
import psutil
import itertools
from tqdm import tqdm
import gc
import json
from datetime import datetime

class CodeCommenterGUI(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.grid(sticky='nsew')
        
        # Initialize variables before UI
        self.use_gpu = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=False)
        self.is_analyzing = False
        self.is_paused = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup UI first
        self.setup_ui()
        
        # Initialize logging
        self.log_message("Initializing...")
        
        # Load models in background to prevent UI freeze
        self.root.after(100, self.setup_models)

    def setup_models(self):
        """Initialize DeepSeek model for advanced code analysis"""
        try:
            self.log_message("Loading DeepSeek model (this may take a few minutes for first-time setup)...")
            
            # Import required components
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Initialize prompts
            self.prompts = {
                'analyze_function': """Analyze this code for key generation or cryptographic operations:
{code}
Identify:
1. Key generation routines
2. Cryptographic operations
3. Suspicious bit manipulations
4. Encryption/decryption logic
5. Hash functions
6. Key schedules or S-boxes

Provide a detailed analysis.""",
                
                'generate_comment': """Generate a technical comment for this code line:
{code}
Focus on:
1. Purpose of the code
2. Any security implications
3. Cryptographic relevance
4. Key operations

Comment:"""
            }
            
            # Use smaller model for faster loading
            model_path = "deepseek-ai/deepseek-coder-1.3b-base"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir="./models",
                local_files_only=False
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir="./models"
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.log_message("Using GPU acceleration")
            else:
                self.log_message("Running on CPU - this may be slower")
            
            self.log_message("DeepSeek model loaded successfully")
            self.analyze_btn.config(state='normal')
            
        except Exception as e:
            self.log_message(f"Error loading DeepSeek model: {str(e)}", error=True)
            self.analyze_btn.config(state='disabled')

    def fine_tune_models(self):
        """Minimal fine-tuning to initialize weights properly"""
        # Prepare sample data
        sample_code = [
            "int x = 0;",
            "void main() {",
            "return 0;"
        ]
        sample_comments = [
            "Initialize variable",
            "Main function entry point",
            "Return success status"
        ]
        
        # Fine-tune CodeBERT
        with torch.no_grad():
            inputs = self.code_tokenizer(
                sample_code,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            self.code_model(**inputs)
        
        # Fine-tune GPT-2
        with torch.no_grad():
            inputs = self.comment_tokenizer(
                sample_comments,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            self.comment_model(**inputs)

    def setup_ui(self):
        """Setup the UI with consistent grid geometry management"""
        # Top frame for file selection
        file_frame = ttk.Frame(self)
        file_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        ttk.Label(file_frame, text="Source File:").grid(row=0, column=0, padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self, text="Settings")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        ttk.Checkbutton(settings_frame, text="Use GPU (if available)", variable=self.use_gpu).grid(row=0, column=0, padx=5, pady=2)
        ttk.Checkbutton(settings_frame, text="Show confidence scores", variable=self.show_confidence).grid(row=0, column=1, padx=5, pady=2)
        
        
        # Text areas with search
        text_frame = ttk.Frame(self)
        text_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        # Original code section
        original_frame = ttk.LabelFrame(text_frame, text="Original Code")
        original_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        
        # Search for original code
        original_search_frame = ttk.Frame(original_frame)
        original_search_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        
        self.original_search_var = tk.StringVar()
        self.original_search_entry = ttk.Entry(original_search_frame, textvariable=self.original_search_var)
        self.original_search_entry.grid(row=0, column=0, sticky='ew', padx=2)
        
        ttk.Button(original_search_frame, text="Find", 
                   command=lambda: self.find_text(self.original_text, self.original_search_var.get())).grid(row=0, column=1, padx=2)
        ttk.Button(original_search_frame, text="Next",
                   command=lambda: self.find_text(self.original_text, self.original_search_var.get(), next=True)).grid(row=0, column=2, padx=2)
        original_search_frame.grid_columnconfigure(0, weight=1)
        
        # Original text area
        self.original_text = scrolledtext.ScrolledText(original_frame, height=20, width=60)
        self.original_text.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Commented code section
        commented_frame = ttk.LabelFrame(text_frame, text="Commented Code")
        commented_frame.grid(row=0, column=1, sticky='nsew', padx=5)
        
        # Search for commented code
        commented_search_frame = ttk.Frame(commented_frame)
        commented_search_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        
        self.commented_search_var = tk.StringVar()
        self.commented_search_entry = ttk.Entry(commented_search_frame, textvariable=self.commented_search_var)
        self.commented_search_entry.grid(row=0, column=0, sticky='ew', padx=2)
        
        ttk.Button(commented_search_frame, text="Find",
                   command=lambda: self.find_text(self.commented_text, self.commented_search_var.get())).grid(row=0, column=1, padx=2)
        ttk.Button(commented_search_frame, text="Next",
                   command=lambda: self.find_text(self.commented_text, self.commented_search_var.get(), next=True)).grid(row=0, column=2, padx=2)
        commented_search_frame.grid_columnconfigure(0, weight=1)
        
        # Commented text area
        self.commented_text = scrolledtext.ScrolledText(commented_frame, height=20, width=60)
        self.commented_text.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Configure text frame weights
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_columnconfigure(1, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        
        # Buttons frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Code", command=self.analyze_code)
        self.analyze_btn.grid(row=0, column=0, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Output", command=self.save_output)
        self.save_btn.grid(row=0, column=1, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_all)
        self.clear_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_analysis, state='disabled')
        self.stop_btn.grid(row=0, column=3, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_btn.grid(row=0, column=4, padx=5)
        
        # Initialize Refine button in enabled state
        self.refine_btn = ttk.Button(button_frame, text="Refine Findings", 
                                    command=self.show_refined_findings,
                                    state='normal')
        self.refine_btn.grid(row=0, column=5, padx=5)
        
        # Progress bar and status
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky='w', padx=5)
        
        # Add log text area
        log_frame = ttk.LabelFrame(self, text="Processing Log")
        log_frame.grid(row=7, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=3)  # Text areas get more space
        self.grid_rowconfigure(7, weight=1)  # Log area gets some space
        
        # Initialize the UI before loading models
        self.root.update_idletasks()

        # Bind Ctrl+F to focus appropriate search box
        self.original_text.bind('<Control-f>', lambda e: self.focus_search(self.original_search_entry))
        self.commented_text.bind('<Control-f>', lambda e: self.focus_search(self.commented_search_entry))
        
        # Bind Return in search boxes
        self.original_search_entry.bind('<Return>', 
                                      lambda e: self.find_text(self.original_text, self.original_search_var.get()))
        self.commented_search_entry.bind('<Return>', 
                                       lambda e: self.find_text(self.commented_text, self.commented_search_var.get()))

    def apply_settings(self):
        """Apply hardware configuration settings"""
        try:
            # Update thread count
            self.max_workers = int(cpu_threads.get())
            
            # Update batch size
            self.batch_size = int(batch_size.get())
            
            # Update memory settings
            memory_settings = {
                "Low": {
                    "chunk_size": 5000,
                    "pattern_queue": 1000,
                    "cache_size": 500
                },
                "Medium": {
                    "chunk_size": 10000,
                    "pattern_queue": 5000,
                    "cache_size": 1000
                },
                "High": {
                    "chunk_size": 20000,
                    "pattern_queue": 10000,
                    "cache_size": 2000
                }
            }
            
            settings = memory_settings[memory_usage.get()]
            self.chunk_size = settings["chunk_size"]
            self.pattern_queue = deque(maxlen=settings["pattern_queue"])
            self.process_single_line = lru_cache(maxsize=settings["cache_size"])(self.process_single_line.__wrapped__)
            
            self.log_message("Settings applied successfully!")
            
        except Exception as e:
            self.log_message(f"Error applying settings: {str(e)}", error=True)

    def log_message(self, message, error=False, warning=False, update_last=False):
        """Log a message to the log text area"""
        if not hasattr(self, 'log_text') or self.log_text is None:
            print(f"Log: {message}")  # Fallback to console if log_text not available
            return
        
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        if error:
            log_entry = f"{timestamp} ERROR: {message}\n"
            tag = 'error'
        elif warning:
            log_entry = f"{timestamp} WARNING: {message}\n"
            tag = 'warning'
        else:
            log_entry = f"{timestamp} INFO: {message}\n"
            tag = 'info'
        
        if update_last:
            # Remove last line if updating
            last_line = self.log_text.get("end-2c linestart", "end-1c")
            if last_line.startswith(timestamp):
                self.log_text.delete("end-2c linestart", "end-1c")
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.tag_add(tag, f"end-{len(log_entry)+1}c", "end-1c")
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('info', foreground='blue')
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def analyze_security_patterns(self, line):
        security_suggestions = []
        
        # Check for key generation patterns - look for more specific keywords
        if re.search(r'(?i)(generate|create|init).*?(key|secret|token|iv|nonce|password|salt)', line):
            security_suggestions.append(
                "WARNING: Key generation detected. Recommendations:\n"
                "- Use cryptographically secure random number generator\n"
                "- Ensure sufficient key length (AES-256, RSA-2048+)\n"
                "- Consider using established key derivation functions (PBKDF2, Argon2)"
            )
        
        # Check for random number usage - more specific pattern
        if re.search(r'(?i)(random\(|rand\(|srand\(|RandomBytes|RNG)', line):
            security_suggestions.append(
                "WARNING: Random number usage detected. Recommendations:\n"
                "- Use CSPRNG instead of rand()\n"
                "- Consider using /dev/urandom or CryptGenRandom()"
            )
        
        # Check for crypto operations - more specific pattern
        if re.search(r'(?i)(encrypt\(|decrypt\(|sign\(|verify\(|hash\(|cipher|AES|RSA|SHA)', line):
            security_suggestions.append(
                "WARNING: Cryptographic operation detected. Verify:\n"
                "- Using current algorithms (AES, RSA, ECC)\n"
                "- Proper key management\n"
                "- Secure mode of operation (CBC, GCM)"
            )
        
        return security_suggestions
        
    @lru_cache(maxsize=2000)  # Increased cache size
    def generate_pattern_based_comment(self, line):
        # Add specific patterns for type definitions
        if 'typedef' in line:
            # Basic types
            if 'unsigned char' in line:
                return "8-bit unsigned integer type"
            if 'unsigned int' in line:
                return "32-bit unsigned integer type"
            if 'unsigned long long' in line:
                return "64-bit unsigned integer type"
            if 'unsigned short' in line:
                return "16-bit unsigned integer type"
            
            # Special types
            if 'GUID' in line:
                return "Globally Unique Identifier type (128-bit)"
            if 'ImageBaseOffset32' in line:
                return "32-bit offset relative to image base"
            if 'wchar' in line:
                return "Wide character type for Unicode support"
            
            # Structures
            if 'struct' in line:
                struct_name = line.split()[-1].strip(';')
                return f"Structure type definition for {struct_name}"
            
            # Unions
            if 'union' in line:
                union_name = line.split()[-1].strip(';')
                return f"Union type definition for {union_name}"
        
        # Check security patterns first
        security_suggestions = self.analyze_security_patterns(line)
        if security_suggestions:
            return "\n".join(security_suggestions)

        # Handle includes
        include_match = re.search(self.code_patterns['include'], line)
        if include_match:
            library = include_match.group(1)
            if 'iostream' in library:
                return "Include for input and output functions"
            elif 'string' in library:
                return "Include for string handling functions"
            elif 'sstream' in library:
                return "Include for string stream functionalities"
            elif 'algorithm' in library:
                return "Include for algorithm functions"
            elif 'Windows.h' in library:
                return "Include for Windows-specific functionalities"
            return f"Include for {library} functionality"
        
        # Handle variable declarations
        var_match = re.search(self.code_patterns['variable_declaration'], line)
        if var_match:
            type_name, var_name, initial_value = var_match.groups()
            if initial_value.strip() in ['""', "0", '{}']:
                return f"Variable {var_name} of type {type_name} initialized empty"
            return f"Variable {var_name} of type {type_name} initialized to {initial_value}"
        
        # Handle for loops
        for_match = re.search(r'for\s*\((.*?)\)', line)
        if for_match:
            condition = for_match.group(1)
            return f"Loop through {condition}"
        
        # Handle if statements
        if_match = re.search(self.code_patterns['if_statement'], line)
        if if_match:
            condition = if_match.group(1)
            if 'ASCII' in condition or ord('A') <= ord(condition[0]) <= ord('Z'):
                return "Check if character is uppercase letter"
            return f"Check condition: {condition}"
        
        # Handle return statements
        return_match = re.search(self.code_patterns['return_statement'], line)
        if return_match:
            value = return_match.group(1)
            if value == '0':
                return "Return 0 to indicate successful execution"
            return f"Return {value}"
        
        # Handle system calls
        system_match = re.search(self.code_patterns['system_call'], line)
        if system_match:
            command = system_match.group(1)
            if 'pause' in command:
                return "Pause the console window so it doesn't close immediately"
            return f"Execute system command: {command}"
        
        return None
    
    def generate_ai_comment(self, line):
        try:
            # Prepare input with proper attention mask
            inputs = self.comment_tokenizer(
                f"Explain this code line briefly: {line}\nComment:",
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            outputs = self.comment_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=self.comment_tokenizer.eos_token_id,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            comment = self.comment_tokenizer.decode(outputs[0], skip_special_tokens=True)
            comment = comment.split("Comment:")[-1].strip()
            
            # Clean up and format
            comment = comment.replace("\n", " ").strip()
            comment = re.sub(r'\s+', ' ', comment)
            comment = comment[:100]
            
            return comment
        except Exception as e:
            print(f"Error generating AI comment: {str(e)}")
            return None

    def browse_file(self):
        """Browse for a file to analyze"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt"),
                ("C files", "*.c"),
                ("Header files", "*.h"),
                ("Assembly files", "*.asm"),
                ("Decompiled files", "*.dec")
            ],
            title="Select File to Analyze"
        )
        if file_path:
            self.file_path_var.set(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.original_text.delete(1.0, tk.END)
                    self.original_text.insert(1.0, content)
                    
                    # Just log if it's a findings file, but don't change button state
                    if "SECURITY/CRYPTO FUNCTION DETECTED" in content or "KEY GENERATION FUNCTION DETECTED" in content:
                        self.log_message("Loaded existing findings file")
                    else:
                        self.log_message("File loaded successfully")
                    
            except UnicodeDecodeError:
                # Try different encodings if utf-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        self.original_text.delete(1.0, tk.END)
                        self.original_text.insert(1.0, content)
                        self.log_message("File loaded with alternative encoding")
                except Exception as e:
                    self.log_message(f"Error loading file: {str(e)}", error=True)
            except Exception as e:
                self.log_message(f"Error loading file: {str(e)}", error=True)

    def save_output(self):
        if not self.commented_text.get('1.0', tk.END).strip():
            messagebox.showwarning("Warning", "No commented code to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".cpp",
            filetypes=[
                ("C++ Files", "*.cpp"),
                ("C Files", "*.c"),
                ("Header Files", "*.h"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.commented_text.get('1.0', tk.END))
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {str(e)}")

    def process_batch(self, lines_batch):
        """Optimized batch processing for high-end CPU"""
        # Pre-allocate arrays for better memory usage
        filtered_lines = []
        quick_comments = []
        
        # Process in larger chunks for i7-9700k
        chunk_size = 1000  # Increased for better CPU utilization
        
        for i in range(0, len(lines_batch), chunk_size):
            chunk = lines_batch[i:i + chunk_size]
            
            # Parallel processing using all available cores
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Map chunks to workers
                chunk_results = list(executor.map(
                    self.process_single_line,
                    chunk,
                    chunksize=50  # Optimal chunk size for i7-9700k
                ))
                
                # Extend results
                filtered_lines.extend(chunk_results)
        
        return filtered_lines

    @lru_cache(maxsize=2000)
    def process_single_line(self, line):
        """Process assembly line with specific assembly knowledge"""
        try:
            if not line or line.startswith('//'):
                return line
            
            # Clean and parse the assembly line
            stripped = line.strip()
            if not stripped:
                return line
            
            # Handle stack frame variable declarations
            if '=' in stripped and 'ptr' in stripped:
                match = re.match(r'(\w+)=\s*(\w+)\s+ptr\s+([^;\n]+)', stripped)
                if match:
                    var_name, var_type, offset = match.groups()
                    
                    # Determine variable size
                    size_map = {
                        'qword': '64-bit',
                        'dword': '32-bit',
                        'word': '16-bit',
                        'byte': '8-bit'
                    }
                    size = next((size_map[k] for k in size_map if k in var_type.lower()), 'unknown-size')
                    
                    # Determine if it's a parameter or local variable
                    if var_name.startswith('arg_'):
                        offset_val = int(offset.strip('h'), 16) if 'h' in offset else int(offset)
                        return f"{stripped:<40}// Input parameter #{offset_val//8} ({size})\n"
                    else:
                        # Local variable with negative offset from frame pointer
                        offset_val = int(offset.strip('h'), 16) if 'h' in offset else int(offset)
                        return f"{stripped:<40}// Local {size} variable at frame offset -{abs(offset_val)}h\n"
            
            # Handle common assembly instructions
            parts = stripped.lower().split()
            if not parts:
                return line
            
            instruction = parts[0]
            operands = ' '.join(parts[1:])
            
            # Instruction-specific analysis
            if instruction == 'mov':
                if '[rsp' in operands or '[rbp' in operands:
                    reg = operands.split(',')[1].strip()
                    return f"{stripped:<40}// Save {reg} to stack frame\n"
                elif 'rsp' in operands or 'rbp' in operands:
                    return f"{stripped:<40}// Stack pointer/frame setup\n"
                else:
                    src = operands.split(',')[1].strip()
                    dst = operands.split(',')[0].strip()
                    return f"{stripped:<40}// Move {src} into {dst}\n"
                
            elif instruction == 'push':
                reg = operands.strip()
                return f"{stripped:<40}// Push {reg} onto stack\n"
            
            elif instruction == 'pop':
                reg = operands.strip()
                return f"{stripped:<40}// Pop value from stack into {reg}\n"
            
            elif instruction == 'sub':
                if 'rsp' in operands:
                    size = operands.split(',')[1].strip()
                    size_val = int(size.strip('h'), 16) if 'h' in size else int(size)
                    return f"{stripped:<40}// Allocate {size_val} bytes of stack space\n"
                
            elif instruction == 'add':
                if 'rsp' in operands:
                    size = operands.split(',')[1].strip()
                    size_val = int(size.strip('h'), 16) if 'h' in size else int(size)
                    return f"{stripped:<40}// Deallocate {size_val} bytes of stack space\n"
                
            elif instruction == 'call':
                return f"{stripped:<40}// Call function {operands}\n"
            
            elif instruction == 'ret':
                return f"{stripped:<40}// Return from function\n"
            
            elif instruction == 'test':
                return f"{stripped:<40}// Compare {operands} against zero\n"
            
            elif instruction == 'cmp':
                ops = operands.split(',')
                return f"{stripped:<40}// Compare {ops[0].strip()} with {ops[1].strip()}\n"
            
            elif instruction in ['je', 'jz']:
                return f"{stripped:<40}// Jump if equal/zero to {operands}\n"
            
            elif instruction in ['jne', 'jnz']:
                return f"{stripped:<40}// Jump if not equal/nonzero to {operands}\n"
            
            elif instruction == 'jmp':
                return f"{stripped:<40}// Unconditional jump to {operands}\n"
            
            elif instruction == 'lea':
                dst, src = operands.split(',')
                return f"{stripped:<40}// Load effective address of {src.strip()} into {dst.strip()}\n"
            
            elif instruction in ['and', 'or', 'xor']:
                ops = operands.split(',')
                return f"{stripped:<40}// Bitwise {instruction} of {ops[0].strip()} with {ops[1].strip()}\n"
            
            return f"{stripped}\n"
            
        except Exception as e:
            self.log_message(f"Error analyzing instruction: {str(e)}", error=True)
            return f"{stripped}\n"

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused.set(not self.is_paused.get())
        self.pause_btn.config(text="Resume" if self.is_paused.get() else "Pause")

    def analyze_code(self):
        try:
            with open(self.file_path_var.get(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            self.progress['maximum'] = total_lines
            self.commented_text.delete(1.0, tk.END)
            
            # Detect if this is assembly code
            is_assembly = any(line.strip().startswith(('mov ', 'push ', 'pop ', 'lea ', 'sub ', 'add ')) 
                             or '=' in line and 'ptr' in line 
                             for line in lines[:20])
            
            # Initialize analysis state
            current_function = []
            in_function = False
            function_start_line = 0
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                if i % 10 == 0:
                    self.progress['value'] = (i / total_lines) * 100
                    self.root.update_idletasks()
                
                # Assembly code analysis
                if is_assembly:
                    commented_line = self.process_single_line(line)
                    self.commented_text.insert(tk.END, commented_line)
                    continue
                
                # High-level code analysis - detect function boundaries
                if re.match(r'(?i)(\w+\s+)*(\w+)\s*\([^)]*\)\s*{', stripped):
                    in_function = True
                    function_start_line = i
                    current_function = [line]
                    continue
                
                if in_function:
                    current_function.append(line)
                    if stripped == '}':
                        # Analyze complete function for crypto/security patterns
                        is_crypto, evidence, score = self.analyze_function_block(current_function)
                        
                        if is_crypto:
                            self.commented_text.insert(tk.END, "\n// ==========================================\n")
                            self.commented_text.insert(tk.END, "// SECURITY/CRYPTO FUNCTION DETECTED\n")
                            self.commented_text.insert(tk.END, f"// Confidence Score: {score}\n")
                            self.commented_text.insert(tk.END, "// Evidence:\n")
                            for e in evidence:
                                self.commented_text.insert(tk.END, f"// - {e}\n")
                            self.commented_text.insert(tk.END, "// ==========================================\n\n")
                            
                            self.log_message(f"Found crypto/security function at line {function_start_line + 1}")
                        
                        # Add commented function with crypto-specific comments if relevant
                        for func_line in current_function:
                            commented = self.add_crypto_specific_comment(func_line, is_crypto)
                            self.commented_text.insert(tk.END, commented)
                        
                        if is_crypto:
                            self.commented_text.insert(tk.END, "\n// =========== END SECURITY FUNCTION ===========\n\n")
                        
                        current_function = []
                        in_function = False
                else:
                    # Handle non-function code
                    self.commented_text.insert(tk.END, self.add_basic_comment(line))
            
            self.status_var.set("Analysis complete!")
            self.refine_btn.config(state='normal')
            
        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}", error=True)
            raise

    def add_intelligent_comment(self, line, is_crypto=False):
        """Add appropriate comment to a line"""
        stripped = line.strip()
        if not stripped or '//' in stripped:
            return line
        
        comment = ""
        if stripped.startswith('#include'):
            comment = "Include directive for external functionality"
        elif stripped.startswith('typedef'):
            comment = "Type definition"
        elif stripped.startswith('struct'):
            comment = "Structure definition"
        elif is_crypto:
            if 'key' in stripped.lower() or 'crypt' in stripped.lower():
                comment = "Cryptographic operation"
            elif 'random' in stripped.lower() or 'rand' in stripped.lower():
                comment = "Random number generation"
            elif 'return' in stripped.lower() and ('0x' in stripped or 'error' in stripped.lower()):
                comment = "Error handling"
            elif '[' in stripped and ']' in stripped:
                comment = "Buffer operation"
        elif len(stripped.split()) >= 2:
            comment = f"Variable of type {stripped.split()[0]}"
        
        if comment:
            padding = max(1, 40 - len(stripped))
            return f"{stripped}{' ' * padding}// {comment}\n"
        
        return line

    def disable_ui_during_analysis(self):
        """Disable UI elements during analysis"""
        self.analyze_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.clear_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.pause_btn.config(state='normal')

    def enable_ui_after_analysis(self):
        """Re-enable UI elements after analysis"""
        self.analyze_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.clear_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.pause_btn.config(state='disabled')

    def _update_ui_batch(self, new_lines):
        """Update UI with new batch of lines"""
        if not new_lines:
            return
        
        # Store scroll position
        yview = self.commented_text.yview()
        
        # Append new lines
        self.commented_text.insert(tk.END, ''.join(new_lines))
        
        # Restore scroll position
        self.commented_text.yview_moveto(yview[0])

    def _update_ui_complete(self, all_lines):
        """Final UI update"""
        self.commented_text.delete('1.0', tk.END)
        self.commented_text.insert(tk.END, ''.join(all_lines))

    def clear_all(self):
        """Clear all content but keep Refine button enabled"""
        self.original_text.delete(1.0, tk.END)
        self.commented_text.delete(1.0, tk.END)
        self.file_path_var.set("")
        self.status_var.set("Ready")

    def stop_analysis(self):
        """Stop the current analysis"""
        if self.is_analyzing:
            self.stop_flag = True
            self.is_paused.set(False)
            self.log_message("Stopping analysis...")
            self.analyze_btn.config(state='normal')
            self.pause_btn.config(state='disabled')
            self.stop_btn.config(state='disabled')

    def analyze_line(self, line, context=None):
        """Analyze and comment a single line of code"""
        stripped = line.strip()
        if not stripped:
            return line
        
        comment = ""
        
        # Include statements
        if stripped.startswith('#include'):
            lib = stripped.split()[1]
            if '<iostream>' in lib:
                comment = "Include for input and output functions"
            elif '<cstring>' in lib:
                comment = "Include for string handling functions like strlen"
            elif '<sstream>' in lib:
                comment = "Include for string stream functionalities"
            elif '<algorithm>' in lib:
                comment = "Include for algorithms like std::reverse"
            elif '<Windows.h>' in lib:
                comment = "Include for Windows-specific functionalities"
            else:
                comment = f"Include for {lib.strip('<>')} functionality"
        
        # Variable declarations
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=?\s*.*;', stripped):
            parts = stripped.split()
            var_type = parts[0]
            var_name = parts[1].rstrip(';').rstrip('=')
            
            if 'char' in var_type and '[' in stripped:
                comment = "Array to store string data"
            elif 'int' in var_type:
                comment = "Integer variable for counting or indexing"
            elif 'DWORD' in var_type:
                comment = "Windows DWORD type for 32-bit unsigned integer"
            elif 'string' in var_type:
                comment = "String variable for text manipulation"
            elif 'ostringstream' in var_type:
                comment = "Output string stream for string construction"
            else:
                comment = f"Variable of type {var_type}"
        
        # Function calls
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\);', stripped):
            func_name = stripped.split('(')[0].strip()
            if 'strlen' in func_name:
                comment = "Calculate string length"
            elif 'cout' in func_name:
                comment = "Output to console"
            elif 'cin' in func_name:
                comment = "Input from console"
            elif 'system' in func_name:
                comment = "Execute system command"
            else:
                comment = "Function call"
        
        # Control structures
        elif stripped.startswith('for'):
            comment = "Loop for iteration"
        elif stripped.startswith('if'):
            comment = "Conditional check"
        elif stripped.startswith('do'):
            comment = "Do-while loop start"
        elif stripped.startswith('while'):
            comment = "While loop condition"
        elif stripped.startswith('return'):
            comment = "Return statement"
        
        # Operations
        elif '+=' in stripped:
            comment = "Add and assign operation"
        elif '*=' in stripped:
            comment = "Multiply and assign operation"
        elif '-=' in stripped:
            comment = "Subtract and assign operation"
        elif '=' in stripped:
            comment = "Assignment operation"
        
        # Add the comment if we generated one
        if comment:
            padding = max(1, 40 - len(stripped))
            return f"{stripped}{' ' * padding}// {comment}\n"
        
        return line

    def analyze_function_block(self, code_block):
        """Analyze and comment a function block with improved detection"""
        # Handle both string and list input
        if isinstance(code_block, list):
            code_text = ''.join(code_block)
            lines = code_block
        else:
            code_text = code_block
            lines = code_block.split('\n')
        
        commented_lines = []
        
        # Key generation specific patterns - simplified and more focused
        key_indicators = {
            'key_buffer': re.compile(r'(?i)(key.*buffer|buffer.*key|\*key|key\[|key_length)'),
            'key_gen': re.compile(r'(?i)(generate|create|make).*key'),
            'random': re.compile(r'(?i)(random|rand|rng)'),
            'validation': re.compile(r'(?i)(validate|verify|check).*key'),
            'error_return': re.compile(r'return.*0x[fF]+'),
            'buffer_ops': re.compile(r'\[\s*[i]\s*\]\s*='),
        }
        
        # Simple scoring - if we match 2 or more patterns, it's likely key generation
        matches = []
        for name, pattern in key_indicators.items():
            if pattern.search(code_text):
                matches.append(name)
        
        is_key_gen = len(matches) >= 2
        score = len(matches)
        
        # If we found key generation code, add the header BEFORE processing lines
        if is_key_gen:
            commented_lines = [
                "",
                "// ===== KEY GENERATION SEGMENT DETECTED =====",
                "// This function appears to be generating cryptographic keys",
                "// Detected patterns: " + ", ".join(matches),
                "// Exercise caution when modifying this code",
                "// ============================================",
                ""
            ]
        
        # Process each line with comments
        for line in lines:
            stripped = line.strip()
            if not stripped:
                commented_lines.append(line)
                continue
            
            # Add line-specific comments
            comment = ""
            if key_indicators['key_buffer'].search(stripped):
                comment = "Key buffer operation"
            elif key_indicators['key_gen'].search(stripped):
                comment = "Key generation step"
            elif key_indicators['random'].search(stripped):
                comment = "Random number generation"
            elif key_indicators['validation'].search(stripped):
                comment = "Key validation"
            elif key_indicators['error_return'].search(stripped):
                comment = "Error handling"
            elif key_indicators['buffer_ops'].search(stripped):
                comment = "Buffer manipulation"
            
            if comment:
                padding = max(1, 40 - len(stripped))
                commented_lines.append(f"{stripped}{' ' * padding}// {comment}")
            else:
                commented_lines.append(line)
        
        # Add footer if this was key generation code
        if is_key_gen:
            commented_lines.extend([
                "",
                "// ===== END OF KEY GENERATION SEGMENT =====",
                ""
            ])
        
        return is_key_gen, matches, score

    def deep_analyze_function(self, code_block):
        """Use AI model for thorough cryptographic analysis"""
        prompt = f"""Analyze this code for cryptographic and key generation functionality. 
        
Code to analyze:
{code_block}

Consider these key indicators:
1. Variable/type names containing: key, crypt, hash, seed, nonce, iv, salt
2. Buffer operations or byte manipulation
3. Random number generation or entropy collection
4. Bit manipulation operations (XOR, ROL, ROR, shifts)
5. Use of cryptographic constants or magic numbers
6. Validation or verification routines
7. Error return values like 0xFFFFFFFF
8. Parameter patterns like (length, buffer) or (size, key)
9. Struct definitions with security-relevant fields
10. Function names suggesting key operations

Provide detailed analysis focusing on security implications.
Is this code related to key generation, encryption, or other security operations?
Answer format: YES/NO followed by explanation of security-relevant features found.
"""

        # Use more aggressive tokenizer settings
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=2,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=3
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # More sophisticated response analysis
        is_security_relevant = (
            "YES" in response.upper() and 
            any(term in response.lower() for term in [
                "key", "crypt", "security", "random", "generate",
                "hash", "encrypt", "decrypt", "validation"
            ])
        )
        
        if is_security_relevant:
            self.log_message("Security-relevant code detected! Analyzing deeper...")
            
        return is_security_relevant

    def add_basic_comment(self, line):
        """Add basic comments to code lines"""
        stripped = line.strip()
        if not stripped or '//' in line:
            return line
        
        comment = ""
        if stripped.startswith('#include'):
            comment = "Include directive"
        elif stripped.startswith('typedef'):
            comment = "Type definition"
        elif stripped.startswith('struct'):
            comment = "Structure definition"
        elif stripped.startswith('void') or stripped.startswith('int') or stripped.startswith('DWORD'):
            comment = "Function declaration"
        elif '=' in stripped:
            comment = f"Variable assignment"
        elif stripped.startswith('if') or stripped.startswith('while'):
            comment = "Control flow statement"
        elif stripped.startswith('return'):
            comment = "Return statement"
        elif len(stripped.split()) >= 2:
            comment = f"Variable of type {stripped.split()[0]}"
        
        if comment:
            padding = max(1, 40 - len(stripped))
            return f"{stripped}{' ' * padding}// {comment}\n"
        return line + '\n'

    def add_crypto_specific_comment(self, line, is_crypto=False):
        """Add crypto-specific comments to code lines"""
        stripped = line.strip()
        if not stripped or '//' in line:
            return line
            
        crypto_patterns = {
            'key_ops': (r'(?i)(key|keys|keygen)', "Key operation"),
            'hash_ops': (r'(?i)(hash|sha|md5)', "Hash operation"),
            'crypt_ops': (r'(?i)(encrypt|decrypt|cipher)', "Cryptographic operation"),
            'rand_ops': (r'(?i)(random|rand|prng)', "Random number generation"),
            'buffer_ops': (r'\[(0x[0-9a-fA-F]+|\d+)\]', "Buffer manipulation"),
            'bit_ops': (r'(<<|>>|\^|\||\&|rol|ror|xor)', "Bit manipulation"),
            'validation': (r'(?i)(if|while).*?(NULL|0x[fF]+|length|size)', "Validation check"),
            'critical': (r'(LOCK|CRITICAL_SECTION|Enter|Leave)', "Critical section"),
            'error_handling': (r'return.*?(NULL|0x[fF]+)', "Error handling"),
            'memory_ops': (r'(malloc|free|new|delete)', "Memory operation")
        }
        
        comment = ""
        for (pattern, desc) in crypto_patterns.items():
            if re.search(pattern[0], stripped):
                comment = pattern[1]
                break
                
        if not comment and len(stripped.split()) >= 2:
            comment = f"Variable of type {stripped.split()[0]}"
        
        if comment:
            padding = max(1, 40 - len(stripped))
            return f"{stripped}{' ' * padding}// {comment}\n"
        return line + '\n'

    def focus_search(self, entry_widget):
        """Focus the appropriate search box"""
        entry_widget.focus_set()
        entry_widget.select_range(0, tk.END)

    def find_text(self, text_widget, search_text, next=False):
        """Find text in the specified text widget"""
        if not search_text:
            return
        
        # Start from next position if continuing search
        if not hasattr(text_widget, 'last_search_pos') or not next:
            text_widget.last_search_pos = '1.0'
        
        # Find the text
        pos = text_widget.search(search_text, text_widget.last_search_pos, tk.END, nocase=True)
        if pos:
            # Select the found text
            line_end = f"{pos}+{len(search_text)}c"
            text_widget.tag_remove('search', '1.0', tk.END)
            text_widget.tag_add('search', pos, line_end)
            text_widget.tag_config('search', background='yellow')
            text_widget.see(pos)
            text_widget.focus_set()
            
            # Update last position for next search
            text_widget.last_search_pos = line_end
        else:
            # If not found, start from beginning
            if next and text_widget.last_search_pos != '1.0':
                text_widget.last_search_pos = '1.0'
                self.find_text(text_widget, search_text, next=True)
            else:
                messagebox.showinfo("Find", f"Cannot find '{search_text}'")

    def show_refined_findings(self):
        """Show dialog to choose refinement source"""
        choice = messagebox.askyesnocancel(
            "Refine Findings",
            "Would you like to refine:\n\n"
            "Yes - Current analysis\n"
            "No - Load external findings file\n"
            "Cancel - Cancel operation"
        )
        
        if choice is None:  # Cancel
            return
        elif choice:  # Yes - Current analysis
            self._show_refinement_window(self.commented_text.get(1.0, tk.END))
        else:  # No - Load file
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("All files", "*.*"),  # All files first
                    ("Text files", "*.txt"),
                    ("Analysis files", "*.analysis"),
                    ("Log files", "*.log")
                ],
                defaultextension="*.*",  # Default to all files
                title="Select Findings File to Refine"
            )
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self._show_refinement_window(content)
                except UnicodeDecodeError:
                    # Try alternative encoding if UTF-8 fails
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                        self._show_refinement_window(content)
                    except Exception as e:
                        messagebox.showerror("Error", f"Error loading file: {str(e)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def _show_refinement_window(self, content):
        """Show refined crypto findings in a new window"""
        refine_window = tk.Toplevel(self.root)
        refine_window.title("Refined Crypto Findings")
        refine_window.geometry("1000x800")
        
        # Add text widget with scrollbar
        text_frame = ttk.Frame(refine_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        refined_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                               width=80, height=40)
        refined_text.pack(fill=tk.BOTH, expand=True)
        
        # Extract and group crypto sections
        sections = self._extract_crypto_sections(content)
        grouped_sections = self._group_similar_sections(sections)
        
        # Add summary header
        refined_text.insert(tk.END, "=== CRYPTO FINDINGS SUMMARY ===\n\n")
        refined_text.insert(tk.END, f"Total crypto functions found: {len(sections)}\n")
        refined_text.insert(tk.END, f"Grouped into {len(grouped_sections)} categories\n\n")
        
        # Add each group
        for category, group in grouped_sections.items():
            refined_text.insert(tk.END, f"\n{'='*50}\n")
            refined_text.insert(tk.END, f"Category: {category}\n")
            refined_text.insert(tk.END, f"Found {len(group)} related functions\n")
            refined_text.insert(tk.END, f"{'='*50}\n\n")
            
            for section in group:
                refined_text.insert(tk.END, section)
                refined_text.insert(tk.END, "\n" + "-"*50 + "\n\n")
        
        # Add buttons frame
        btn_frame = ttk.Frame(refine_window)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add save button
        save_btn = ttk.Button(btn_frame, text="Save Summary",
                             command=lambda: self._save_refined_findings(refined_text.get(1.0, tk.END)))

def main():
    root = tk.Tk()
    root.title("DeepSeek Code Analyzer")
    root.geometry("1200x900")
    app = CodeCommenterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
