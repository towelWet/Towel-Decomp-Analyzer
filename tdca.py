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
import tempfile
from typing import Optional, List, Set, Dict
import hashlib

class CodeCommenterGUI(tk.Frame):
    def __init__(self, root):
        """Initialize the application"""
        super().__init__(root)
        self.root = root
        self.root.title("DeepSeek Code Analyzer")
        
        # Set initial window size
        self.root.geometry("1800x1000")
        
        # Pack the main frame
        self.pack(fill=tk.BOTH, expand=True)
        
        # Initialize variables
        self.should_stop = False
        self.is_paused = False
        self.evidence = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Initialize code patterns 
        self.code_patterns = {
            'include': r'#include\s*[<"]([^>"]+)[>"]',
            'variable_declaration': r'(\w+)\s+(\w+)\s*=\s*(.*?);',
            'if_statement': r'if\s*\((.*?)\)',
            'return_statement': r'return\s+(.*?);',
            'system_call': r'system\s*\("([^"]*)"\)',
            'assembly': {
                'instruction': r'^(mov|push|pop|lea|sub|add|xor|and|or|shl|shr|cmp|jmp|j[a-z]{1,4}|call|ret|nop|inc|dec|test)\b',
                'label': r'^[a-zA-Z_][a-zA-Z0-9_]*:',
                'directive': r'^(\.|\balign\b|\bdb\b|\bdw\b|\bdd\b|\bdq\b)',
                'memory': r'\[(.*?)\]',
                'register': r'\b(e?[abcd]x|[abcd]l|[abcd]h|[er]?[sb]p|[er]?[sd]i|r\d+[dwb]?)\b'
            }
        }

        # Setup UI first (this will create analyze_btn)
        self.setup_ui()
        
        # Enable analyze button immediately for pattern-based analysis
        self.analyze_btn.config(state='normal')  # Enable by default
        
        # Initialize model in background
        self.model = None
        self.tokenizer = None
        threading.Thread(target=self.setup_models, daemon=True).start()

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
            self.use_ai_var.set(True)  # Enable AI after model loads
            
        except Exception as e:
            self.log_message(f"Error loading DeepSeek model: {str(e)}", error=True)
            self.use_ai_var.set(False)  # Disable AI on error

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
        # Initialize checkbox variables
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.show_confidence_var = tk.BooleanVar(value=False)
        self.use_ai_var = tk.BooleanVar(value=False)  # Default to fast mode
        
        # Configure main window grid
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)  # Add column for crypto findings
        
        # File selection frame
        file_frame = ttk.Frame(self)
        file_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        ttk.Label(file_frame, text="Source File:").pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self, text="Settings")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Create a grid layout for settings
        ttk.Checkbutton(settings_frame, text="Use GPU (if available)", 
                        variable=self.use_gpu_var).grid(row=0, column=0, padx=5, pady=2)
        ttk.Checkbutton(settings_frame, text="Show confidence scores", 
                        variable=self.show_confidence_var).grid(row=0, column=1, padx=5, pady=2)
        
        # Add AI checkbox in the same grid layout
        self.use_ai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Use AI Analysis", 
                        variable=self.use_ai_var,
                        command=self._update_analyze_command).grid(row=0, column=2, padx=5, pady=2)
        
        # Configure grid columns to expand evenly
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_frame.grid_columnconfigure(2, weight=1)
        
        # Three panel frame
        panels_frame = ttk.Frame(self)
        panels_frame.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        panels_frame.grid_columnconfigure(0, weight=1)
        panels_frame.grid_columnconfigure(1, weight=1)
        panels_frame.grid_columnconfigure(2, weight=1)
        
        # Original code panel
        original_frame = ttk.LabelFrame(panels_frame, text="Original Code")
        original_frame.grid(row=0, column=0, sticky='nsew', padx=2)
        
        # Search frame for original code
        search_frame = ttk.Frame(original_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(search_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(search_frame, text="Find").pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="Next").pack(side=tk.LEFT, padx=2)
        
        self.original_text = scrolledtext.ScrolledText(original_frame)
        self.original_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Commented code panel
        commented_frame = ttk.LabelFrame(panels_frame, text="Commented Code")
        commented_frame.grid(row=0, column=1, sticky='nsew', padx=2)
        
        # Search frame for commented code
        search_frame = ttk.Frame(commented_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(search_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(search_frame, text="Find").pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="Next").pack(side=tk.LEFT, padx=2)
        
        self.commented_text = scrolledtext.ScrolledText(commented_frame)
        self.commented_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crypto findings panel
        crypto_frame = ttk.LabelFrame(panels_frame, text="Crypto Findings")
        crypto_frame.grid(row=0, column=2, sticky='nsew', padx=2)
        
        # Search for crypto findings
        crypto_search_frame = ttk.Frame(crypto_frame)
        crypto_search_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        
        self.crypto_search_var = tk.StringVar()
        self.crypto_search_entry = ttk.Entry(crypto_search_frame, textvariable=self.crypto_search_var)
        self.crypto_search_entry.grid(row=0, column=0, sticky='ew', padx=2)
        
        ttk.Button(crypto_search_frame, text="Find",
                   command=lambda: self.find_text(self.crypto_text, self.crypto_search_var.get())).grid(row=0, column=1, padx=2)
        ttk.Button(crypto_search_frame, text="Next",
                   command=lambda: self.find_text(self.crypto_text, self.crypto_search_var.get(), next=True)).grid(row=0, column=2, padx=2)
        crypto_search_frame.grid_columnconfigure(0, weight=1)
        
        # Crypto text area
        self.crypto_text = scrolledtext.ScrolledText(crypto_frame, height=20, width=60)
        self.crypto_text.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Configure text frame weights for three columns
        crypto_frame.grid_columnconfigure(0, weight=1)
        crypto_frame.grid_columnconfigure(1, weight=1)
        crypto_frame.grid_columnconfigure(2, weight=1)
        
        # Buttons frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Initialize analyze button with pattern-based analysis by default
        self.analyze_btn = ttk.Button(
            button_frame, 
            text="Analyze Code", 
            command=self.analyze_code  # Start with pattern-based analysis
        )
        self.analyze_btn.grid(row=0, column=0, padx=5)
        
        # Add Find Crypto button
        self.find_crypto_btn = ttk.Button(
            button_frame,
            text="Find Crypto",
            command=self.find_crypto_functions
        )
        self.find_crypto_btn.grid(row=0, column=1, padx=5)
        
        # Add Refine button
        self.refine_btn = ttk.Button(
            button_frame,
            text="Refine Findings",
            command=self.show_refined_findings,
            state='disabled'  # Disabled until crypto is found
        )
        self.refine_btn.grid(row=0, column=2, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Output", command=self.save_output)
        self.save_btn.grid(row=0, column=3, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_all)
        self.clear_btn.grid(row=0, column=4, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(self, text="Processing Log")
        log_frame.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _update_analyze_command(self):
        """Update analyze button command based on AI checkbox"""
        if self.use_ai_var.get():
            self.analyze_btn.config(
                command=self.analyze_with_deepseek,
                text="Analyze with AI"
            )
            self.log_message("Switched to AI-powered analysis mode")
        else:
            self.analyze_btn.config(
                command=self.analyze_code,
                text="Analyze Code"
            )
            self.log_message("Switched to pattern-based analysis mode")

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

    def log_message(self, message, error=False):
        """Log a message to the log text widget"""
        try:
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            prefix = "ERROR: " if error else "INFO: "
            log_message = f"{timestamp} {prefix}{message}\n"
            
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
            self.update()
        except Exception as e:
            print(f"Error logging message: {str(e)}")

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
    
    def _generate_ai_comment(self, line):
        """Generate AI-powered comment for code line"""
        try:
            # Skip empty lines
            stripped = line.strip()
            if not stripped:
                return line
            
            # Skip existing comments
            if stripped.startswith(('/', ';', '//')):
                return line
            
            # Skip labels and directives
            if stripped.endswith(':') or stripped.startswith('.'):
                return f"{stripped}\n"

            # Generate focused comment
            comment = self._get_code_explanation(stripped)
            
            # Format output with consistent padding
            padding = max(1, 40 - len(stripped))
            return f"{stripped}{' ' * padding}// {comment}\n"
            
        except Exception as e:
            self.log_message(f"Error generating AI comment: {str(e)}", error=True)
            return f"{stripped}\n"

    def _get_code_explanation(self, code):
        """Get a clean, focused explanation for the code"""
        try:
            # First, identify the instruction type
            parts = code.strip().split(None, 1)
            instruction = parts[0].lower() if parts else ""
            
            # Use specific handlers for common instructions
            if instruction in self.instruction_handlers:
                return self.instruction_handlers[instruction](code)
            
            # For other instructions, use AI generation
            prompt = f"Code: {code}\nBrief technical explanation:\n"
            
            # Generate and validate response
            response = self._generate_and_validate_response(prompt, code)
            
            return response
            
        except Exception as e:
            self.log_message(f"Error getting explanation: {str(e)}", error=True)
            return "Performs specified operation."

    def _generate_and_validate_response(self, prompt, code):
        """Generate response and validate it meets quality standards"""
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=30,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=False,
            num_beams=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the response
        response = self._clean_response(response)
        
        # Validate response quality
        if not self._is_valid_response(response, code):
            # Fall back to pattern-based explanation
            return self._get_pattern_based_explanation(code)
        
        return response

    def _clean_response(self, response):
        """Clean up the generated response"""
        # Remove prompt text
        if 'Code:.*?Brief technical explanation:' in response:
            response = response.split('Explain this assembly instruction:', 1)[1]
        
        # Clean up the text
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'^[:\-\s]+', '', response)
        response = re.sub(r'^(This|The|It)\s+(instruction\s+)?', '', response)
        
        # Ensure proper sentence
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        if response and not response.endswith(('.', '?', '!')):
            response += '.'
        
        return response

    def _is_valid_response(self, response, code):
        """Validate response meets quality standards"""
        if not response:
            return False
        
        # Check for incomplete responses
        if any(x in response.lower() for x in ['how it', 'what registers', 'this instruction', '2.', '3.', '4.']):
            return False
        
        # Check for response just echoing the code
        if response.lower().replace(' ', '') in code.lower().replace(' ', ''):
            return False
        
        # Check minimum length and word count
        if len(response) < 10 or len(response.split()) < 3:
            return False
        
        return True

    @property
    def instruction_handlers(self):
        """Dictionary of instruction-specific handlers"""
        return {
            'mov': self._handle_mov,
            'push': self._handle_push,
            'pop': self._handle_pop,
            'lea': self._handle_lea,
            'xor': self._handle_xor,
            'sub': self._handle_sub,
            'add': self._handle_add,
        }

    def _handle_mov(self, code):
        """Handle MOV instruction"""
        parts = code.split(',', 1)
        if len(parts) != 2:
            return "Moves data between registers or memory."
        dest = parts[0].split()[-1]
        src = parts[1].strip()
        return f"Copies value from {src} to {dest}."

    def _handle_push(self, code):
        """Handle PUSH instruction"""
        reg = code.split()[-1]
        return f"Saves {reg} register onto the stack."

    def _handle_pop(self, code):
        """Handle POP instruction"""
        reg = code.split()[-1]
        return f"Restores {reg} register from the stack."

    def _handle_lea(self, code):
        """Handle LEA instruction"""
        parts = code.split(',', 1)
        if len(parts) != 2:
            return "Calculates effective address."
        dest = parts[0].split()[-1]
        src = parts[1].strip()
        return f"Loads effective address of {src} into {dest}."

    def _handle_xor(self, code):
        """Handle XOR instruction"""
        parts = code.split(',', 1)
        if len(parts) != 2:
            return "Performs XOR operation."
        dest = parts[0].split()[-1]
        src = parts[1].strip()
        if dest == src:
            return f"Clears {dest} by XORing with itself."
        return f"Performs XOR between {src} and {dest}."

    def _handle_sub(self, code):
        """Handle SUB instruction"""
        parts = code.split(',', 1)
        if len(parts) != 2:
            return "Subtracts values."
        dest = parts[0].split()[-1]
        src = parts[1].strip()
        return f"Subtracts {src} from {dest}."

    def _handle_add(self, code):
        """Handle ADD instruction"""
        parts = code.split(',', 1)
        if len(parts) != 2:
            return "Adds values."
        dest = parts[0].split()[-1]
        src = parts[1].strip()
        return f"Adds {src} to {dest}."

    def _get_pattern_based_explanation(self, code):
        """Fallback to pattern-based explanation for code"""
        return self.generate_pattern_based_comment(code)

    def _is_assembly_instruction(self, line):
        """Enhanced assembly instruction detection with IDA/Ghidra support"""
        if not line:
            return False
            
        # Clean formatted line first
        stripped = self._clean_ghidra_line(line)
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith(('/', ';', '//', '.')):
            return False
        
        # Check standard assembly patterns
        patterns = {
            'instructions': r'^\s*(mov|push|pop|lea|sub|add|xor|and|or|shl|shr|cmp|jmp|j[a-z]{1,4}|call|ret|nop|inc|dec|test|imul|movzx|movsx)\b',
            'registers': r'\b(r[0-9]+[dwb]?|[er]?[abcd]x|[abcd][hl]|[er]?[sbi]p|[er]?[sd]i|[er]?ip)\b',
            'memory': r'(?:byte|word|dword|qword)?\s*ptr|[\[\]]'
        }
        
        for pattern in patterns.values():
            if re.search(pattern, stripped, re.IGNORECASE):
                return True
                
        return False

    def _clean_ghidra_line(self, line: str) -> str:
        """Clean Ghidra/IDA formatted line and extract instruction"""
        # Match IDA format: .text:ADDRESS INSTRUCTION
        if match := re.match(r'\.[\w\d]+:([0-9a-f]+)\s+(.+)$', line.strip(), re.IGNORECASE):
            return match.group(2).strip()
            
        # Match Ghidra format: [bytes] [address] [instruction]
        elif match := re.match(r'^([0-9a-f\s]+)?\s*([0-9a-f]{8})?\s*(.+)$', line.strip(), re.IGNORECASE):
            _, _, instruction = match.groups()
            return instruction.strip() if instruction else line.strip()
            
        return line.strip()

    def _get_instruction_type(self, line: str) -> str:
        """Get instruction type with Ghidra support"""
        # Clean Ghidra formatting first
        line = self._clean_ghidra_line(line)
        instruction = line.split()[0].lower() if line else ""
        
        # Extended instruction map including Ghidra-specific patterns
        instruction_map = {
            'mov': 'mov',
            'push': 'push', 
            'pop': 'pop',
            'lea': 'lea',
            'sub': 'sub',
            'add': 'add',
            'xor': 'xor',
            'imul': 'imul',  # Added for Ghidra
            'and': 'and',
            'or': 'or',
            'shl': 'shl',
            'shr': 'shr',
            'cmp': 'cmp',
            'test': 'test',
            'jmp': 'jmp',
            'call': 'call',
            'ret': 'ret'
        }
        
        return instruction_map.get(instruction) or ('j' if instruction.startswith('j') else None)

    def _get_instruction_comment(self, line: str, inst_type: str) -> str:
        """Generate comment with IDA/Ghidra support"""
        # Clean format first
        line = self._clean_ghidra_line(line)
        parts = line.split(None, 1)
        if len(parts) < 2:
            return "Invalid instruction format"
        
        instruction = parts[0].lower()
        operands = parts[1] if len(parts) > 1 else ""
        
        # Split operands and clean
        op_parts = [p.strip() for p in operands.split(',')]
        dest = op_parts[0] if op_parts else ""
        src = op_parts[1] if len(op_parts) > 1 else ""
        
        # Enhanced handlers with IDA support
        handlers = {
            'mov': lambda: (
                f"Load {src} into {dest}" if "ptr" in src else
                f"Store value into {dest}" if "ptr" in dest else
                f"Copy {src} to {dest}"
            ),
            'lea': lambda: f"Calculate effective address of {src} into {dest}",
            'call': lambda: f"Call function {src or dest}",
            'sub': lambda: (
                f"Allocate {src} bytes of stack space" if dest.lower() == "rsp" else
                f"Subtract {src} from {dest}"
            ),
            'add': lambda: (
                f"Deallocate {src} bytes of stack space" if dest.lower() == "rsp" else
                f"Add {src} to {dest}"
            ),
            'xor': lambda: (
                f"Zero out {dest}" if dest == src else
                f"XOR {dest} with {src}"
            )
        }
        
        if inst_type in handlers:
            return handlers[inst_type]()
        elif inst_type == 'j':
            condition = instruction[1:].upper()
            target = operands.strip()
            return f"Jump to {target} if {condition} condition is met"
            
        return f"Perform {instruction} operation"

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
                    
                    # Always enable analyze button after loading file
                    self.analyze_btn.config(state='normal')
                    self.log_message("File loaded successfully")
                    
            except UnicodeDecodeError:
                # Try different encodings if utf-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        self.original_text.delete(1.0, tk.END)
                        self.original_text.insert(1.0, content)
                        self.analyze_btn.config(state='normal')
                        self.log_message("File loaded with alternative encoding")
                except Exception as e:
                    self.log_message(f"Error loading file: {str(e)}", error=True)
            except Exception as e:
                self.log_message(f"Error loading file: {str(e)}", error=True)

    def save_output(self):
        """Save output with option to choose which content to save"""
        try:
            # Create selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Save Output")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Center the dialog
            dialog.geometry("+%d+%d" % (
                self.root.winfo_rootx() + self.root.winfo_width()/2 - 150,
                self.root.winfo_rooty() + self.root.winfo_height()/2 - 75
            ))
            
            # Add label
            ttk.Label(dialog, text="Select content to save:").pack(pady=10)
            
            # Add radio buttons
            save_type = tk.StringVar(value="commented")
            ttk.Radiobutton(
                dialog, 
                text="Commented Code", 
                value="commented", 
                variable=save_type
            ).pack(pady=5)
            
            ttk.Radiobutton(
                dialog, 
                text="Crypto Findings", 
                value="crypto", 
                variable=save_type
            ).pack(pady=5)
            
            # Add save button
            def do_save():
                dialog.destroy()
                if save_type.get() == "commented":
                    self._save_content(self.commented_text)
                else:
                    self._save_content(self.crypto_text)
                    
            ttk.Button(
                dialog, 
                text="Save", 
                command=do_save
            ).pack(pady=10)
            
        except Exception as e:
            self.log_message(f"Error in save dialog: {str(e)}", error=True)

    def _save_content(self, text_widget):
        """Save content from specified text widget"""
        try:
            content = text_widget.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("Warning", "No content to save!")
                return
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("Log files", "*.log"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_message(f"Content saved to: {file_path}")
                
        except Exception as e:
            self.log_message(f"Error saving content: {str(e)}", error=True)

    def clear_all(self):
        """Clear all analysis results"""
        self.original_text.delete("1.0", tk.END)
        self.commented_text.delete("1.0", tk.END)
        self.crypto_text.delete("1.0", tk.END)
        self.log_text.delete("1.0", tk.END)
        self.evidence = []
        self.log_message("Analysis cleared")

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

    def process_single_line(self, line):
        """Process line with enhanced AI-powered analysis"""
        try:
            if not line or line.startswith('//'):
                return line
            
            stripped = line.strip()
            if not stripped:
                return line

            # Detect if this is an assembly instruction
            if self._is_assembly_instruction(stripped):
                return self._analyze_assembly_instruction(stripped)
            
            # Use AI model for other code
            return self._generate_ai_comment(stripped)
            
        except Exception as e:
            self.log_message(f"Error analyzing line: {str(e)}", error=True)
            return f"{stripped}\n"

    def _analyze_assembly_instruction(self, line):
        """Analyze assembly instruction and generate clean comment"""
        try:
            # First try pattern-based analysis
            instruction_type = self._get_instruction_type(line)
            if instruction_type:
                comment = self._get_instruction_comment(line, instruction_type)
                if comment:
                    padding = max(1, 40 - len(line))
                    return f"{line}{' ' * padding}// {comment}\n"
            
            # Fall back to AI analysis if needed
            prompt = (
                f"Explain this x86_64 assembly instruction in one clear, technical sentence:\n"
                f"{line}\n"
                f"Focus on what the instruction does, not how it works."
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=1,
                    temperature=0.3,
                    do_sample=False
                )
            
            comment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            comment = self._clean_ai_comment(comment)
            
            padding = max(1, 40 - len(line))
            return f"{line}{' ' * padding}// {comment}\n"
        
        except Exception as e:
            self.log_message(f"Analysis failed: {str(e)}", error=True)
            return f"{line}\n"

    def _clean_ai_comment(self, comment: str) -> str:
        """Clean and format AI-generated comment"""
        # Remove prompt and instruction text
        comment = re.sub(r'.*Comment:', '', comment, flags=re.DOTALL)
        
        # Remove numbered points and their content
        comment = re.sub(r'\d+\.\s+.*?(?=\d+\.|$)', '', comment, flags=re.DOTALL)
        
        # Remove common filler phrases
        filler_phrases = [
            'this line',
            'this code',
            'the code',
            'basically',
            'simply',
            'just',
            'here we',
            'we are'
        ]
        for phrase in filler_phrases:
            comment = re.sub(rf'\b{phrase}\b', '', comment, flags=re.IGNORECASE)
        
        # Clean up formatting
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()
        
        # Ensure proper sentence
        if comment and comment[0].islower():
            comment = comment[0].upper() + comment[1:]
        if comment and not comment.endswith(('.', '?', '!')):
            comment += '.'
        
        # Limit length
        MAX_LEN = 80
        if len(comment) > MAX_LEN:
            comment = comment[:MAX_LEN].rstrip() + '...'
        
        return comment

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused.set(not self.is_paused.get())
        self.pause_btn.config(text="Resume" if self.is_paused.get() else "Pause")

    def analyze_code(self):
        """Analyze code without AI, using pattern matching and crypto detection"""
        try:
            content = self.original_text.get("1.0", tk.END)
            if not content.strip():
                self.log_message("No code to analyze!", error=True)
                return
                
            self.log_message("Starting pattern-based analysis...")
            
            # Process the file line by line
            lines = content.split('\n')
            commented_lines = []
            current_function = []
            in_function = False
            function_start_line = 0
            
            for i, line in enumerate(lines):
                if i % 10 == 0:
                    self.log_message(f"Processing line {i+1}/{len(lines)}")
            
                stripped = line.strip()
                
                # Detect function boundaries for crypto analysis
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
                            commented_lines.append("\n// ==========================================")
                            commented_lines.append("// SECURITY/CRYPTO FUNCTION DETECTED")
                            commented_lines.append(f"// Confidence Score: {score}")
                            commented_lines.append("// Evidence:")
                            for e in evidence:
                                commented_lines.append(f"// - {e}")
                            commented_lines.append("// ==========================================\n")
                            
                            self.log_message(f"Found crypto/security function at line {function_start_line + 1}")
                        
                        # Add function lines with appropriate comments
                        for func_line in current_function:
                            if func_line.strip():
                                if self._is_assembly_instruction(func_line):
                                    instruction_type = self._get_instruction_type(func_line)
                                    if instruction_type:
                                        comment = self._get_instruction_comment(func_line, instruction_type)
                                        commented_lines.append(f"{func_line:<40}// {comment}")
                                    else:
                                        commented_lines.append(func_line)
                                else:
                                    security_patterns = self.analyze_security_patterns(func_line)
                                    if security_patterns:
                                        commented_lines.append(f"{func_line:<40}// {security_patterns[0]}")
                                    elif pattern_comment := self._try_pattern_match(func_line):
                                        commented_lines.append(pattern_comment)
                                    else:
                                        commented_lines.append(func_line)
                            else:
                                commented_lines.append(func_line)
                        
                        in_function = False
                        current_function = []
                    continue
                
                # Process non-function lines
                if stripped:
                    if self._is_assembly_instruction(stripped):
                        instruction_type = self._get_instruction_type(stripped)
                        if instruction_type:
                            comment = self._get_instruction_comment(stripped, instruction_type)
                            commented_lines.append(f"{stripped:<40}// {comment}")
                        else:
                            commented_lines.append(stripped)
                    else:
                        security_patterns = self.analyze_security_patterns(stripped)
                        if security_patterns:
                            commented_lines.append(f"{stripped:<40}// {security_patterns[0]}")
                        elif pattern_comment := self._try_pattern_match(stripped):
                            commented_lines.append(pattern_comment)
                        else:
                            commented_lines.append(stripped)
                else:
                    commented_lines.append(line)
                    
            # Update the commented text widget
            self.commented_text.delete("1.0", tk.END)
            self.commented_text.insert("1.0", '\n'.join(commented_lines))
            
            self.log_message("Analysis complete!")
            
        except Exception as e:
            self.log_message(f"Analysis failed: {str(e)}", error=True)
            import traceback
            self.log_message(traceback.format_exc(), error=True)

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
        """Analyze a function block for cryptographic operations"""
        # Convert list to string if needed
        if isinstance(code_block, list):
            code_block = '\n'.join(code_block)
            
        # Crypto-related patterns
        patterns = {
            'key_generation': [
                r'(?i)(generate|create|init).*?(key|secret|token|iv|nonce)',
                r'(?i)(key\s*=|secret\s*=|token\s*=)',
                r'(?i)(random|rand).*?(key|iv|salt)',
                r'(?i)key(gen|generation|schedule)',
            ],
            'encryption': [
                r'(?i)(encrypt|decrypt|cipher)',
                r'(?i)(AES|DES|RSA|RC4|Blowfish)',
                r'(?i)(CBC|ECB|GCM|CTR|OFB|CFB)',
                r'(?i)(block.*?cipher|stream.*?cipher)',
            ],
            'hashing': [
                r'(?i)(hash|digest|SHA|MD5)',
                r'(?i)(SHA\-?1|SHA\-?2|SHA\-?256|SHA\-?512)',
                r'(?i)(blake2|keccak|whirlpool)',
            ],
            'random': [
                r'(?i)(random|rand).*?(bytes|number|generator)',
                r'(?i)(PRNG|RNG|CSPRNG)',
                r'(?i)(urandom|CryptGenRandom)',
            ],
            'bit_operations': [
                r'(?i)(xor|rotate|shift).*?(left|right|bits?)',
                r'(?i)(bit.*?manipulation|bit.*?operation)',
                r'(?i)(sbox|substitution.*?box)',
                r'(?i)(permutation|transposition)',
            ]
        }
        
        score = 0
        evidence = []
        
        # Check each pattern category
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, code_block)
                if matches:
                    score += len(matches)
                    evidence.append(f"Found {category} pattern: {matches[0]}")
        
        # Check for suspicious variable names
        suspicious_vars = re.findall(r'(?i)(?:^|\s)(key|iv|nonce|salt|cipher|hash|crypt)(?:$|\s|[0-9])', code_block)
        if suspicious_vars:
            score += len(suspicious_vars)
            evidence.append(f"Suspicious variable names: {', '.join(suspicious_vars)}")
        
        # Check for bit manipulation operations
        bit_ops = re.findall(r'(?i)(?:<<|>>|\^|\||\&|\~)', code_block)
        if bit_ops:
            score += len(bit_ops)
            evidence.append(f"Bit manipulation operations found: {len(bit_ops)} occurrences")
        
        # Check for large number operations (common in crypto)
        if re.search(r'0x[0-9A-Fa-f]{8,}', code_block):
            score += 2
            evidence.append("Large hexadecimal constants found")
        
        # Check for array/matrix operations (common in crypto)
        if re.search(r'\[\s*\d+\s*\]\s*\[\s*\d+\s*\]', code_block):
            score += 2
            evidence.append("Matrix operations detected")
        
        # Store evidence for later use
        if evidence:
            self.evidence.extend(evidence)
        
        # Return tuple of (is_crypto, evidence, score)
        is_crypto = score >= 2
        return (is_crypto, evidence, score)

    def _is_assembly_code(self, lines):
        """Detect if code is assembly based on initial lines"""
        asm_indicators = {
            'mov', 'push', 'pop', 'lea', 'call', 'ret', 'jmp', 
            'xor', 'and', 'or', 'shl', 'shr', 'rol', 'ror'
        }
        
        asm_line_count = 0
        for line in lines:
            tokens = line.strip().lower().split()
            if tokens and tokens[0] in asm_indicators:
                asm_line_count += 1
                
        return asm_line_count >= 3

    def _analyze_high_level(self, lines):
        """Analyze high-level code for cryptographic operations"""
        evidence = []
        score = 0
        
        # High-level crypto indicators
        crypto_functions = {
            'encrypt': 'Encryption function',
            'decrypt': 'Decryption function',
            'generatekey': 'Key generation',
            'derivekey': 'Key derivation',
            'hash': 'Hash function',
            'hmac': 'HMAC operation',
            'sign': 'Digital signature',
            'verify': 'Signature verification',
            'random': 'Random number generation',
            'cipher': 'Cipher operation'
        }
        
        crypto_types = {
            'aes': 'AES algorithm',
            'des': 'DES algorithm',
            'rsa': 'RSA algorithm',
            'sha': 'SHA hash',
            'md5': 'MD5 hash',
            'key': 'Cryptographic key',
            'iv': 'Initialization vector',
            'nonce': 'Cryptographic nonce',
            'salt': 'Cryptographic salt'
        }
        
        # Context tracking
        has_crypto_imports = False
        has_key_ops = False
        has_crypto_vars = False
        
        for line in lines:
            stripped = line.strip().lower()
            
            # Check for crypto-related imports
            if 'import' in stripped and any(x in stripped for x in ['crypto', 'cipher', 'security']):
                has_crypto_imports = True
                evidence.append("Crypto-related import found")
                score += 2
            
            # Check for crypto function names
            for func, desc in crypto_functions.items():
                if func in stripped.replace('_', '').replace(' ', ''):
                    evidence.append(f"Found {desc}")
                    score += 2
                    if 'key' in func:
                        has_key_ops = True
            
            # Check for crypto types/variables
            for type_name, desc in crypto_types.items():
                if type_name in stripped:
                    has_crypto_vars = True
                    evidence.append(f"Found {desc} usage")
                    score += 1
        
        is_crypto = (score >= 4 or 
                    (has_crypto_imports and has_crypto_vars) or 
                    (has_key_ops and has_crypto_vars))
        
        return is_crypto, evidence, score

    def _analyze_assembly(self, lines):
        """Analyze assembly code for cryptographic operations"""
        evidence = []
        score = 0
        
        # Track crypto patterns
        key_ops = 0
        bit_ops = 0
        table_lookups = 0
        block_ops = 0
        
        # Common crypto block sizes
        block_sizes = {'0x10', '0x20', '0x40', '16', '32', '64'}
        
        # Crypto constants
        crypto_constants = {
            '0x63516358', '0x52525252',  # AES
            '0x67452301', '0xefcdab89',  # SHA-1
            '0x98badcfe', '0x10325476',
            '0x0f0f0f0f', '0x55555555'   # DES
        }
        
        consecutive_ops = 0
        max_consecutive = 0
        in_loop = False
        
        for line in lines:
            stripped = line.strip().lower()
            
            # Reset consecutive counter on labels or directives
            if ':' in stripped or stripped.startswith('.'):
                consecutive_ops = 0
                continue
                
            # Track loops
            if any(x in stripped for x in ['loop', 'rep', 'jnz', 'jne']):
                in_loop = True
            
            # Check for key operations
            if any(x in stripped for x in ['key', 'schedule', 'sbox']):
                key_ops += 1
                consecutive_ops += 1
            
            # Check for bit manipulation
            if any(x in stripped for x in ['rol', 'ror', 'shl', 'shr', 'xor']):
                bit_ops += 1
                consecutive_ops += 1
            
            # Check for table lookups
            if '[' in stripped and ']' in stripped:
                table_lookups += 1
                consecutive_ops += 1
            
            # Check for block operations
            if any(size in stripped for size in block_sizes):
                block_ops += 1
                consecutive_ops += 1
            
            # Check for crypto constants
            if any(const in stripped for const in crypto_constants):
                evidence.append("Cryptographic constant found")
                score += 2
                consecutive_ops += 1
            
            max_consecutive = max(max_consecutive, consecutive_ops)
        
        # Score the patterns
        if key_ops > 0:
            evidence.append(f"Found {key_ops} key operations")
            score += key_ops
            
        if bit_ops >= 3:
            evidence.append(f"Found {bit_ops} bit manipulation operations")
            score += 2
            
        if table_lookups >= 3:
            evidence.append(f"Found {table_lookups} table lookups")
            score += 2
            
        if block_ops > 0:
            evidence.append(f"Found {block_ops} crypto block size operations")
            score += block_ops
            
        if max_consecutive >= 4:
            evidence.append("Found sequence of crypto operations")
            score += 2
            
        if in_loop:
            evidence.append("Crypto operations in loop structure")
            score += 1
        
        is_crypto = score >= 4 or max_consecutive >= 4 or (key_ops > 0 and bit_ops >= 2)
        
        return is_crypto, evidence, score

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
        """Show refined crypto findings in a new window"""
        try:
            content = self.commented_text.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("Warning", "No findings to refine!")
                return
                
            # Create new window
            refine_window = tk.Toplevel(self.root)
            refine_window.title("Refined Crypto Findings")
            refine_window.geometry("1000x800")
            
            # Add text widget with scrollbar
            text_frame = ttk.Frame(refine_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            refined_text = scrolledtext.ScrolledText(
                text_frame, 
                wrap=tk.WORD,
                width=80, 
                height=40,
                font=('Courier', 10)
            )
            refined_text.pack(fill=tk.BOTH, expand=True)
            
            # Add the content
            refined_text.insert(tk.END, content)
            
            # Add save button
            save_btn = ttk.Button(
                refine_window, 
                text="Save Findings",
                command=lambda: self.save_findings(refined_text.get("1.0", tk.END))
            )
            save_btn.pack(pady=5)
            
        except Exception as e:
            self.log_message(f"Error showing findings: {str(e)}", error=True)

    def save_findings(self, content):
        """Save findings to a file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("All files", "*.*"),
                    ("Text files", "*.txt"),
                    ("Log files", "*.log")
                ],
                title="Save Refined Findings"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_message(f"Findings saved to: {file_path}")
                
        except Exception as e:
            self.log_message(f"Error saving findings: {str(e)}", error=True)

    def process_file(self, content):
        """Process file content using parallel processing"""
        try:
            # Split content into lines
            lines = content.split('\n')
            total_lines = len(lines)
            self.log_message(f"Starting parallel processing of {total_lines} lines...")
            
            # Create temporary directory for line files
            with tempfile.TemporaryDirectory() as temp_dir:
                self.log_message(f"Created temporary directory: {temp_dir}")
                
                # Create tasks for parallel processing
                tasks = []
                for i, line in enumerate(lines):
                    if line.strip():  # Skip empty lines
                        file_path = os.path.join(temp_dir, f'line_{i:05d}.txt')
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(line)
                        tasks.append((file_path, i))
                        if i % 100 == 0:
                            self.log_message(f"Created task file {i}/{total_lines}")
                
                total_tasks = len(tasks)
                self.log_message(f"Created {total_tasks} task files")
                
                # Process lines in parallel
                results = {}
                num_workers = min(32, os.cpu_count() * 4)
                self.log_message(f"Starting processing with {num_workers} workers")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    future_to_line = {
                        executor.submit(self._process_line_file, file_path): (file_path, idx)
                        for file_path, idx in tasks
                    }
                    
                    # Process results as they complete
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_line):
                        file_path, idx = future_to_line[future]
                        try:
                            result = future.result()
                            results[idx] = result
                            completed += 1
                            if completed % 10 == 0:  # More frequent progress updates
                                self.log_message(f"Processed {completed}/{total_tasks} lines ({(completed/total_tasks)*100:.1f}%)")
                        except Exception as e:
                            self.log_message(f"Error processing line {idx} ({file_path}): {str(e)}", error=True)
                            results[idx] = lines[idx]
                
                # Verify all lines were processed
                self.log_message(f"Completed processing {len(results)}/{total_tasks} lines")
                
                # Reconstruct file with comments
                self.log_message("Reconstructing file...")
                commented_lines = []
                for i in range(len(lines)):
                    if i in results:
                        commented_lines.append(results[i])
                    else:
                        self.log_message(f"Missing result for line {i}", error=True)
                        commented_lines.append(lines[i])
                
                self.log_message("Processing complete!")
                return '\n'.join(commented_lines)
                
        except Exception as e:
            self.log_message(f"Error in parallel processing: {str(e)}", error=True)
            return content

    def _process_line_file(self, file_path):
        """Process a single line file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line = f.read().strip()
            
            if not line or line.startswith((';', '//', '{')):
                return line
                
            # Handle variable definitions
            if '=' in line and 'ptr' in line:
                return self._handle_variable_definition(line)
                
            # Handle instructions
            instruction_type = self._get_instruction_type(line)
            if instruction_type:
                comment = self._get_instruction_comment(line, instruction_type)
                padding = max(1, 40 - len(line))
                return f"{line}{' ' * padding}// {comment}"
                
            return line
            
        except Exception as e:
            self.log_message(f"Error processing file {file_path}: {str(e)}", error=True)
            return line

    def _clean_comment(self, comment):
        """Clean and format comment text"""
        # Remove numbered points and filler phrases
        comment = re.sub(r'\b\d+\.\s.*?(?=(\b\d+\.\s|$))', '', comment, flags=re.DOTALL)
        comment = re.sub(r'(?i)(this instruction|what registers|code:|brief technical explanation:|how it)', '', comment)
        
        # Clean up formatting
        comment = re.sub(r'\s+', ' ', comment).strip()
        comment = re.sub(r'^[:\-\s]+', '', comment)
        
        # Ensure proper sentence
        if comment and comment[0].islower():
            comment = comment[0].upper() + comment[1:]
        if comment and not comment.endswith(('.', '?', '!')):
            comment += '.'
            
        # Limit length
        MAX_LEN = 80
        if len(comment) > MAX_LEN:
            comment = comment[:MAX_LEN].rstrip() + "..."
            
        return comment

    def _handle_variable_definition(self, line):
        """Handle variable definition lines"""
        parts = line.split('=')
        if len(parts) != 2:
            return line
            
        var_name = parts[0].strip()
        var_type = parts[1].strip()
        
        size_map = {
            'byte ptr': '8-bit',
            'word ptr': '16-bit',
            'dword ptr': '32-bit',
            'qword ptr': '64-bit',
            'xmmword ptr': '128-bit'
        }
        
        for type_str, size_desc in size_map.items():
            if type_str in var_type:
                offset = var_type.split()[-1]
                is_local = offset.startswith('-')
                desc = f"local {size_desc} variable" if is_local else f"{size_desc} parameter"
                return f"{line:<40}// Defines {desc} at offset {offset}."
        
        return line

    def _get_instruction_type(self, line: str) -> str:
        """Get instruction type with Ghidra support"""
        # Clean Ghidra formatting first
        line = self._clean_ghidra_line(line)
        instruction = line.split()[0].lower() if line else ""
        
        # Extended instruction map including Ghidra-specific patterns
        instruction_map = {
            'mov': 'mov',
            'push': 'push', 
            'pop': 'pop',
            'lea': 'lea',
            'sub': 'sub',
            'add': 'add',
            'xor': 'xor',
            'imul': 'imul',  # Added for Ghidra
            'and': 'and',
            'or': 'or',
            'shl': 'shl',
            'shr': 'shr',
            'cmp': 'cmp',
            'test': 'test',
            'jmp': 'jmp',
            'call': 'call',
            'ret': 'ret'
        }
        
        return instruction_map.get(instruction) or ('j' if instruction.startswith('j') else None)

    def _get_instruction_comment(self, line: str, inst_type: str) -> str:
        """Generate comment with IDA/Ghidra support"""
        # Clean format first
        line = self._clean_ghidra_line(line)
        parts = line.split(None, 1)
        if len(parts) < 2:
            return "Invalid instruction format"
        
        instruction = parts[0].lower()
        operands = parts[1] if len(parts) > 1 else ""
        
        # Split operands and clean
        op_parts = [p.strip() for p in operands.split(',')]
        dest = op_parts[0] if op_parts else ""
        src = op_parts[1] if len(op_parts) > 1 else ""
        
        # Enhanced handlers with IDA support
        handlers = {
            'mov': lambda: (
                f"Load {src} into {dest}" if "ptr" in src else
                f"Store value into {dest}" if "ptr" in dest else
                f"Copy {src} to {dest}"
            ),
            'lea': lambda: f"Calculate effective address of {src} into {dest}",
            'call': lambda: f"Call function {src or dest}",
            'sub': lambda: (
                f"Allocate {src} bytes of stack space" if dest.lower() == "rsp" else
                f"Subtract {src} from {dest}"
            ),
            'add': lambda: (
                f"Deallocate {src} bytes of stack space" if dest.lower() == "rsp" else
                f"Add {src} to {dest}"
            ),
            'xor': lambda: (
                f"Zero out {dest}" if dest == src else
                f"XOR {dest} with {src}"
            )
        }
        
        if inst_type in handlers:
            return handlers[inst_type]()
        elif inst_type == 'j':
            condition = instruction[1:].upper()
            target = operands.strip()
            return f"Jump to {target} if {condition} condition is met"
            
        return f"Perform {instruction} operation"

    def _generate_fallback_comment(self, line):
        """Generate comment using AI with strict validation"""
        try:
            inputs = self.tokenizer(
                f"Explain this assembly instruction briefly: {line}",
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=False,
                num_beams=1
            )
            
            comment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            comment = self._validate_comment(comment, line)
            return comment
            
        except Exception:
            return "Performs specified operation."

    def _validate_comment(self, comment, line):
        """Validate and clean generated comment"""
        # Remove prompt and clean
        comment = re.sub(r'Explain this assembly instruction briefly:', '', comment)
        comment = comment.strip()
        
        # Remove common bad patterns
        if any(x in comment.lower() for x in [
            'how it', 'what registers', '2.', '3.', '4.',
            'this instruction', 'the instruction'
        ]):
            return "Performs specified operation."
        
        # Clean up formatting
        comment = re.sub(r'\s+', ' ', comment)
        comment = re.sub(r'^[:\-\s]+', '', comment)
        
        # Ensure proper sentence
        if comment and comment[0].islower():
            comment = comment[0].upper() + comment[1:]
        if comment and not comment.endswith(('.', '?', '!')):
            comment += '.'
        
        return comment

    def _test_parallel_processing(self):
        """Test parallel processing functionality"""
        test_content = "\n".join([f"mov rax, {i}" for i in range(100)])
        processed = self.process_file(test_content)
        lines = processed.split('\n')
        self.log_message(f"Test processed {len(lines)} lines")
        return all('// Copies value' in line for line in lines if line.strip())

    def _add_high_level_comment(self, line: str) -> str:
        """Generate meaningful comments for high-level code using DeepSeek"""
        stripped = line.strip()
        if not stripped or '//' in stripped:
            return line
        
        try:
            # First try pattern matching for common cases
            if pattern_comment := self._try_pattern_match(stripped):
                return pattern_comment
                
            # Use DeepSeek for complex cases
            prompt = f"""Explain this line of C/C++ code with a technical comment:
{stripped}

Requirements:
- Focus on what the code does technically
- Explain memory operations and data flow
- Be specific about operations and their effects
- Keep it concise (one line)
- Format: code // comment

Example:
*(uint *)(ptr + -0x10) = *(uint *)(ptr + -0x10) + 1; // Increment reference counter in memory block header
"""
            
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            comment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the comment part
            if '//' in comment:
                comment = comment.split('//')[-1].strip()
                comment = self._clean_comment(comment)
                return f"{stripped:<40}// {comment}"
                
            return stripped + '\n'
            
        except Exception as e:
            self.log_message(f"DeepSeek comment generation failed: {str(e)}", error=True)
            return stripped + '\n'

    def _create_comment_prompt(self, line: str) -> str:
        """Create context-aware prompt for the AI model"""
        return f"""Generate a short, technical comment for this C/C++ code line:
{line}

Focus on:
1. What the code does
2. Any important side effects
3. Key operations or data flow

Comment:"""

    def _clean_ai_comment(self, comment: str) -> str:
        """Clean and format AI-generated comment"""
        # Remove prompt and instruction text
        comment = re.sub(r'.*Comment:', '', comment, flags=re.DOTALL)
        
        # Remove numbered points and their content
        comment = re.sub(r'\d+\.\s+.*?(?=\d+\.|$)', '', comment, flags=re.DOTALL)
        
        # Remove common filler phrases
        filler_phrases = [
            'this line',
            'this code',
            'the code',
            'basically',
            'simply',
            'just',
            'here we',
            'we are'
        ]
        for phrase in filler_phrases:
            comment = re.sub(rf'\b{phrase}\b', '', comment, flags=re.IGNORECASE)
        
        # Clean up formatting
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()
        
        # Ensure proper sentence
        if comment and comment[0].islower():
            comment = comment[0].upper() + comment[1:]
        if comment and not comment.endswith(('.', '?', '!')):
            comment += '.'
        
        # Limit length
        MAX_LEN = 80
        if len(comment) > MAX_LEN:
            comment = comment[:MAX_LEN].rstrip() + '...'
        
        return comment

    def _try_pattern_match(self, line: str) -> Optional[str]:
        """Try to match common patterns before using AI"""
        # Variable declarations
        if var_match := re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*\s*\**)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;', line):
            type_name = var_match.group(1).strip()
            var_name = var_match.group(2)
            if '*' in type_name:
                return f"{line:<40}// Declare pointer to {type_name.replace('*', '')}\n"
            return f"{line:<40}// Declare {type_name} variable {var_name}\n"
        
        # Function declarations
        if func_match := re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*\s+[\*\s]*)([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{?', line):
            return f"{line:<40}// Function definition\n"
        
        # Simple patterns that don't need AI
        patterns = {
            r'^\s*return\s*;': "Return from function",
            r'^\s*break\s*;': "Break from loop or switch",
            r'^\s*continue\s*;': "Skip to next iteration",
            r'^\s*LOCK\(\);': "Begin atomic operation",
            r'^\s*UNLOCK\(\);': "End atomic operation",
            r'^\s*{\s*$': "Begin block",
            r'^\s*}\s*$': "End block"
        }
        
        for pattern, comment in patterns.items():
            if re.match(pattern, line):
                return f"{line:<40}// {comment}\n"
        
        return None

    def process_code_with_ai(self, code_text: str) -> str:
        """Process large code files in chunks using AI"""
        # Split into chunks of reasonable size
        chunks = self._split_into_chunks(code_text)
        commented_chunks = []
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Create context-aware prompt for this chunk
                prompt = self._create_chunk_prompt(chunk, chunk_idx, len(chunks))
                
                # Generate comments using AI
                inputs = self.tokenizer(
                    prompt,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Process AI output
                comments = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                commented_chunk = self._apply_ai_comments(chunk, comments)
                commented_chunks.append(commented_chunk)
                
            except Exception as e:
                self.log_message(f"Chunk {chunk_idx} processing failed: {str(e)}", error=True)
                commented_chunks.append(chunk)  # Keep original if AI fails
        
        return '\n'.join(commented_chunks)

    def _split_into_chunks(self, text: str, max_lines: int = 10) -> List[str]:
        """Split code into logical chunks"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            current_chunk.append(line)
            
            # Start new chunk on function boundaries or max lines
            if len(current_chunk) >= max_lines or line.strip() == '}':
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Add remaining lines
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _create_chunk_prompt(self, chunk: str, idx: int, total: int) -> str:
        """Create context-aware prompt for code chunk"""
        return f"""Add technical comments to this C/C++ code segment (chunk {idx + 1} of {total}):

{chunk}

Focus on:
1. Variable declarations and their purpose
2. Function calls and their effects
3. Control flow and conditions
4. Memory operations and pointer usage

Generate concise, technical comments for each line. Format:
code // comment

Comments:"""

    def _apply_ai_comments(self, chunk: str, ai_output: str) -> str:
        """Apply AI-generated comments to code chunk"""
        # Split into lines
        code_lines = chunk.split('\n')
        comment_lines = ai_output.split('\n')
        
        # Clean up AI output
        comment_lines = [
            line.split('//')[-1].strip() 
            for line in comment_lines 
            if '//' in line and line.strip()
        ]
        
        # Ensure we have enough comments
        while len(comment_lines) < len(code_lines):
            comment_lines.append('')
        
        # Combine code with comments
        commented_lines = []
        for code, comment in zip(code_lines, comment_lines):
            if comment:
                commented_lines.append(f"{code:<40}// {comment}")
            else:
                commented_lines.append(code)
        
        return '\n'.join(commented_lines)

    def analyze_with_deepseek(self):
        """Analyze code using DeepSeek model with enhanced AI analysis"""
        try:
            content = self.original_text.get("1.0", tk.END)
            if not content.strip():
                self.log_message("No code to analyze!", error=True)
                return
                
            self.log_message("Starting AI-powered analysis...")
            
            # First do crypto analysis (keep this functionality)
            self.log_message("Running crypto analysis...")
            self.evidence = []
            processed_content = self.process_file(content)
            
            # Extract crypto findings
            sections = self._extract_crypto_sections(processed_content)
            if sections:
                self.log_message("Found potential crypto sections")
                self._show_refinement_window(processed_content)
            
            # Then do AI-powered analysis
            self.log_message("Starting DeepSeek analysis...")
            lines = content.split('\n')
            commented_lines = []
            
            for i, line in enumerate(lines):
                if i % 10 == 0:
                    self.log_message(f"Processing line {i+1}/{len(lines)}")
                    
                if line.strip():
                    try:
                        # Use DeepSeek for each non-empty line
                        prompt = self._create_comment_prompt(line)
                        
                        inputs = self.tokenizer(
                            prompt,
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors="pt",
                            return_attention_mask=True
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=64,
                                temperature=0.1,
                                do_sample=False,
                                num_beams=1
                            )
                        
                        comment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        comment = self._clean_ai_comment(comment)
                        commented_lines.append(f"{line:<40}// {comment}")
                        
                    except Exception as e:
                        self.log_message(f"AI analysis failed for line {i+1}, falling back to pattern matching", warning=True)
                        # Fall back to pattern matching if AI fails
                        if pattern_comment := self._try_pattern_match(line):
                            commented_lines.append(pattern_comment)
                        else:
                            commented_lines.append(line)
                else:
                    commented_lines.append(line)
                    
            # Update the commented text widget
            self.commented_text.delete("1.0", tk.END)
            self.commented_text.insert("1.0", '\n'.join(commented_lines))
            
            self.log_message("AI-powered analysis complete!")
            
        except Exception as e:
            self.log_message(f"AI analysis failed: {str(e)}", error=True)
            import traceback
            self.log_message(traceback.format_exc(), error=True)

    def analyze_crypto(self):
        """Analyze code for cryptographic patterns"""
        try:
            content = self.commented_text.get("1.0", tk.END)
            if not content.strip():
                content = self.original_text.get("1.0", tk.END)
                
            if not content.strip():
                self.log_message("No code to analyze!", error=True)
                return
                
            self.log_message("Starting crypto analysis...")
            self.evidence = []  # Reset evidence list
            
            # Process for crypto findings
            processed_content = self.process_file(content)
            sections = self._extract_crypto_sections(processed_content)
            
            # Clear and update crypto findings
            self.crypto_text.delete("1.0", tk.END)
            
            if sections:
                self.crypto_text.insert(tk.END, "=== CRYPTO FINDINGS ===\n\n")
                for section in sections:
                    self.crypto_text.insert(tk.END, f"Found in section:\n{section}\n\n")
                
                for item in self.evidence:
                    self.crypto_text.insert(tk.END, f"- {item}\n")
            else:
                self.crypto_text.insert(tk.END, "No crypto-related code sections found.")
            
            self.log_message("Crypto analysis complete!")
            
        except Exception as e:
            self.log_message(f"Crypto analysis failed: {str(e)}", error=True)
            import traceback
            self.log_message(traceback.format_exc(), error=True)

    def find_rc4(self, function_text: str) -> tuple[list[str], int]:
        """Detect RC4 implementation in C or ASM"""
        evidence = []
        score = 0
        
        # C patterns
        c_patterns = [
            ('rc4_key', 'Found RC4 key handling'),
            ('prepare_key', 'Found key preparation'),
            ('swap_byte', 'Found byte swapping'),
            ('unsigned char state[256]', 'Found state array'),
            (r'index[12]\s*=', 'Found index manipulation'),
            (r'state\[counter\]', 'Found state counter')
        ]
        
        # ASM patterns
        asm_patterns = [
            (r'(?i)rc4\s+proc', 'Found RC4 procedure'),
            (r'(?i)rc4\s+segment', 'Found RC4 segment'),
            (r'(?i)rc4_key\s+proc', 'Found RC4 key procedure'),
            (r'(?i)prepare_key\s+proc', 'Found key preparation'),
            (r'(?i)swap_byte\s+proc', 'Found byte swapping'),
            (r'(?i)db\s+256\s+dup', 'Found state array'),
            (r'(?i)xchg\s+.*,\s*.*', 'Found exchange operation'),
            (r'(?i)mov\s+byte\s+ptr', 'Found byte movement'),
            (r'(?i)loop\s+\w+', 'Found loop construct')
        ]
        
        # Check C patterns
        for pattern, msg in c_patterns:
            if pattern in function_text or re.search(pattern, function_text, re.IGNORECASE):
                evidence.append(msg)
                score += 1
        
        # Check ASM patterns
        for pattern, msg in asm_patterns:
            if re.search(pattern, function_text):
                evidence.append(msg)
                score += 1
        
        # Additional ASM indicators
        if 'state' in function_text.lower() and 'index' in function_text.lower():
            evidence.append("Found state/index usage")
            score += 1
        
        if score >= 2:
            evidence.append("Found RC4 implementation (C/ASM)")
            score = 2  # Normalize score
            
        return evidence, score

    def find_blowfish(self, content: str) -> tuple[list[str], list[str]]:
        """Detect Blowfish implementation"""
        evidence = []
        findings = []
        
        blowfish_indicators = [
            'Blowfish', 'BLOWFISH', 'blowfish',
            'unsigned long P[18]', 'unsigned long S[4][256]',
            'xL', 'xR', 'F(', 'encipher', 'decipher',
            'decimal', 'checkstack', 'init-boxes'
        ]
        
        for indicator in blowfish_indicators:
            if indicator in content:
                evidence.append(f"Found Blowfish indicator: {indicator}")
        
        if evidence:
            findings.append("\n// ==========================================")
            findings.append("// BLOWFISH IMPLEMENTATION DETECTED")
            findings.append("// Evidence:")
            for e in evidence:
                findings.append(f"// - {e}")
            findings.append("// ==========================================\n")
            start_idx = max(0, content.lower().find('blowfish'))
            end_idx = min(len(content), start_idx + 1000)
            findings.append(content[start_idx:end_idx])
            findings.append("\n// =========== END BLOWFISH IMPLEMENTATION ===========\n\n")
        
        return evidence, findings

    def find_des(self, function_text: str) -> tuple[list[str], int]:
        """Detect DES implementation"""
        evidence = []
        score = 0
        
        # DES patterns
        des_patterns = [
            # Common DES identifiers
            ('des', 'Found DES reference'),
            ('DES', 'Found DES reference'),
            ('Data Encryption Standard', 'Found DES reference'),
            
            # Key handling
            ('des_key', 'Found DES key handling'),
            ('key_schedule', 'Found key schedule'),
            ('key_permutation', 'Found key permutation'),
            ('subkey', 'Found subkey generation'),
            
            # Core operations
            ('des_encrypt', 'Found DES encryption'),
            ('des_decrypt', 'Found DES decryption'),
            ('initial_permutation', 'Found initial permutation'),
            ('final_permutation', 'Found final permutation'),
            ('permuted_choice_1', 'Found PC1'),
            ('permuted_choice_2', 'Found PC2'),
            ('expansion_permutation', 'Found E-box'),
            ('p_box', 'Found P-box'),
            
            # Data structures
            ('unsigned char des_key[8]', 'Found DES key array'),
            ('unsigned long left', 'Found DES block processing'),
            ('unsigned long right', 'Found DES block processing'),
            ('[LR]\\d\\s*=', 'Found DES round operation'),
            
            # ASM specific
            (r'(?i)des\s+proc', 'Found DES procedure'),
            (r'(?i)des\s+segment', 'Found DES segment'),
            (r'(?i)des_key\s+proc', 'Found DES key procedure'),
            
            # Common operations
            ('feistel', 'Found Feistel network'),
            ('round_function', 'Found round function'),
            ('f_function', 'Found F function'),
            
            # Specific constants
            ('28', 'Found potential DES rotation constant'),
            ('48', 'Found potential DES expansion size'),
            ('32', 'Found potential DES block half size'),
            ('56', 'Found potential DES key size')
        ]
        
        # Check patterns
        for pattern, msg in des_patterns:
            if pattern in function_text or re.search(pattern, function_text, re.IGNORECASE):
                evidence.append(msg)
                score += 1
        
        # Check for S-box references
        if self.find_sbox(function_text)[1] > 0:
            evidence.append("Found S-box usage in DES context")
            score += 1
        
        # Check for characteristic DES operations
        if re.search(r'(?i)(rotate|shift|xor)\s*.*\s*(28|56)', function_text):  # DES uses 28-bit rotations
            evidence.append("Found DES key schedule operations")
            score += 1
        
        # Check for DES-specific bit operations
        if re.search(r'(?i)(<<|>>|rol|ror)\s*[1-2]', function_text):
            evidence.append("Found DES bit rotations")
            score += 1
        
        # Check for 16 rounds structure
        if re.search(r'(?i)for\s*\(\s*\w+\s*=\s*0\s*;\s*\w+\s*<\s*16\s*;', function_text):
            evidence.append("Found DES round structure")
            score += 1
        
        # Check for permutation tables
        if re.search(r'(?i)(PC1|PC2|IP|FP|E|P)\s*\[\s*\d+\s*\]', function_text):
            evidence.append("Found DES permutation tables")
            score += 1

        if score >= 2:
            evidence.append("Found DES implementation")
            score = 2  # Normalize score
            
        return evidence, score

    def find_sbox(self, function_text: str) -> tuple[list[str], int]:
        """Detect S-box implementations"""
        evidence = []
        score = 0
        
        # S-box patterns
        sbox_patterns = [
            # Common S-box declarations
            (r'(?i)s_?box', 'Found S-box reference'),
            (r'(?i)sbox\s*\[\s*\d+\s*\]', 'Found S-box array'),
            (r'(?i)substitution_box', 'Found substitution box'),
            (r'unsigned\s+char\s+[Ss](?:\d|_box)', 'Found S-box declaration'),
            
            # Common sizes
            (r'\[\s*8\s*\]\s*\[\s*64\s*\]', 'Found DES-like S-box dimensions'),
            (r'\[\s*256\s*\]', 'Found 256-entry S-box'),
            (r'\[\s*16\s*\]\s*\[\s*16\s*\]', 'Found AES-like S-box dimensions'),
            
            # ASM patterns
            (r'(?i)sbox\s+segment', 'Found S-box segment'),
            (r'(?i)sbox\s+proc', 'Found S-box procedure'),
            (r'(?i)db\s+\d+\s*,\s*\d+\s*,\s*\d+\s*,', 'Found S-box data'),
            
            # Lookup operations
            (r'[Ss](?:\d|_box)\s*\[\s*\w+\s*\]', 'Found S-box lookup'),
            (r'substitute\s*\(\s*\w+\s*\)', 'Found substitution operation')
        ]
        
        # Check patterns
        for pattern, msg in sbox_patterns:
            if re.search(pattern, function_text):
                evidence.append(msg)
                score += 1
        
        # Check for characteristic S-box content
        if re.search(r'(?i)(0x[0-9a-f]{2}\s*,\s*){8,}', function_text):
            evidence.append("Found S-box lookup table")
            score += 1
        
        if score >= 2:
            evidence.append("Found S-box implementation")
            score = 2  # Normalize score
            
        return evidence, score

    def find_crypto_functions(self):
        """Find cryptographic functions in the code"""
        try:
            content = self.original_text.get("1.0", tk.END)
            if not content.strip():
                self.log_message("No code to analyze!", error=True)
                return
                
            self.log_message("Starting crypto analysis...")
            self.evidence = []  # Reset evidence list
            
            # Check for Blowfish
            blowfish_evidence, blowfish_findings = self.find_blowfish(content)
            self.evidence.extend(blowfish_evidence)
            crypto_findings = blowfish_findings
            
            # Split into potential function blocks
            lines = content.split('\n')
            current_function = []
            in_function = False
            
            # Analyze each function
            for line in lines:
                stripped = line.strip()
                
                # Check for function start
                if re.match(r'(?i)(\w+\s+)*(\w+)\s*\([^)]*\)\s*{', stripped) or re.search(r'typedef\s+struct', stripped) or re.search(r'(?i)proc\s+|segment\s+', stripped):
                    in_function = True
                    current_function = [line]
                # Inside function/struct
                elif in_function:
                    current_function.append(line)
                    if stripped == '}' or re.search(r'(?i)endp\s*$', stripped):
                        # Analyze complete block
                        function_text = '\n'.join(current_function)
                        evidence = []
                        score = 0
                        
                        # Keep original pattern matching
                        if re.search(r'(?i)(random|rand|RNG)', function_text):
                            evidence.append("Found random pattern: RNG")
                            score += 1
                        if re.search(r'(?i)(key|encrypt|decrypt|cipher)', function_text):
                            evidence.append("Found crypto pattern: cryptographic operations")
                            score += 1
                        if re.search(r'0x[0-9a-fA-F]{6,}', function_text):
                            evidence.append("Large hexadecimal constants found")
                            score += 1

                        # Check for RC4
                        rc4_evidence, rc4_score = self.find_rc4(function_text)
                        evidence.extend(rc4_evidence)
                        score += rc4_score

                        # Check for DES
                        des_evidence, des_score = self.find_des(function_text)
                        evidence.extend(des_evidence)
                        score += des_score

                        # Check for standalone S-box (not part of other algorithms)
                        sbox_evidence, sbox_score = self.find_sbox(function_text)
                        if (sbox_score > 0 and des_score == 0):  # Only count if not part of DES
                            evidence.extend(sbox_evidence)
                            score += sbox_score

                        # Keep common crypto implementation features
                        if re.search(r'(?i)(rounds?|iterations?)\s*=\s*\d+', function_text):
                            evidence.append("Found round structure")
                            score += 1
                        if re.search(r'(?i)(rotate|shift|xor|[lr]?or|and)\s*\(', function_text):
                            evidence.append("Found bit manipulation operations")
                            score += 1
                            
                        # Report findings
                        if score >= 2:
                            crypto_findings.append("\n// ==========================================")
                            crypto_findings.append("// SECURITY/CRYPTO FUNCTION DETECTED")
                            crypto_findings.append(f"// Confidence Score: {score}")
                            crypto_findings.append("// Evidence:")
                            for e in evidence:
                                crypto_findings.append(f"// - {e}")
                                self.evidence.append(e)
                            crypto_findings.append("// ==========================================\n")
                            crypto_findings.extend(current_function)
                            crypto_findings.append("\n// =========== END CRYPTO FUNCTION ===========\n")
                        
                        current_function = []
                        in_function = False
            
            if self.evidence:
                self.log_message(f"Found {len(self.evidence)} potential crypto indicators")
                self.refine_btn.config(state='normal')
                
                if hasattr(self, 'crypto_text'):
                    self.crypto_text.delete("1.0", tk.END)
                    self.crypto_text.insert("1.0", '\n'.join(crypto_findings))
            else:
                self.log_message("No crypto functions detected")
                self.refine_btn.config(state='disabled')
                if hasattr(self, 'crypto_text'):
                    self.crypto_text.delete("1.0", tk.END)
            
        except Exception as e:
            self.log_message(f"Crypto analysis failed: {str(e)}", error=True)
            import traceback
            self.log_message(traceback.format_exc(), error=True)

def main():
    root = tk.Tk()
    root.title("DeepSeek Code Analyzer")
    root.geometry("1800x1000")  # Increased size
    app = CodeCommenterGUI(root)
    if app._test_parallel_processing():
        print("Parallel processing test passed!")
    root.mainloop()

if __name__ == "__main__":  # Use == instead of ": "
    main()
