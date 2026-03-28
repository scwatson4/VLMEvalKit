"""
DCVLR Standalone Scorer

A standalone scoring system for DCVLR (Data Curation for Vision-Language Reasoning) benchmarks 
that implements a 4-stage answer matching pipeline incorporating strategies from multiple VLMEvalKit benchmarks.

Usage:
    python scripts/dcvlr_standalone_scorer.py \
        --benchmarks VMCBench_DEV VMCBench_TEST \
        --input-dir results/full/Qwen2.5-VL-7B-Instruct \
        --llm-backend openai \
        --model gpt-4o-mini \
        --verbose

    python scripts/dcvlr_standalone_scorer.py \
        --benchmarks VMCBench_DEV \
        --input-dir results/full/Qwen2.5-VL-7B-Instruct \
        --llm-backend qwen \
        --model qwen3-4b
"""

import argparse
import json
import logging
import os
import re
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Mathematical libraries
try:
    import sympy as sp
    from sympy import simplify, expand, trigsimp, sympify, Eq
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Mathematical equivalence checking will be limited.")

# LLM backend libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Yale Physics benchmarks
YALE_PHYSICS_BENCHMARKS = [
    'atomic_dataset',
    'electro_dataset', 
    'mechanics_dataset',
    'optics_dataset',
    'quantum_dataset',
    'statistics_dataset'
]


class VLMEvalKitScorer:
    """
    Main scorer class that implements the 4-stage answer matching pipeline.
    """
    
    def __init__(self, benchmarks: List[str], input_dir: str, output_dir: Optional[str] = None,
                 llm_backend: str = 'openai', model: str = 'gpt-4o-mini', 
                 api_key: Optional[str] = None, verbose: bool = False, max_samples: Optional[int] = None,
                 resume: bool = False, num_workers: Optional[int] = None,
                 qwen_judge_batch_size: int = 32):
        """
        Initialize the VMCBench scorer.
        
        Args:
            benchmarks: List of benchmark names to process
            input_dir: Directory containing input XLSX files
            output_dir: Directory for output files (defaults to input_dir)
            llm_backend: LLM backend ('openai' or 'anthropic')
            model: Model name for LLM judge
            api_key: API key for LLM service
            verbose: Enable verbose logging
            max_samples: Maximum number of samples to process per benchmark (for testing)
            resume: Resume from existing results file by skipping processed samples
            num_workers: Number of worker threads for the scoring pipeline
            qwen_judge_batch_size: Batch size for qwen/vLLM Stage 3 judging
        """
        self.benchmarks = benchmarks
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.verbose = verbose
        self.max_samples = max_samples
        self.resume = resume
        self.qwen_judge_batch_size = max(1, qwen_judge_batch_size)
        self._invalid_char_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
        cpu_count = os.cpu_count() or 4
        if num_workers is None:
            self.num_workers = max(1, min(8, cpu_count))
        else:
            self.num_workers = max(1, num_workers)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize LLM judge
        self.llm_judge = self._init_llm_judge(llm_backend, model, api_key)
        
        self.logger.info(f"Initialized VMCBench scorer for benchmarks: {benchmarks}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_llm_judge(self, backend: str, model: str, api_key: Optional[str]):
        """Initialize LLM judge backend."""
        if backend == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available. Install with: pip install openai")
            return OpenAIJudge(model=model, api_key=api_key)
        elif backend == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available. Install with: pip install anthropic")
            return AnthropicJudge(model=model, api_key=api_key)
        elif backend == 'qwen':
            return QwenJudge(model=model, batch_size=self.qwen_judge_batch_size)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")
    
    def find_benchmark_files(self) -> Dict[str, Path]:
        """
        Find XLSX files matching the benchmark naming pattern.
        
        Returns:
            Dictionary mapping benchmark names to file paths
        """
        found_files = {}
        
        for benchmark in self.benchmarks:
            # Pattern: {model_name}_{benchmark_name}.xlsx
            pattern = f"*_{benchmark}.xlsx"
            matches = [f for f in self.input_dir.glob(pattern) if not f.name.startswith('~$')]
            
            if not matches and benchmark in YALE_PHYSICS_BENCHMARKS:
                # Try fallback patterns for physics datasets
                fallback_patterns = [
                    f"*_{benchmark}_gpt_4o_mini.xlsx",  # Look for judge result files first
                ]
                
                for fallback_pattern in fallback_patterns:
                    fallback_matches = [f for f in self.input_dir.glob(fallback_pattern) if not f.name.startswith('~$')]
                    if fallback_matches:
                        matches = fallback_matches
                        self.logger.info(f"Found {benchmark} file using fallback pattern '{fallback_pattern}': {fallback_matches[0]}")
                        break
            
            if matches:
                if len(matches) > 1:
                    self.logger.warning(f"Multiple files found for {benchmark}: {matches}")
                    self.logger.warning(f"Using: {matches[0]}")
                found_files[benchmark] = matches[0]
                self.logger.info(f"Found file for {benchmark}: {matches[0]}")
            else:
                self.logger.error(f"No file found for benchmark {benchmark} with pattern {pattern}")
        
        return found_files
    
    def load_benchmark_data(self, file_path: Path, benchmark_name: str) -> pd.DataFrame:
        """
        Load benchmark data from XLSX file.
        
        Args:
            file_path: Path to XLSX file
            benchmark_name: Name of the benchmark being processed
            
        Returns:
            DataFrame with benchmark data
        """
        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Handle OlympiadBench-style data with final_answer column
            if 'answer' not in df.columns and 'final_answer' in df.columns:
                print(f"\nWarning: 'answer' column not detected but 'final_answer' column found.")
                print(f"Assuming this is OlympiadBench or similar benchmark with list-wrapped answers.")
                print(f"Converting 'final_answer' to 'answer' by removing list wrapper ['...']")
                
                # Create answer column by extracting from final_answer
                df['answer'] = df['final_answer'].apply(lambda x: str(x)[2:-2] if isinstance(x, str) and len(str(x)) > 4 else str(x))
                
                # Handle multiple answers - drop for now
                if 'is_multiple_answer' in df.columns:
                    multi_answer_mask = df['is_multiple_answer'] == 1.0
                    num_multi = multi_answer_mask.sum()
                    if num_multi > 0:
                        print(f"\nWarning: Found {num_multi} rows with is_multiple_answer=True.")
                        print(f"Dropping these rows from evaluation as multiple answer handling is not yet supported.")
                        df = df[~multi_answer_mask].reset_index(drop=True)
                        self.logger.info(f"Dropped {num_multi} multiple answer rows, {len(df)} rows remaining")
            
            # Validate required columns
            required_cols = ['answer', 'prediction']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure we have a unique identifier column
            if 'index' not in df.columns and df.index.name != 'index':
                df = df.reset_index(drop=False)
                if 'index' not in df.columns:
                    df['index'] = range(len(df))
            
            # Handle resume mode
            if self.resume:
                df = self._filter_unprocessed_samples(df, file_path, benchmark_name)
            
            # Convert to string and handle NaN values
            df['answer'] = df['answer'].astype(str).fillna('')
            df['prediction'] = df['prediction'].astype(str).fillna('')
            
            # Apply sampling if max_samples is specified
            if self.max_samples and len(df) > self.max_samples:
                df = df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
                self.logger.info(f"Sampled {len(df)} rows from original dataset")
            
            self.logger.info(f"Available columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def _extract_choices(self, row: pd.Series) -> Dict[str, str]:
        """
        Extract multiple choice options from row if available.
        
        Args:
            row: Pandas Series containing row data
            
        Returns:
            Dictionary mapping choice letters to choice text
        """
        choices = {}
        for i in range(9):  # Support up to 9 choices (A-Z)
            choice_letter = chr(65 + i)  # A, B, C, D, ...
            if choice_letter in row and pd.notna(row[choice_letter]):
                choices[choice_letter] = str(row[choice_letter])
        return choices
    
    def _sanitize_text(self, value: Any) -> Any:
        """Strip worksheet-illegal control characters from strings."""
        if isinstance(value, str):
            return self._invalid_char_pattern.sub('', value)
        return value

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text sanitization across an entire DataFrame."""
        return df.applymap(self._sanitize_text)
    
    def _filter_unprocessed_samples(self, df: pd.DataFrame, file_path: Path, benchmark_name: str) -> pd.DataFrame:
        """
        Filter out samples that have already been processed when resuming.
        
        Args:
            df: Input DataFrame
            file_path: Path to the input file
            benchmark_name: Name of the benchmark
            
        Returns:
            DataFrame with only unprocessed samples
        """
        output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
        
        if not output_path.exists():
            raise FileNotFoundError(
                f"Resume mode enabled but no existing results file found at: {output_path}. "
                f"Remove --resume flag to create a new results file."
            )
        
        try:
            # Load existing results
            existing_df = pd.read_excel(output_path)
            self.logger.info(f"Found existing results file with {len(existing_df)} rows")
            
            # Get processed sample indices/IDs
            if 'index' in existing_df.columns:
                processed_indices = set(existing_df['index'].values)
            else:
                processed_indices = set(existing_df.index.values)
            
            # Filter out processed samples
            if 'index' in df.columns:
                mask = ~df['index'].isin(processed_indices)
            else:
                mask = ~df.index.isin(processed_indices)
            
            unprocessed_df = df[mask].reset_index(drop=True)
            
            self.logger.info(
                f"Resume mode: Found {len(existing_df)} processed samples, "
                f"{len(unprocessed_df)} samples remaining to process"
            )
            
            return unprocessed_df
            
        except Exception as e:
            raise RuntimeError(
                f"Error reading existing results file {output_path}: {e}. "
                f"File may be corrupted or in wrong format."
            )
    
    def _append_results_to_existing(self, new_results: pd.DataFrame, output_path: Path):
        """
        Append new results to existing results file.
        
        Args:
            new_results: New DataFrame with results to append
            output_path: Path to the existing results file
        """
        try:
            existing_df = pd.read_excel(output_path)
            combined_df = pd.concat([existing_df, new_results], ignore_index=True)
            combined_df = self._sanitize_dataframe(combined_df)
            combined_df.to_excel(output_path, index=False)
            self.logger.info(f"Appended {len(new_results)} new results to existing file")
        except Exception as e:
            self.logger.error(f"Error appending results: {e}")
            raise
    
    def process_benchmark(self, benchmark_name: str, file_path: Path):
        """
        Process a single benchmark file through the 4-stage pipeline.
        
        Args:
            benchmark_name: Name of the benchmark
            file_path: Path to the benchmark file
        """
        self.logger.info(f"Processing benchmark: {benchmark_name}")
        
        # Load data
        df = self.load_benchmark_data(file_path, benchmark_name)
        
        # Check if there are any samples to process
        if len(df) == 0:
            self.logger.info(f"No samples to process for {benchmark_name} (all already processed)")
            return
        
        # Apply 4-stage pipeline
        df_scored = self.apply_four_stage_pipeline(df)
        df_scored = self._sanitize_dataframe(df_scored)
        
        # Save results
        output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
        
        if self.resume and output_path.exists():
            # Append to existing results
            self._append_results_to_existing(df_scored, output_path)
        else:
            # Create new results file
            df_scored.to_excel(output_path, index=False)
            self.logger.info(f"Saved scored results to: {output_path}")
        
        # Clean up intermediate files
        self._cleanup_intermediate_files()
        
        # Print summary statistics
        self._print_summary_stats(df_scored, benchmark_name)
    
    def stage1_simple_match(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """
        Stage 1: Simple exact matching strategies only.
        
        Applies only the most conservative exact matching strategies:
        1. Exact match: prediction == answer (after stripping)
        2. Case-insensitive match: prediction.lower() == answer.lower()
        3. Whitespace-normalized match: handles extra/missing whitespace
        
        Note: Moved single character extraction and boolean pattern matching to Stage 2
        to consolidate with similar strategies and reduce duplication.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        # Clean inputs
        pred_clean = prediction.strip()
        answer_clean = answer.strip()
        
        # Strategy 1: Exact match
        if pred_clean == answer_clean:
            return answer_clean, True, "Exact match"
        
        # Strategy 2: Case-insensitive match
        if pred_clean.lower() == answer_clean.lower():
            return answer_clean, True, "Case-insensitive match"
        
        # Strategy 3: Extra whitespace removal
        pred_no_space = re.sub(r'\s+', ' ', pred_clean).strip()
        answer_no_space = re.sub(r'\s+', ' ', answer_clean).strip()
        if pred_no_space.lower() == answer_no_space.lower():
            return answer_clean, True, "Whitespace-normalized match"
        
        return pred_clean, False, "No simple match found"
    
    def stage2_complex_match(self, prediction: str, answer: str, 
                           choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Stage 2: Unified complex extraction strategies with comprehensive match collection.
        
        Behavior:
        - Searches from END of response to prioritize final answers over intermediate work
        - Collects ALL matches from ALL strategies for full visibility
        - Uses priority order: LaTeX > Math > Tags > SymPy > MCQ* > Language > Boolean*
        - Applies heuristics to reduce false positives:
          * MCQ and Boolean detectors only run when ground truth is short (<15 chars)
          * Numeric detector removed completely (too many false positives)
          * SymPy equivalence prioritized above simple pattern matching
        - Returns first successful match but reports all findings in error message
        - Includes match positions in text for debugging
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Dictionary of multiple choice options (optional)
            
        Returns:
            Tuple of (extracted_answer, success, detailed_match_report)
        """
        # Define core strategies (always run)
        core_strategies = [
            ("LaTeX Boxed", lambda: self._extract_latex_boxed_with_positions(prediction)),
            ("Math Expressions", lambda: self._extract_math_expressions_with_positions(prediction)),
            ("Structured Tags", lambda: self._extract_structured_tags_with_positions(prediction)),
            ("SymPy Equivalence", lambda: self._check_mathematical_equivalence_with_positions(prediction, answer)),
            ("Natural Language", lambda: self._extract_natural_language_with_positions(prediction)),
        ]
        
        # Define conditional strategies (only run when ground truth is short)
        conditional_strategies = []
        if len(answer.strip()) < 15:  # Only run for short ground truth to reduce false positives
            conditional_strategies.extend([
                ("Multiple Choice", lambda: self._extract_multiple_choice_with_positions(prediction, choices_dict)),
                ("Boolean Answers", lambda: self._extract_boolean_answers_with_positions(prediction)),
            ])
        
        # Combine strategies in priority order
        strategies = core_strategies + conditional_strategies
        # Note: Numeric Answers detector removed completely due to excessive false positive risk
        
        all_matches = []
        selected_result = None
        selected_strategy = None
        
        # Collect all matches from all strategies
        for strategy_name, strategy_func in strategies:
            try:
                matches = strategy_func()
                if matches:  # matches is now a list of (content, start_pos, end_pos) tuples
                    all_matches.extend([(strategy_name, content, start_pos, end_pos) for content, start_pos, end_pos in matches])
                    
                    # Select first valid result using priority order (if not already selected)
                    if not selected_result:
                        # Get the last (rightmost) match from this strategy
                        last_match = max(matches, key=lambda x: x[1])  # max by start_pos
                        content = last_match[0]
                        if content and content.strip() != prediction.strip():
                            # Strict length-based heuristic: require exact length match for structured extraction
                            # This prevents structured tag extraction from returning full explanations instead of short answers
                            gt_len = len(answer.strip())
                            content_len = len(content.strip())
                            
                            # Apply strict length filter for structured extraction strategies
                            # Exception: If both ground truth and content can be converted to float AND neither are integers, bypass length filter
                            should_bypass_length_filter = False
                            try:
                                gt_float = float(answer.strip())
                                content_float = float(content.strip())
                                # Only bypass if both convert to float AND at least one has decimal places (not an integer)
                                gt_is_int = gt_float.is_integer()
                                content_is_int = content_float.is_integer()
                                should_bypass_length_filter = not (gt_is_int and content_is_int)
                            except (ValueError, TypeError):
                                pass
                            
                            if (strategy_name in ["LaTeX Boxed", "Structured Tags", "Math Expressions", "Natural Language", "Boolean Answers"] 
                                and not should_bypass_length_filter):
                                # Require exact length match - if lengths don't match, reject this extraction
                                if content_len != gt_len:
                                    continue  # Skip this match, try next strategy
                            
                            selected_result = content.strip()
                            selected_strategy = strategy_name
                            
            except Exception as e:
                all_matches.append((strategy_name, f"ERROR: {str(e)}", -1, -1))
                continue
        
        # Build comprehensive match report
        if all_matches:
            # Sort matches by position (rightmost first for "end-searching" perspective)
            position_sorted = sorted([m for m in all_matches if m[2] >= 0], key=lambda x: x[2], reverse=True)
            error_matches = [m for m in all_matches if m[2] == -1]
            
            match_details = []
            for strategy, content, start, end in position_sorted:
                if start >= 0:
                    match_details.append(f"{strategy}@{start}-{end}: '{content}'")
            
            for strategy, error, _, _ in error_matches:
                match_details.append(f"{strategy}: {error}")
            
            # Add heuristic info to report
            heuristic_info = f"GT_len={len(answer.strip())}, MCQ/Bool={'enabled' if len(answer.strip()) < 15 else 'disabled'}, Strict_length_filter=enabled"
            match_report = f"Matches found: {'; '.join(match_details)} | Heuristics: {heuristic_info}"
            
            if selected_result:
                # Check if selected result matches ground truth
                if selected_result.lower() == answer.lower():
                    return answer, True, f"SUCCESS with {selected_strategy}: '{selected_result}' | {match_report}"
                else:
                    return selected_result, True, f"EXTRACTED with {selected_strategy}: '{selected_result}' | {match_report}"
            else:
                return prediction, False, f"NO VALID EXTRACTION | {match_report}"
        else:
            heuristic_info = f"GT_len={len(answer.strip())}, MCQ/Bool={'enabled' if len(answer.strip()) < 15 else 'disabled'}, Strict_length_filter=enabled"
            return prediction, False, f"No matches found across all strategies | Heuristics: {heuristic_info}"
    
    def stage3_llm_judge(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Stage 3: LLM equivalence checking.
        
        Uses the initialized LLM judge to determine semantic equivalence.
        For multiple choice questions (LiveXivVQA, VMCBench_DEV), provides choice context.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Optional dictionary mapping choice letters to their values (A->value, B->value, etc.)
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        return self.llm_judge.judge_equivalence(prediction, answer, choices_dict)
    
    # Unified extraction methods for Stage 2 with position tracking
    def _extract_latex_boxed_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract content from LaTeX \boxed{} format with positions."""
        matches = []
        
        # Method 1: Complex nested bracket parsing (most robust)
        for match in re.finditer(r'\\boxed{', prediction):
            start_index = match.end()
            end_index = start_index
            stack = 1
            
            while stack > 0 and end_index < len(prediction):
                if prediction[end_index] == '{':
                    stack += 1
                elif prediction[end_index] == '}':
                    stack -= 1
                end_index += 1
            
            if stack == 0:
                content = prediction[start_index:end_index - 1].strip()
                if content:
                    matches.append((content, match.start(), end_index))
        
        # Method 2: Simple regex fallback for basic cases (if no complex matches found)
        if not matches:
            for match in re.finditer(r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}', prediction):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_latex_boxed(self, prediction: str) -> Optional[str]:
        """Extract content from LaTeX \boxed{} format with nested bracket support."""
        # Method 1: Complex nested bracket parsing (most robust)
        boxed_matches = re.finditer(r'\\boxed{', prediction)
        results = []
        
        for match in boxed_matches:
            start_index = match.end()
            end_index = start_index
            stack = 1
            
            while stack > 0 and end_index < len(prediction):
                if prediction[end_index] == '{':
                    stack += 1
                elif prediction[end_index] == '}':
                    stack -= 1
                end_index += 1
            
            if stack == 0:
                content = prediction[start_index:end_index - 1]
                results.append(content.strip())
        
        if results:
            return results[-1]  # Return last (most recent) boxed content
        
        # Method 2: Simple regex fallback for basic cases
        simple_pattern = r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
        matches = re.findall(simple_pattern, prediction)
        if matches:
            return matches[-1].strip()
        
        # Backward compatibility - return last match
        matches = self._extract_latex_boxed_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_math_expressions_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract mathematical expressions with positions."""
        matches = []
        
        # Dollar-wrapped math: $expression$
        for match in re.finditer(r'\$([^$]+)\$', prediction):
            content = match.group(1).strip()
            if content:
                matches.append((content, match.start(), match.end()))
        
        # LaTeX expressions without \boxed (only if SymPy available)
        if SYMPY_AVAILABLE:
            try:
                from sympy.parsing.latex import parse_latex
                # Try to parse the whole prediction as LaTeX
                expr = parse_latex(prediction)
                expr_str = str(expr)
                if expr_str != prediction:
                    matches.append((expr_str, 0, len(prediction)))
            except Exception:
                pass
        
        return matches
    
    def _extract_math_expressions(self, prediction: str) -> Optional[str]:
        """Extract mathematical expressions from various formats."""
        # Dollar-wrapped math: $expression$
        dollar_pattern = r'\$([^$]+)\$'
        dollar_matches = re.findall(dollar_pattern, prediction)
        if dollar_matches:
            return dollar_matches[-1].strip()
        
        # LaTeX expressions without \boxed
        if SYMPY_AVAILABLE:
            try:
                from sympy.parsing.latex import parse_latex
                expr = parse_latex(prediction)
                return str(expr)
            except Exception:
                pass
        else:
            self.logger.warning("SymPy not available - LaTeX expression parsing disabled. Install with: pip install sympy")
        
        # Backward compatibility - return last match
        matches = self._extract_math_expressions_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_structured_tags_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract content from structured tags with positions."""
        matches = []
        
        # XML-style tags
        tag_patterns = [
            r'<ans>(.*?)</ans>',           # Omni3D format
            r'<answer>(.*?)</answer>',     # Generic answer tags
            r'<result>(.*?)</result>',     # Result tags
            r'<final>(.*?)</final>',       # Final answer tags
            r'\[ANSWER\](.*?)\[/ANSWER\]', # Bracket format
            r'\[ANS\](.*?)\[/ANS\]',       # Alternative bracket
        ]
        
        for pattern in tag_patterns:
            for match in re.finditer(pattern, prediction, re.IGNORECASE | re.DOTALL):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_structured_tags(self, prediction: str) -> Optional[str]:
        """Extract content from structured XML-style tags and brackets."""
        # XML-style tags
        tag_patterns = [
            r'<ans>(.*?)</ans>',           # Omni3D format
            r'<answer>(.*?)</answer>',     # Generic answer tags
            r'<result>(.*?)</result>',     # Result tags
            r'<final>(.*?)</final>',       # Final answer tags
            r'\[ANSWER\](.*?)\[/ANSWER\]', # Bracket format
            r'\[ANS\](.*?)\[/ANS\]',       # Alternative bracket
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Backward compatibility - return last match
        matches = self._extract_structured_tags_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_multiple_choice_with_positions(self, prediction: str, choices_dict: Optional[Dict[str, str]]) -> List[Tuple[str, int, int]]:
        """Extract multiple choice answers with positions (enhanced with Stage 1 logic)."""
        matches = []
        
        if not choices_dict:
            # Generic MCQ extraction without specific choices (enhanced with Stage 1 patterns)
            mcq_patterns = [
                r'\b([A-Z])\b',  # Isolated single letter (merged from Stage 1)
                r'(?:^|\s)([A-Z])(?:\s|$|\.|,)',  # Single letter with boundaries
                r'(?:is|are)\s+([A-Z])(?:\s|$|\.|,)',  # "The answer is A"
                r'(?:option|choice)\s+([A-Z])(?:\s|$|\.|,)',  # "Option A"
                r'([A-Z])\s*(?:is|are)\s*(?:correct|right)',  # "A is correct"
                r'\(([A-Z])\)',  # (A)
                r'([A-Z])\.',    # A.
                r'^([A-Z])\.\s',  # B. (at start of string/line)
                r'\n([A-Z])\.\s',  # B. (at start of new line)
            ]
            
            for pattern in mcq_patterns:
                for match in re.finditer(pattern, prediction, re.IGNORECASE):
                    content = match.group(1).upper().strip()
                    if content:
                        matches.append((content, match.start(), match.end()))
        else:
            # Specific choice-based extraction
            response = str(prediction)
            all_choices = list(choices_dict.keys())
            
            # Pattern 1: Bracketed options (A), (B) and formatted choices A., B. 
            for choice in all_choices:
                patterns = [
                    f'\\({choice}\\)',      # (A)
                    f'{choice}\\.\\s',      # A. (anywhere)
                    f'^{choice}\\.\\s',     # A. (at start of string)
                    f'\\n{choice}\\.\\s',    # A. (at start of line)
                ]
                for pattern in patterns:
                    for match in re.finditer(pattern, response):
                        matches.append((choice, match.start(), match.end()))
            
            # Pattern 2: Standalone letters
            for choice in all_choices:
                for match in re.finditer(f'\\s{choice}\\s', response):
                    matches.append((choice, match.start(), match.end()))
            
            # Pattern 3: Content matching (assign position as end of prediction)
            for choice, choice_text in choices_dict.items():
                if choice_text.lower() in response.lower():
                    pos = response.lower().find(choice_text.lower())
                    if pos >= 0:
                        matches.append((choice, pos, pos + len(choice_text)))
        
        return matches
    
    def _extract_multiple_choice(self, prediction: str, choices_dict: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract multiple choice answers (A, B, C, D, etc.)."""
        if not choices_dict:
            # Generic MCQ extraction without specific choices
            mcq_patterns = [
                r'(?:^|\s)([A-Z])(?:\s|$|\.|,)',  # Single letter
                r'(?:is|are)\s+([A-Z])(?:\s|$|\.|,)',  # "The answer is A"
                r'(?:option|choice)\s+([A-Z])(?:\s|$|\.|,)',  # "Option A"
                r'([A-Z])\s*(?:is|are)\s*(?:correct|right)',  # "A is correct"
                r'\(([A-Z])\)',  # (A)
                r'([A-Z])\.',    # A.
            ]
            
            for pattern in mcq_patterns:
                matches = re.findall(pattern, prediction, re.IGNORECASE)
                if matches:
                    return matches[0].upper().strip()
            return None
        
        # Specific choice-based extraction
        response = str(prediction)
        all_choices = list(choices_dict.keys())
        
        # Clean response
        for char in [',', '.', '!', '?', ';', ':', "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
        
        candidates = []
        
        # Pattern 1: Bracketed options (A), (B) or A., B.
        for choice in all_choices:
            if f'({choice})' in response or f'{choice}. ' in response:
                candidates.append(choice)
        
        # Pattern 2: Standalone letters " A ", " B "
        if len(candidates) == 0:
            for choice in all_choices:
                if f' {choice} ' in response:
                    candidates.append(choice)
        
        # Pattern 3: Content matching
        if len(candidates) == 0 and len(response.split()) > 5:
            for choice, choice_text in choices_dict.items():
                if choice_text.lower() in response.lower():
                    candidates.append(choice)
        
        # Backward compatibility - return last match
        matches = self._extract_multiple_choice_with_positions(prediction, choices_dict)
        return matches[-1][0] if matches else None
    
    def _extract_natural_language_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract answers from natural language patterns with positions."""
        matches = []
        
        # Common answer introduction patterns
        patterns = [
            r'So the final answer is\s*([^.\n]+)',
            r'Therefore,?\s*the answer is\s*([^.\n]+)',
            r'The answer is\s*([^.\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Final Answer:\s*([^\n]+)',
            r'Solution:\s*([^\n]+)',
            r'Result:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, prediction, re.IGNORECASE):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_natural_language_answers(self, prediction: str) -> Optional[str]:
        """Extract answers from natural language patterns."""
        # Common answer introduction patterns
        patterns = [
            r'So the final answer is\s*([^.\n]+)',
            r'Therefore,?\s*the answer is\s*([^.\n]+)',
            r'The answer is\s*([^.\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Final Answer:\s*([^\n]+)',
            r'Solution:\s*([^\n]+)',
            r'Result:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Backward compatibility - return last match
        matches = self._extract_natural_language_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_boolean_answers_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract boolean answers with positions (enhanced with Stage 1 logic)."""
        matches = []
        pred_lower = prediction.lower()
        
        # Enhanced patterns (merged from Stage 1)
        true_patterns = ['yes', 'true', 'correct', 'right']
        false_patterns = ['no', 'false', 'incorrect', 'wrong']
        
        # Find all true/yes patterns
        for pattern in true_patterns:
            for match in re.finditer(r'\b' + pattern + r'\b', pred_lower):
                matches.append(('yes', match.start(), match.end()))
        
        # Find all false/no patterns  
        for pattern in false_patterns:
            for match in re.finditer(r'\b' + pattern + r'\b', pred_lower):
                matches.append(('no', match.start(), match.end()))
        
        return matches
    
    def _extract_boolean_answers(self, prediction: str) -> Optional[str]:
        """Extract boolean/yes-no style answers."""
        pred_lower = prediction.lower()
        
        # Yes patterns
        yes_patterns = ['yes', 'true', 'correct', 'right']
        no_patterns = ['no', 'false', 'incorrect', 'wrong']
        
        # Count occurrences to handle cases like "not true"
        yes_count = sum(1 for pattern in yes_patterns if pattern in pred_lower)
        no_count = sum(1 for pattern in no_patterns if pattern in pred_lower)
        
        if yes_count > no_count and yes_count > 0:
            return 'yes'
        elif no_count > yes_count and no_count > 0:
            return 'no'
        
        # Backward compatibility - return last match
        matches = self._extract_boolean_answers_with_positions(prediction)
        if not matches:
            return None
        
        # Count yes vs no to determine final answer
        yes_count = sum(1 for content, _, _ in matches if content == 'yes')
        no_count = sum(1 for content, _, _ in matches if content == 'no')
        
        if yes_count > no_count:
            return 'yes'
        elif no_count > yes_count:
            return 'no'
        else:
            return matches[-1][0]  # If tied, return last occurrence
    
    def _extract_numeric_answers_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract numeric answers with positions."""
        matches = []
        
        # Look for numbers (decimals first for specificity)
        number_patterns = [
            r'-?\d+\.\d+',  # Decimals
            r'-?\d+',       # Integers
        ]
        
        for pattern in number_patterns:
            for match in re.finditer(pattern, prediction):
                content = match.group(0)
                matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_numeric_answers(self, prediction: str) -> Optional[str]:
        """Extract numeric answers from text."""
        # Look for numbers (integers and decimals)
        number_patterns = [
            r'-?\d+\.\d+',  # Decimals first (more specific)
            r'-?\d+',       # Integers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, prediction)
            if matches:
                return matches[-1]  # Return last number found
        
        # Backward compatibility - return last match
        matches = self._extract_numeric_answers_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _check_mathematical_equivalence_with_positions(self, prediction: str, answer: str) -> List[Tuple[str, int, int]]:
        """Check mathematical equivalence with position info."""
        if not SYMPY_AVAILABLE:
            return []
            
        try:
            # Try to parse both as mathematical expressions
            pred_expr = sympify(prediction)
            answer_expr = sympify(answer)
            
            # Check if expressions are equivalent
            diff = simplify(pred_expr - answer_expr)
            if diff == 0:
                # Return the answer as equivalent (position spans whole prediction)
                return [(answer, 0, len(prediction))]
                
        except Exception:
            pass
        
        return []
    
    def _check_mathematical_equivalence(self, prediction: str, answer: str) -> Optional[str]:
        """Check mathematical equivalence using SymPy."""
        if not SYMPY_AVAILABLE:
            self.logger.warning("SymPy not available - mathematical equivalence checking disabled. Install with: pip install sympy")
            return None
            
        try:
            # Try to parse both as mathematical expressions
            pred_expr = sympify(prediction)
            answer_expr = sympify(answer)
            
            # Check if expressions are equivalent
            diff = simplify(pred_expr - answer_expr)
            if diff == 0:
                return answer
                
        except Exception:
            pass
        
        return None

    def _build_empty_row_result(self) -> Dict[str, Any]:
        """Create the default result structure for a single row."""
        return {
            'stage1_match': 0,
            'stage2_match': 0,
            'stage3_match': 0,
            'stage4_match': 0,
            'final_answer': '',
            'hit': 0,
            'stage_errors': ''
        }

    def _finalize_row_result(
        self,
        row: pd.Series,
        row_result: Dict[str, Any],
        errors: List[str],
        final_answer: Optional[str],
        stage1_success: bool,
        stage2_success: bool,
        stage3_outcome: Optional[Tuple[str, bool, str]] = None,
    ) -> Dict[str, Any]:
        """Finalize row outputs after Stage 3 has either run or been skipped."""
        answer = str(row['answer'])
        stage3_success = False

        if stage3_outcome is not None:
            stage3_result, stage3_success, stage3_error = stage3_outcome
            if stage3_success:
                row_result['stage3_match'] = 1
                final_answer = stage3_result
            errors.append(f"Stage3: {stage3_error}")
        else:
            errors.append("Stage3: Skipped - Earlier stage succeeded")

        if not (stage1_success or stage2_success or stage3_success):
            final_answer = "NOMATCH"
            row_result['stage4_match'] = 1
            errors.append("Stage4: Fallback - NOMATCH")
        else:
            errors.append("Stage4: Not needed")

        row_result['final_answer'] = self._sanitize_text(final_answer or "NOMATCH")

        if 'answer_type' in row and row['answer_type'] == 'float':
            score_result = self._calculate_mra_score(final_answer, answer)
            row_result['hit'] = score_result['score']
            row_result['mra_details'] = self._sanitize_text(score_result.get('mra_details', ''))
        else:
            row_result['hit'] = 1 if final_answer == answer else 0

        row_result['stage_errors'] = self._sanitize_text(" | ".join(errors))
        return row_result

    def _process_row_until_stage2(self, row: pd.Series) -> Dict[str, Any]:
        """Run Stage 1 and Stage 2 for a single row and defer Stage 3 if needed."""
        row = row.copy()
        prediction = str(row['prediction'])
        answer = str(row['answer'])

        choices_dict = self._extract_choices(row)
        row_result = self._build_empty_row_result()

        errors = []
        final_answer = None
        try:
            stage1_result, stage1_success, stage1_error = self.stage1_simple_match(prediction, answer)
            if stage1_success:
                row_result['stage1_match'] = 1
                final_answer = stage1_result
            errors.append(f"Stage1: {stage1_error}")
        except Exception as e:
            errors.append(f"Stage1: ERROR - {str(e)}")
            stage1_success = False

        if not stage1_success:
            try:
                stage2_result, stage2_success, stage2_error = self.stage2_complex_match(
                    prediction, answer, choices_dict
                )
                if stage2_success:
                    row_result['stage2_match'] = 1
                    final_answer = stage2_result
                errors.append(f"Stage2: {stage2_error}")
            except Exception as e:
                errors.append(f"Stage2: ERROR - {str(e)}")
                stage2_success = False
        else:
            errors.append("Stage2: Skipped - Stage 1 succeeded")
            stage2_success = False

        needs_stage3 = not (stage1_success or stage2_success)
        return {
            'row': row,
            'prediction': prediction,
            'answer': answer,
            'choices_dict': choices_dict,
            'row_result': row_result,
            'errors': errors,
            'final_answer': final_answer,
            'stage1_success': stage1_success,
            'stage2_success': stage2_success,
            'needs_stage3': needs_stage3,
        }

    def _process_single_row(self, row: pd.Series) -> Dict[str, Any]:
        """Run the four-stage pipeline for a single row."""
        partial = self._process_row_until_stage2(row)
        stage3_outcome: Optional[Tuple[str, bool, str]] = None

        if partial['needs_stage3']:
            try:
                stage3_outcome = self.stage3_llm_judge(
                    partial['prediction'],
                    partial['answer'],
                    partial['choices_dict'],
                )
            except Exception as e:
                stage3_outcome = (
                    partial['prediction'],
                    False,
                    f"Stage 3 judge error: {str(e)}",
                )

        return self._finalize_row_result(
            row=partial['row'],
            row_result=partial['row_result'],
            errors=partial['errors'],
            final_answer=partial['final_answer'],
            stage1_success=partial['stage1_success'],
            stage2_success=partial['stage2_success'],
            stage3_outcome=stage3_outcome,
        )

    def _apply_four_stage_pipeline_with_batched_qwen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the pipeline while batching Stage 3 requests through the Qwen judge."""
        total_rows = len(df)
        worker_count = max(1, min(self.num_workers, total_rows))

        self.logger.info(
            "Applying 4-stage pipeline with batched Qwen Stage 3 using %d worker(s) for %d rows...",
            worker_count,
            total_rows,
        )

        if worker_count == 1:
            row_source = (row for _, row in df.iterrows())
        else:
            row_source = [row for _, row in df.iterrows()]

        executor = None
        if worker_count == 1:
            partial_iter = (self._process_row_until_stage2(row) for row in row_source)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            partial_iter = executor.map(self._process_row_until_stage2, row_source)

        partial_results: List[Dict[str, Any]] = []
        try:
            for partial in tqdm(
                partial_iter,
                total=total_rows,
                desc="Processing rows (Stages 1-2)",
            ):
                partial_results.append(partial)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        batch_size = getattr(self.llm_judge, "batch_size", 32)
        finalized_results: List[Optional[Dict[str, Any]]] = [None] * total_rows
        pending_indices: List[int] = []
        pending_requests: List[Tuple[str, str, Optional[Dict[str, str]]]] = []

        def flush_pending_batch():
            nonlocal pending_indices, pending_requests
            if not pending_requests:
                return

            try:
                outcomes = self.llm_judge.judge_equivalence_batch(pending_requests)
            except Exception as e:
                outcomes = [
                    (prediction, False, f"Qwen batch judge error: {str(e)}")
                    for prediction, _, _ in pending_requests
                ]

            for idx, outcome in zip(pending_indices, outcomes):
                partial = partial_results[idx]
                finalized_results[idx] = self._finalize_row_result(
                    row=partial['row'],
                    row_result=partial['row_result'],
                    errors=partial['errors'],
                    final_answer=partial['final_answer'],
                    stage1_success=partial['stage1_success'],
                    stage2_success=partial['stage2_success'],
                    stage3_outcome=outcome,
                )

            pending_indices = []
            pending_requests = []

        finalized_count = 0
        for idx, partial in enumerate(
            tqdm(partial_results, total=total_rows, desc="Processing rows (Stage 3 batch)")
        ):
            if partial['needs_stage3']:
                pending_indices.append(idx)
                pending_requests.append(
                    (
                        partial['prediction'],
                        partial['answer'],
                        partial['choices_dict'],
                    )
                )
                if len(pending_requests) >= batch_size:
                    flush_pending_batch()
            else:
                finalized_results[idx] = self._finalize_row_result(
                    row=partial['row'],
                    row_result=partial['row_result'],
                    errors=partial['errors'],
                    final_answer=partial['final_answer'],
                    stage1_success=partial['stage1_success'],
                    stage2_success=partial['stage2_success'],
                    stage3_outcome=None,
                )

            while finalized_count < total_rows and finalized_results[finalized_count] is not None:
                finalized_count += 1
                if finalized_count % 100 == 0:
                    ready_results = [res for res in finalized_results[:finalized_count] if res is not None]
                    self._save_intermediate_results(df.iloc[:finalized_count], ready_results, finalized_count)

        flush_pending_batch()

        results = [res for res in finalized_results if res is not None]
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)

    def apply_four_stage_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the 4-stage answer matching pipeline to all rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional scoring columns
        """
        total_rows = len(df)
        if total_rows == 0:
            return df

        if isinstance(self.llm_judge, QwenJudge):
            return self._apply_four_stage_pipeline_with_batched_qwen(df)

        worker_count = max(1, min(self.num_workers, total_rows))

        self.logger.info(
            "Applying 4-stage pipeline with %d worker(s) for %d rows...",
            worker_count,
            total_rows,
        )

        results: List[Dict[str, Any]] = []
        if worker_count == 1:
            row_source = (row for _, row in df.iterrows())
        else:
            # materialize rows so each worker receives an independent copy
            row_source = [row for _, row in df.iterrows()]

        executor = None
        if worker_count == 1:
            row_iter = (self._process_single_row(row) for row in row_source)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            row_iter = executor.map(self._process_single_row, row_source)

        processed_count = 0
        try:
            for row_result in tqdm(
                row_iter,
                total=total_rows,
                desc="Processing rows",
            ):
                results.append(row_result)
                processed_count += 1
                if processed_count % 100 == 0:
                    self._save_intermediate_results(df, results, processed_count)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        
        # Combine original data with results
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)
    
    def _calculate_mra_score(self, predicted_answer: str, ground_truth_answer: str) -> dict:
        """
        Calculate Mean Relative Accuracy (MRA) score for float answer types.
        
        MRA methodology from VSI-Bench paper (https://arxiv.org/abs/2412.14171):
        - Tests prediction accuracy at multiple relative error thresholds
        - MRA = average accuracy across all thresholds
        - Formula: |ground_truth - prediction| / ground_truth < threshold
        
        Args:
            predicted_answer: The predicted answer string
            ground_truth_answer: The ground truth answer string
            
        Returns:
            Dictionary containing MRA score and details
        """
        # MRA thresholds from Omni3DBench implementation
        mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        
        try:
            # Convert to float values
            pred_float = float(predicted_answer)
            gt_float = float(ground_truth_answer)
            
            # Handle division by zero
            if gt_float == 0:
                if pred_float == 0:
                    # Both are zero - perfect match
                    return {
                        'score': 1.0,
                        'mra_details': 'Perfect match: both values are zero',
                        'threshold_scores': {str(t): 1.0 for t in mra_thresholds}
                    }
                else:
                    # Ground truth is zero but prediction is not - no match
                    return {
                        'score': 0.0,
                        'mra_details': 'No match: ground truth is zero but prediction is not',
                        'threshold_scores': {str(t): 0.0 for t in mra_thresholds}
                    }
            
            # Calculate relative error: |gt - pred| / gt
            relative_error = abs(gt_float - pred_float) / abs(gt_float)
            
            # Test each threshold
            threshold_scores = {}
            for threshold in mra_thresholds:
                if relative_error < threshold:
                    threshold_scores[str(threshold)] = 1.0
                else:
                    threshold_scores[str(threshold)] = 0.0
            
            # Calculate MRA as average across all thresholds
            mra_score = sum(threshold_scores.values()) / len(mra_thresholds)
            
            return {
                'score': mra_score,
                'mra_details': f'Relative_error={relative_error:.4f}, MRA={mra_score:.3f}',
                'threshold_scores': threshold_scores,
                'relative_error': relative_error
            }
            
        except (ValueError, TypeError) as e:
            # Could not convert to float - return zero score
            return {
                'score': 0.0,
                'mra_details': f'Type conversion error: {e}',
                'threshold_scores': {str(t): 0.0 for t in mra_thresholds}
            }
    
    def _print_summary_stats(self, df: pd.DataFrame, benchmark_name: str):
        """Print summary statistics for the scored benchmark."""
        total_rows = len(df)
        hits = df['hit'].sum()
        accuracy = hits / total_rows if total_rows > 0 else 0
        
        stage_stats = {
            'Stage 1 (Simple)': df['stage1_match'].sum(),
            'Stage 2 (Complex)': df['stage2_match'].sum(), 
            'Stage 3 (LLM)': df['stage3_match'].sum(),
            'Stage 4 (Fallback)': df['stage4_match'].sum()
        }
        
        print(f"\n=== {benchmark_name} Summary ===")
        print(f"Total rows: {total_rows}")
        print(f"Score total: {hits:.3f}")  # Changed to show fractional scores for MRA
        print(f"Average score: {accuracy:.3f}")
        
        # Check if we have MRA scoring (float answer types)
        if 'answer_type' in df.columns:
            float_rows = df[df['answer_type'] == 'float']
            if len(float_rows) > 0:
                float_score = float_rows['hit'].sum()
                float_avg = float_score / len(float_rows)
                print(f"MRA scoring (float types): {len(float_rows)} rows, avg score: {float_avg:.3f}")
                
                # Show breakdown by answer type
                for answer_type in df['answer_type'].unique():
                    type_rows = df[df['answer_type'] == answer_type]
                    type_score = type_rows['hit'].sum()
                    type_avg = type_score / len(type_rows) if len(type_rows) > 0 else 0
                    print(f"  {answer_type} type: {len(type_rows)} rows, avg score: {type_avg:.3f}")
        
        print("\nStage success counts:")
        for stage, count in stage_stats.items():
            percentage = count / total_rows * 100 if total_rows > 0 else 0
            print(f"  {stage}: {count} ({percentage:.1f}%)")
    
    def _save_intermediate_results(self, df: pd.DataFrame, results: List[Dict], processed_count: int):
        """
        Save intermediate results to a temporary file every 100 rows.
        
        Args:
            df: Original DataFrame
            results: List of processed results so far
            processed_count: Number of rows processed
        """
        try:
            # Create a partial results DataFrame
            partial_df = df.iloc[:processed_count].copy()
            partial_results_df = pd.DataFrame(results)
            combined_df = pd.concat([partial_df, partial_results_df], axis=1)
            combined_df = self._sanitize_dataframe(combined_df)
            
            # Save to temporary file with processed count in filename
            temp_filename = f"intermediate_results_{processed_count}_rows.xlsx"
            temp_path = self.output_dir / temp_filename
            
            combined_df.to_excel(temp_path, index=False)
            
            # Calculate intermediate statistics
            hits = combined_df['hit'].sum() if 'hit' in combined_df.columns else 0
            accuracy = hits / processed_count if processed_count > 0 else 0
            
            self.logger.info(f"Saved intermediate results: {processed_count} rows processed, {hits} hits, {accuracy:.3f} accuracy -> {temp_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")
    
    def _cleanup_intermediate_files(self):
        """Clean up intermediate result files after processing is complete."""
        try:
            # Find all intermediate result files
            intermediate_files = list(self.output_dir.glob("intermediate_results_*_rows.xlsx"))
            
            if intermediate_files:
                self.logger.info(f"Cleaning up {len(intermediate_files)} intermediate files...")
                for file_path in intermediate_files:
                    try:
                        file_path.unlink()
                        self.logger.debug(f"Deleted intermediate file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting intermediate file {file_path}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error during intermediate file cleanup: {e}")
    
    def _save_summary_to_file(self, summary_data: dict):
        """Save summary statistics to a JSON file, merging with existing data if present."""
        summary_path = self.output_dir / "dcvlr_scoring_summary.json"
        
        try:
            # Load existing JSON if it exists
            existing_data = {}
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    self.logger.info(f"Loaded existing summary data with {len(existing_data.get('benchmark_results', {}))} benchmarks")
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Could not read existing summary file: {e}. Creating new file.")
                    existing_data = {}
            
            # Merge/update the data
            # Update metadata with current run info
            existing_data.update({
                'last_updated': summary_data['timestamp'],
                'total_runs': existing_data.get('total_runs', 0) + 1,
                'llm_backend': summary_data['llm_backend'],
                'resume_mode': summary_data['resume_mode'],
                'max_samples': summary_data['max_samples']
            })
            
            # Initialize benchmark_results if not present
            if 'benchmark_results' not in existing_data:
                existing_data['benchmark_results'] = {}
            
            # Update benchmark results (overwrite duplicates)
            for benchmark_name, benchmark_data in summary_data['benchmark_results'].items():
                existing_data['benchmark_results'][benchmark_name] = benchmark_data
                self.logger.info(f"Updated results for benchmark: {benchmark_name}")
            
            # Recalculate overall statistics from all benchmarks
            if existing_data['benchmark_results']:
                total_samples = sum(data['total_samples'] for data in existing_data['benchmark_results'].values())
                total_hits = sum(data['hits'] for data in existing_data['benchmark_results'].values())
                overall_accuracy = total_hits / total_samples if total_samples > 0 else 0
                
                existing_data['overall_statistics'] = {
                    'total_benchmarks': len(existing_data['benchmark_results']),
                    'total_samples': total_samples,
                    'total_hits': total_hits,
                    'overall_accuracy': overall_accuracy
                }
            
            # Save updated JSON
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved summary statistics to: {summary_path}")
            self.logger.info(f"Summary now contains {len(existing_data['benchmark_results'])} benchmarks")
            
        except Exception as e:
            self.logger.error(f"Error saving summary file: {e}")
    
    def process_all(self):
        """Process all benchmarks."""
        found_files = self.find_benchmark_files()
        
        if not found_files:
            self.logger.error("No benchmark files found!")
            return
        
        # Track overall statistics across all benchmarks
        all_benchmark_stats = []
        
        # Prepare summary data structure
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        summary_data = {
            'timestamp': timestamp,
            'benchmarks_processed': list(self.benchmarks),
            'llm_backend': self.llm_judge.__class__.__name__ if hasattr(self, 'llm_judge') else 'N/A',
            'resume_mode': self.resume,
            'max_samples': self.max_samples,
            'benchmark_results': {}
        }
        
        for benchmark_name, file_path in found_files.items():
            try:
                # Load the results to get final statistics
                output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
                
                self.process_benchmark(benchmark_name, file_path)
                
                # Read the results file to get final stats
                if output_path.exists():
                    results_df = pd.read_excel(output_path)
                    total_rows = len(results_df)
                    hits = results_df['hit'].sum() if 'hit' in results_df.columns else 0
                    accuracy = hits / total_rows if total_rows > 0 else 0
                    
                    # Prepare benchmark data for JSON
                    benchmark_data = {
                        'total_samples': total_rows,
                        'hits': float(hits),  # Ensure JSON serializable
                        'accuracy': float(accuracy),
                        'last_processed': timestamp,
                        'file_path': str(file_path)
                    }
                    
                    # Add stage breakdown if available
                    stage_cols = ['stage1_match', 'stage2_match', 'stage3_match', 'stage4_match']
                    if all(col in results_df.columns for col in stage_cols):
                        stage_stats = {
                            'stage1_simple': int(results_df['stage1_match'].sum()),
                            'stage2_complex': int(results_df['stage2_match'].sum()),
                            'stage3_llm': int(results_df['stage3_match'].sum()),
                            'stage4_fallback': int(results_df['stage4_match'].sum())
                        }
                        benchmark_data['stage_breakdown'] = stage_stats
                    
                    # Add answer type breakdown if available
                    if 'answer_type' in results_df.columns:
                        type_breakdown = {}
                        for answer_type in results_df['answer_type'].unique():
                            type_rows = results_df[results_df['answer_type'] == answer_type]
                            type_score = type_rows['hit'].sum()
                            type_avg = type_score / len(type_rows) if len(type_rows) > 0 else 0
                            type_breakdown[str(answer_type)] = {
                                'count': len(type_rows),
                                'score': float(type_score),
                                'accuracy': float(type_avg)
                            }
                        benchmark_data['answer_type_breakdown'] = type_breakdown
                    
                    # Store in summary data
                    summary_data['benchmark_results'][benchmark_name] = benchmark_data
                    
                    # Store stats for overall summary (for backwards compatibility)
                    all_benchmark_stats.append({
                        'benchmark': benchmark_name,
                        'total': total_rows,
                        'hits': hits,
                        'accuracy': accuracy
                    })
                
            except Exception as e:
                self.logger.error(f"Error processing {benchmark_name}: {e}")
                # Store error in JSON as well
                summary_data['benchmark_results'][benchmark_name] = {
                    'error': str(e),
                    'last_processed': timestamp,
                    'status': 'failed'
                }
                continue
        
        # Calculate and display overall statistics
        if all_benchmark_stats:
            total_samples = sum(stat['total'] for stat in all_benchmark_stats)
            total_hits = sum(stat['hits'] for stat in all_benchmark_stats)
            overall_accuracy = total_hits / total_samples if total_samples > 0 else 0
            
            # Add overall statistics to JSON
            summary_data['overall_statistics'] = {
                'total_benchmarks': len(all_benchmark_stats),
                'total_samples': total_samples,
                'total_hits': total_hits,
                'overall_accuracy': overall_accuracy
            }
            
            # Create console output
            overall_summary = [
                "=" * 60,
                "OVERALL SUMMARY",
                "=" * 60,
                f"Total benchmarks: {len(all_benchmark_stats)}",
                f"Total samples: {total_samples}",
                f"Total correct: {total_hits}",
                f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)",
                "",
                "Per-benchmark breakdown:"
            ]
            
            for stat in all_benchmark_stats:
                overall_summary.append(
                    f"  {stat['benchmark']}: {stat['hits']}/{stat['total']} = {stat['accuracy']:.3f} ({stat['accuracy']*100:.1f}%)"
                )
            
            # Print overall summary to console
            print("\n" + "\n".join(overall_summary))
        
        # Final completion message
        completion_msg = [
            "",
            "=" * 60,
            "PROCESSING COMPLETE",
            "=" * 60,
            f"Processed {len(found_files)} benchmarks",
            f"Output directory: {self.output_dir}",
            f"Summary saved to: {self.output_dir / 'dcvlr_scoring_summary.json'}"
        ]
        
        print("\n" + "\n".join(completion_msg))
        
        # Save summary to JSON file
        self._save_summary_to_file(summary_data)


class LLMEquivalenceJudge:
    """Base class for LLM equivalence judges."""
    
    SYSTEM_PROMPT = "You are an assistant that compares responses for semantic or mathematical equivalence."
    
    USER_PROMPT_TEMPLATE = """
Compare the following model response to the ground truth answer. Extract the model's actual answer from its response, and then determine if the extracted answer is semantically or mathematically equivalent to the ground truth. Condition your choice of equivalence checking on the contents of the prompt, whether mathematical or semantic. Model predictions may contain extensive internal chains of thought or may enclose the answer in special containers such as "/boxed{}", <ans></ans>, etc.

Response 1 (Model): {prediction}
Response 2 (Ground Truth): {answer}

Return a JSON response with this exact format:
{{
    "equivalent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Focus on semantic meaning rather than exact text matching.
Consider responses equivalent if they convey the same core meaning, even if wording differs.
"""

    USER_PROMPT_WITH_CHOICES_TEMPLATE = """
Compare the following model response to the ground truth answer for a multiple choice question. Extract the model's actual answer from its response, and then determine if the extracted answer is semantically or mathematically equivalent to the ground truth. The model may reference choice letters (A, B, C, D) or the actual choice values.

Multiple Choice Options:
{choices_context}

Response 1 (Model): {prediction}
Response 2 (Ground Truth): {answer} (which corresponds to the value: {correct_choice_value})

Return a JSON response with this exact format:
{{
    "equivalent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Consider responses equivalent if:
- The model selects the correct choice letter ({answer})
- The model provides the correct choice value ({correct_choice_value})
- The model's answer is semantically equivalent to the correct choice value
- The model's reasoning leads to the correct conclusion even if not explicitly stated

Focus on semantic meaning rather than exact text matching.
"""
    
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Judge semantic equivalence between prediction and answer.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Optional dictionary mapping choice letters to their values
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        raise NotImplementedError

    def _build_user_prompt(
        self,
        prediction: str,
        answer: str,
        choices_dict: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build the user prompt for a single equivalence request."""
        if choices_dict and answer in choices_dict:
            choices_context = "\n".join(
                [f"{letter}: {value}" for letter, value in choices_dict.items()]
            )
            correct_choice_value = choices_dict[answer]

            user_prompt = self.USER_PROMPT_WITH_CHOICES_TEMPLATE.replace(
                "{prediction}", str(prediction)
            )
            user_prompt = user_prompt.replace("{answer}", str(answer))
            user_prompt = user_prompt.replace("{choices_context}", choices_context)
            user_prompt = user_prompt.replace(
                "{correct_choice_value}", str(correct_choice_value)
            )
            return user_prompt

        user_prompt = self.USER_PROMPT_TEMPLATE.replace("{prediction}", str(prediction))
        user_prompt = user_prompt.replace("{answer}", str(answer))
        return user_prompt

    def _parse_judge_response(
        self,
        response_text: str,
        prediction: str,
        answer: str,
    ) -> Tuple[str, bool, str]:
        """Parse a judge response into the scorer's tuple format."""
        try:
            result = json.loads(response_text)
            equivalent = result.get('equivalent', False)
            reasoning = result.get('reasoning', 'No reasoning provided')

            if equivalent:
                return answer, True, f"LLM judge: equivalent - {reasoning}"
            return prediction, False, f"LLM judge: not equivalent - {reasoning}"

        except json.JSONDecodeError:
            if 'true' in response_text.lower():
                return answer, True, "LLM judge: equivalent (fallback parsing)"
            return prediction, False, "LLM judge: not equivalent (fallback parsing)"
    

class OpenAIJudge(LLMEquivalenceJudge):
    """OpenAI-based equivalence judge."""
    
    def __init__(self, model: str = 'gpt-4o-mini', api_key: Optional[str] = None):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """Judge equivalence using OpenAI API."""
        try:
            user_prompt = self._build_user_prompt(prediction, answer, choices_dict)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512,
                # Note: max_completion_tokens controls response length, input token limit is model-dependent
                # OpenAI models like gpt-4o-mini support up to 128k input tokens by default
            )
            
            response_text = response.choices[0].message.content or ""
            return self._parse_judge_response(response_text, prediction, answer)
        except Exception as e:
            return prediction, False, f"OpenAI API error: {str(e)}"


class AnthropicJudge(LLMEquivalenceJudge):
    """Anthropic-based equivalence judge."""
    
    def __init__(self, model: str = 'claude-3-sonnet-20240229', api_key: Optional[str] = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
        
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """Judge equivalence using Anthropic API."""
        try:
            user_prompt = self._build_user_prompt(prediction, answer, choices_dict)
            response = self.client.messages.create(
                model=self.model,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=512
                # Note: Anthropic models have generous input token limits (200k+ for Claude models)
                # Input token handling is automatic and doesn't require explicit configuration
            )
            
            response_text = response.content[0].text
            return self._parse_judge_response(response_text, prediction, answer)
        except Exception as e:
            return prediction, False, f"Anthropic API error: {str(e)}"


class QwenJudge(LLMEquivalenceJudge):
    """vLLM-backed local equivalence judge for Qwen3-4B."""

    SUPPORTED_MODEL = "qwen3-4b"
    MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, model: str = SUPPORTED_MODEL, batch_size: int = 32):
        if model != self.SUPPORTED_MODEL:
            raise ValueError(
                f"Unsupported qwen judge model: {model}. Only '{self.SUPPORTED_MODEL}' is supported."
            )
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM library not available. Install with: pip install vllm")

        try:
            import torch
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 1

        tp_size = max(1, min(gpu_count, 4))
        self.model = model
        self.batch_size = max(1, batch_size)
        self._lock = threading.Lock()
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None,
        )
        self._llm = LLM(
            model=self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            max_num_seqs=self.batch_size,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
        )

    def _format_chat_prompt(self, user_prompt: str) -> str:
        tokenizer = self._llm.get_tokenizer()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def judge_equivalence_batch(
        self,
        requests: List[Tuple[str, str, Optional[Dict[str, str]]]],
    ) -> List[Tuple[str, bool, str]]:
        prompts = [
            self._format_chat_prompt(
                self._build_user_prompt(prediction, answer, choices_dict)
            )
            for prediction, answer, choices_dict in requests
        ]

        with self._lock:
            outputs = self._llm.generate(prompts, self._sampling_params)

        results: List[Tuple[str, bool, str]] = []
        for (prediction, answer, _), output in zip(requests, outputs):
            response_text = output.outputs[0].text if output.outputs else ""
            results.append(self._parse_judge_response(response_text, prediction, answer))
        return results

    def judge_equivalence(
        self,
        prediction: str,
        answer: str,
        choices_dict: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, bool, str]:
        """Judge equivalence using a local Qwen3-4B vLLM instance."""
        try:
            return self.judge_equivalence_batch(
                [(prediction, answer, choices_dict)]
            )[0]
        except Exception as e:
            return prediction, False, f"Qwen judge error: {str(e)}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DCVLR Standalone Scorer - 4-stage answer matching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process VMCBench with OpenAI (full dataset)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV VMCBench_TEST \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini --verbose
      
  # Test with small sample (50 rows)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini \\
      --max-samples 50 --verbose
      
  # Resume from existing results (skip processed samples)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini \\
      --resume --verbose
      
  # Process with Anthropic backend
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend anthropic --model claude-3-sonnet-20240229

  # Process with local Qwen judge via vLLM
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend qwen --model qwen3-4b \\
      --qwen-judge-batch-size 64
        """
    )
    
    parser.add_argument(
        '--benchmarks', 
        nargs='+', 
        required=True,
        help='List of benchmark names to process (e.g., VMCBench_DEV VMCBench_TEST)'
    )
    parser.add_argument(
        '--input-dir', 
        required=True,
        help='Directory containing XLSX files'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for output files (defaults to input-dir)'
    )
    parser.add_argument(
        '--llm-backend',
        choices=['openai', 'anthropic', 'qwen'],
        default='openai',
        help='LLM backend for Stage 3 equivalence checking'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help="Model name for LLM judge (for --llm-backend qwen, only 'qwen3-4b' is supported)"
    )
    parser.add_argument(
        '--api-key',
        help='API key for LLM service (or use environment variables)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process per benchmark (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results file by skipping already processed samples'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of worker threads for the scoring pipeline (defaults to min(8, CPU cores))'
    )
    parser.add_argument(
        '--qwen-judge-batch-size',
        type=int,
        default=32,
        help='Batch size for qwen/vLLM Stage 3 judging (only used with --llm-backend qwen)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run scorer
    scorer = VLMEvalKitScorer(
        benchmarks=args.benchmarks,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm_backend=args.llm_backend,
        model=args.model,
        api_key=args.api_key,
        verbose=args.verbose,
        max_samples=args.max_samples,
        resume=args.resume,
        num_workers=args.num_workers,
        qwen_judge_batch_size=args.qwen_judge_batch_size,
    )
    
    scorer.process_all()


if __name__ == '__main__':
    main()
