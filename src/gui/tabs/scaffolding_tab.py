"""
Project Scaffolding Assistant GUI Tab Component (Milestone M5 Extension).

Provides a human-in-the-loop interface for generating, inspecting, and reviewing
AI-assisted RISC-V starter code with strict disk-write boundaries.

Enhancements (v1.1):
  - Complete 6-file project scaffold (models, requirements, train, tests, config, README)
  - Automated AST syntax validation badges shown before human review
  - Multi-turn conversational iteration (Regenerate with follow-up prompt)
  - Cumulative session cost tracker
  - Post-disk-write pipeline handoff button
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

from scaffolding.agent import ScaffoldingAgent
from scaffolding.config import ScaffoldingConfig


class ScaffoldingTab(ttk.Frame):
    """
    Experimental Project Scaffolding GUI Tab — v1.1.
    """

    def __init__(self, parent: ttk.Notebook, pipeline_callback: Optional[Any] = None) -> None:
        """
        Args:
            parent: Parent notebook widget.
            pipeline_callback: Optional callable(model_path) to load generated model
                               into the main TATVA optimization pipeline.
        """
        super().__init__(parent, padding=15)
        self.config = ScaffoldingConfig.load()
        self.agent = ScaffoldingAgent(self.config)
        self.pipeline_callback = pipeline_callback

        self.generated_data: Optional[Dict[str, Any]] = None
        self.is_generating = False
        self._last_disk_dir: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        # --- Header: BETA badge + disclaimer ---
        banner_frame = ttk.Frame(self, padding=5)
        banner_frame.pack(fill=tk.X, pady=(0, 10))

        lbl_badge = tk.Label(
            banner_frame,
            text=" EXPERIMENTAL / BETA ",
            font=("JetBrains Mono", 9, "bold"),
            bg="#F59E0B",
            fg="white",
            padx=6,
            pady=2,
        )
        lbl_badge.pack(side=tk.LEFT, padx=(0, 10))

        lbl_disclaimer = ttk.Label(
            banner_frame,
            text="⚠️ AI-generated code — review all files carefully before approving disk write.",
            font=("Inter", 9, "italic"),
            foreground="#b45309",
        )
        lbl_disclaimer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Cumulative cost label ---
        self.lbl_cumulative = ttk.Label(
            banner_frame,
            text="Session Total: $0.00000",
            font=("JetBrains Mono", 9, "bold"),
            foreground="#6B7280",
        )
        self.lbl_cumulative.pack(side=tk.RIGHT, padx=(0, 5))

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(self, text=" Project Specifications & Prompt ", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 8))

        lbl_prompt = ttk.Label(input_frame, text="Describe your RISC-V project requirements:")
        lbl_prompt.pack(anchor=tk.W, pady=(0, 4))

        self.txt_prompt = tk.Text(input_frame, height=4, font=("Consolas", 9), wrap=tk.WORD)
        self.txt_prompt.pack(fill=tk.X, pady=(0, 8))
        self.txt_prompt.insert(
            tk.END,
            "Keyword spotting classifier model targeting RV64GCV with dynamic 8-bit quantization and fast softmax.",
        )
        self.txt_prompt.bind("<KeyRelease>", self._update_cost_estimate)

        model_bar = ttk.Frame(input_frame)
        model_bar.pack(fill=tk.X)

        lbl_model = ttk.Label(model_bar, text="AI Backend:")
        lbl_model.pack(side=tk.LEFT, padx=(0, 5))

        self.cbo_model = ttk.Combobox(model_bar, values=self.config.models, state="readonly", width=28)
        self.cbo_model.pack(side=tk.LEFT, padx=(0, 15))
        self.cbo_model.set(self.config.default_model)
        self.cbo_model.bind("<<ComboboxSelected>>", self._update_cost_estimate)

        self.lbl_cost = ttk.Label(
            model_bar,
            text="Est. Cost: $0.00035",
            font=("JetBrains Mono", 9, "bold"),
            foreground="#047857",
        )
        self.lbl_cost.pack(side=tk.RIGHT)

        # --- Action Row 1: Generate + Status ---
        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, pady=(0, 5))

        self.btn_generate = ttk.Button(btn_bar, text="▶ Generate Project Scaffold", command=self._start_generation)
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_gen_status = ttk.Label(btn_bar, text="Status: Ready", font=("Inter", 9))
        self.lbl_gen_status.pack(side=tk.LEFT)

        # --- Multi-turn: Iterate Frame ---
        iter_frame = ttk.LabelFrame(self, text=" Multi-Turn Iteration (Follow-up Prompt) ", padding=8)
        iter_frame.pack(fill=tk.X, pady=(0, 8))

        self.txt_iterate = tk.Text(iter_frame, height=2, font=("Consolas", 9), wrap=tk.WORD, state=tk.DISABLED)
        self.txt_iterate.pack(fill=tk.X, pady=(0, 6))

        iter_btn_bar = ttk.Frame(iter_frame)
        iter_btn_bar.pack(fill=tk.X)

        self.btn_iterate = ttk.Button(
            iter_btn_bar,
            text="↺ Regenerate / Iterate",
            command=self._start_iteration,
            state=tk.DISABLED,
        )
        self.btn_iterate.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_reset = ttk.Button(
            iter_btn_bar,
            text="⟳ New Session",
            command=self._reset_session,
        )
        self.btn_reset.pack(side=tk.LEFT)

        lbl_iter_hint = ttk.Label(
            iter_btn_bar,
            text='e.g. "add data augmentation to train.py" or "change target to RV32IMC"',
            font=("Inter", 8, "italic"),
            foreground="#6B7280",
        )
        lbl_iter_hint.pack(side=tk.LEFT, padx=(12, 0))

        # --- Code Inspection Notebook (Safety Review) ---
        review_frame = ttk.LabelFrame(
            self,
            text=" Code Inspection & Safety Review — IN-MEMORY ONLY (not written to disk) ",
            padding=10,
        )
        review_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.code_notebook = ttk.Notebook(review_frame)
        self.code_notebook.pack(fill=tk.BOTH, expand=True)

        self.preview_text_widgets: Dict[str, tk.Text] = {}
        self._add_placeholder_tab("No code generated yet. Click '▶ Generate Project Scaffold' above.")

        # --- Action Row 2: Accept / Discard Gate ---
        gate_bar = ttk.Frame(self)
        gate_bar.pack(fill=tk.X)

        self.btn_accept = ttk.Button(
            gate_bar,
            text="✅ Review & Accept — Write to Disk",
            command=self._accept_and_write_to_disk,
            state=tk.DISABLED,
        )
        self.btn_accept.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_discard = ttk.Button(
            gate_bar,
            text="🗑 Discard / Cancel",
            command=self._discard_generated,
            state=tk.DISABLED,
        )
        self.btn_discard.pack(side=tk.LEFT, padx=(0, 20))

        self.btn_pipeline = ttk.Button(
            gate_bar,
            text="🚀 Run Through TATVA Pipeline",
            command=self._run_pipeline_handoff,
            state=tk.DISABLED,
        )
        self.btn_pipeline.pack(side=tk.LEFT)

        self.lbl_pipeline_hint = ttk.Label(
            gate_bar,
            text="",
            font=("Inter", 8, "italic"),
            foreground="#047857",
        )
        self.lbl_pipeline_hint.pack(side=tk.LEFT, padx=(8, 0))

        self._update_cost_estimate(None)

    # --- Utilities ---

    def _add_placeholder_tab(self, msg: str) -> None:
        for tab in self.code_notebook.tabs():
            self.code_notebook.forget(tab)
        self.preview_text_widgets.clear()

        frame = ttk.Frame(self.code_notebook, padding=10)
        self.code_notebook.add(frame, text=" Preview ")
        txt = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 9))
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, msg)

    def _update_cost_estimate(self, event: Any) -> None:
        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        model_name = self.cbo_model.get()
        cost = self.config.estimate_cost(prompt, model_name)
        char_count = len(prompt)
        self.lbl_cost.config(text=f"Est. Cost: ${cost:.5f} ({char_count} chars)")

    def _update_cumulative_display(self) -> None:
        self.lbl_cumulative.config(
            text=f"Session Total: ${self.agent.cumulative_cost_usd:.5f}"
        )

    def _set_buttons_generating(self, state: str) -> None:
        """Enable or disable action buttons during generation."""
        self.btn_generate.config(state=state)
        self.btn_iterate.config(state=state)
        self.btn_accept.config(state=tk.DISABLED)
        self.btn_discard.config(state=tk.DISABLED)
        self.txt_iterate.config(state=tk.DISABLED if state == tk.DISABLED else tk.NORMAL)

    # --- Generation Flow ---

    def _start_generation(self) -> None:
        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Missing Prompt", "Please enter a project description prompt.")
            return

        self.is_generating = True
        self._set_buttons_generating(tk.DISABLED)
        self.lbl_gen_status.config(text="Status: Generating 6-file project scaffold…", foreground="#3B82F6")

        model_name = self.cbo_model.get()
        threading.Thread(target=self._generation_worker, args=(prompt, model_name), daemon=True).start()

    def _start_iteration(self) -> None:
        follow_up = self.txt_iterate.get("1.0", tk.END).strip()
        if not follow_up:
            messagebox.showwarning("Missing Follow-up", "Enter a follow-up instruction to iterate.")
            return

        self.is_generating = True
        self._set_buttons_generating(tk.DISABLED)
        self.lbl_gen_status.config(text="Status: Iterating on previous scaffold…", foreground="#3B82F6")

        model_name = self.cbo_model.get()
        threading.Thread(target=self._iteration_worker, args=(follow_up, model_name), daemon=True).start()

    def _generation_worker(self, prompt: str, model_name: str) -> None:
        try:
            res = self.agent.generate(prompt, model_name)
            self.after(0, lambda r=res: self._on_generation_success(r))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._on_generation_error(msg))

    def _iteration_worker(self, follow_up: str, model_name: str) -> None:
        try:
            res = self.agent.iterate(follow_up, model_name)
            self.after(0, lambda r=res: self._on_generation_success(r))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._on_generation_error(msg))

    def _on_generation_success(self, data: Dict[str, Any]) -> None:
        self.is_generating = False
        self.generated_data = data

        # Re-enable buttons
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_iterate.config(state=tk.NORMAL)
        self.btn_accept.config(state=tk.NORMAL)
        self.btn_discard.config(state=tk.NORMAL)
        self.txt_iterate.config(state=tk.NORMAL)
        self.btn_pipeline.config(state=tk.DISABLED)

        # Count AST failures
        files = data.get("files", [])
        n_py = sum(1 for f in files if f["path"].endswith(".py"))
        n_pass = sum(
            1 for f in files
            if f["path"].endswith(".py") and f.get("ast_check", {}).get("passed", True)
        )
        n_fail = n_py - n_pass

        if n_fail == 0:
            self.lbl_gen_status.config(
                text=f"Status: ✅ Generated In-Memory — AST Syntax: {n_pass}/{n_py} PASSED — Pending Review",
                foreground="#047857",
            )
        else:
            self.lbl_gen_status.config(
                text=f"Status: ⚠️ Generated — AST Syntax: {n_pass}/{n_py} PASSED, {n_fail} FAILED — Review Before Accepting",
                foreground="#b45309",
            )

        self._update_cumulative_display()
        self._render_file_tabs(files)

    def _on_generation_error(self, err_msg: str) -> None:
        self.is_generating = False
        self.btn_generate.config(state=tk.NORMAL)
        self.txt_iterate.config(state=tk.NORMAL)
        self.lbl_gen_status.config(text="Status: [ERROR] Generation Failed", foreground="red")
        messagebox.showerror("Scaffolding Error", f"Failed to generate scaffolding:\n{err_msg}")

    def _render_file_tabs(self, files: List[Dict[str, Any]]) -> None:
        """Render file preview tabs with AST check status badge in tab title."""
        for tab in self.code_notebook.tabs():
            self.code_notebook.forget(tab)
        self.preview_text_widgets.clear()

        for file_info in files:
            path_str = file_info["path"]
            content = file_info["content"]
            ast_result = file_info.get("ast_check", {})
            ast_passed = ast_result.get("passed", True)
            ast_msg = ast_result.get("message", "N/A")

            # Tab title: show syntax badge if Python file
            if path_str.endswith(".py"):
                badge = "✅" if ast_passed else "❌"
                tab_title = f" {badge} {path_str} "
            else:
                tab_title = f" {path_str} "

            tab_frame = ttk.Frame(self.code_notebook, padding=5)
            self.code_notebook.add(tab_frame, text=tab_title)

            # AST check header bar inside tab
            if path_str.endswith(".py"):
                ast_color = "#047857" if ast_passed else "#b91c1c"
                ast_text = f"  AST Syntax Check: {ast_msg}  "
                lbl_ast = tk.Label(
                    tab_frame,
                    text=ast_text,
                    font=("JetBrains Mono", 8, "bold"),
                    bg=("#D1FAE5" if ast_passed else "#FEE2E2"),
                    fg=ast_color,
                    anchor=tk.W,
                    padx=4,
                    pady=2,
                )
                lbl_ast.pack(fill=tk.X, pady=(0, 4))

            txt = tk.Text(tab_frame, wrap=tk.WORD, font=("Consolas", 9))
            txt.pack(fill=tk.BOTH, expand=True)
            txt.insert(tk.END, content)
            self.preview_text_widgets[path_str] = txt

    # --- Accept / Discard / Reset ---

    def _accept_and_write_to_disk(self) -> None:
        if not self.generated_data or "files" not in self.generated_data:
            return

        # Warn if any Python file failed AST check
        failed_files = [
            f["path"]
            for f in self.generated_data["files"]
            if f["path"].endswith(".py") and not f.get("ast_check", {}).get("passed", True)
        ]
        if failed_files:
            proceed = messagebox.askyesno(
                "AST Syntax Warnings",
                f"The following files failed AST syntax check:\n  {chr(10).join(failed_files)}\n\n"
                "Are you sure you want to write these files to disk?",
            )
            if not proceed:
                return

        target_dir = filedialog.askdirectory(title="Select Destination Directory for Scaffolding")
        if not target_dir:
            return

        try:
            files_data = self.generated_data["files"]
            created = self.agent.write_to_disk(target_dir, files_data)
            self._last_disk_dir = target_dir

            self.agent.logger.log_generation_attempt(
                prompt_text=self.txt_prompt.get("1.0", tk.END).strip(),
                model_name=self.cbo_model.get(),
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=self.generated_data.get("estimated_cost_usd", 0.0),
                accepted_by_user=True,
            )

            messagebox.showinfo(
                "Scaffolding Created",
                f"✅ Successfully written {len(created)} file(s) to:\n{target_dir}\n\n"
                "Click '🚀 Run Through TATVA Pipeline' to optimize models/model.onnx.",
            )
            self.lbl_gen_status.config(text="Status: ✅ Written to Disk", foreground="#047857")

            # Enable pipeline handoff if model.onnx will exist
            onnx_path = f"{target_dir}/models/model.onnx"
            self.btn_pipeline.config(state=tk.NORMAL)
            self.lbl_pipeline_hint.config(
                text=f"→ tatva optimize {onnx_path} --target RV64GCV"
            )
        except Exception as e:
            messagebox.showerror("Disk Write Error", f"Failed to write scaffolding files:\n{e}")

    def _discard_generated(self) -> None:
        self.generated_data = None
        self.btn_accept.config(state=tk.DISABLED)
        self.btn_discard.config(state=tk.DISABLED)
        self.btn_pipeline.config(state=tk.DISABLED)
        self.lbl_pipeline_hint.config(text="")
        self.lbl_gen_status.config(text="Status: Discarded — Disk untouched", foreground="#6B7280")
        self._add_placeholder_tab("Generated code discarded. Disk remains untouched.")

    def _reset_session(self) -> None:
        """Start a fresh multi-turn session (clear chat history and cost tracker)."""
        self.agent.reset_session()
        self._discard_generated()
        self.txt_iterate.config(state=tk.NORMAL)
        self.txt_iterate.delete("1.0", tk.END)
        self.txt_iterate.config(state=tk.DISABLED)
        self._update_cumulative_display()
        self.lbl_gen_status.config(text="Status: New session started — history cleared", foreground="#6B7280")

    # --- Pipeline Handoff ---

    def _run_pipeline_handoff(self) -> None:
        """
        After disk write: feed models/model.onnx directly into the TATVA optimization pipeline.
        Either triggers the parent app callback (Tab 1 load) or shows CLI instructions.
        """
        if not self._last_disk_dir:
            messagebox.showinfo("Pipeline Handoff", "Please accept and write files to disk first.")
            return

        import os
        onnx_path = os.path.join(self._last_disk_dir, "models", "model.onnx")

        if self.pipeline_callback and os.path.exists(onnx_path):
            # Load generated model directly into Tab 1
            self.pipeline_callback(onnx_path)
            messagebox.showinfo(
                "Pipeline Handoff",
                f"✅ Model loaded into TATVA pipeline:\n{onnx_path}\n\n"
                "Switch to Tab 1 (Model Input) → Tab 2 → Tab 4 to run the full optimization.",
            )
        else:
            # First run train.py to generate model.onnx, then show CLI instructions
            if not os.path.exists(onnx_path):
                msg = (
                    f"models/model.onnx not found at:\n{onnx_path}\n\n"
                    "Run `python train.py` first to export the ONNX model, then click this button again.\n\n"
                    f"CLI Command:\n  cd {self._last_disk_dir}\n"
                    "  python train.py\n"
                    "  tatva optimize models/model.onnx --target RV64GCV"
                )
            else:
                msg = (
                    f"✅ One-click pipeline handoff.\n\n"
                    f"Run these commands to optimize your model:\n\n"
                    f"  cd {self._last_disk_dir}\n"
                    f"  tatva optimize models/model.onnx --target RV64GCV\n\n"
                    f"Or open Tab 1, browse to:\n  {onnx_path}"
                )
            messagebox.showinfo("Pipeline Handoff — CLI Instructions", msg)
