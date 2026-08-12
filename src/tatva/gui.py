"""
TATVA Standalone Desktop GUI Application — Modern Gemini / Claude Dark Theme.

Provides a unified conversational interface & autonomous closed-loop engineering engine
("Tatva Antigravity Engine") with:
  - Gemini / Claude Dark Slate theme styling (#131314 base bg, #1E1E20 cards, #282A2C borders, #A8C7FA pill tabs)
  - Subprocess toolchain execution runner (riscv64-gcc, QEMU system & user emulation)
  - Ollama local LLM integration + Cloud API fallback
  - Autonomous closed-loop self-correction engine (iterative GCC compile -> QEMU run -> parity verification -> auto-fix)
"""

import contextlib
import os
import queue
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox, ttk
from typing import Any

# ── Ensure src/ is in sys.path for robust module imports ──────────────────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Build Metadata ────────────────────────────────────────────────────────────
# Read from the package rather than restated here. A hardcoded string in this file
# is a string that goes stale the moment anyone bumps the real version.
from tatva import DISPLAY_VERSION, __version__  # noqa: E402

# The badge shows the friendly name ("Beta 2.0"); __version__ stays PEP 440 for
# packaging. Both come from tatva/__init__.py so there is still one source of truth.
BUILD_VERSION = DISPLAY_VERSION
BUILD_LABEL = f"{DISPLAY_VERSION} · {__version__}"
# ─────────────────────────────────────────────────────────────────────────────

# Deferred module containers (populated during splash screen)
_backend: dict[str, Any] = {}


def load_backend_libraries(callback: Any) -> None:
    """
    Deferred import worker that loads heavy backend libraries off the main UI thread.
    """
    try:
        from tatva.compiler import DEFAULT_TARGET, TARGETS, analyze_graph, import_model
        from tatva.diagnostics import classify_failure, explain
        from tatva.optimizer import compare_configs, quantize, select_fast_softmax_kernel
        from tatva.runner import (
            ExecutionEnvironment,
            compile_model,
            establish_baseline,
            run_and_measure,
            verify_target,
        )

        _backend["import_model"] = import_model
        _backend["analyze_graph"] = analyze_graph
        _backend["TARGETS"] = TARGETS
        _backend["DEFAULT_TARGET"] = DEFAULT_TARGET
        _backend["verify_target"] = verify_target
        _backend["establish_baseline"] = establish_baseline
        _backend["select_fast_softmax_kernel"] = select_fast_softmax_kernel
        _backend["quantize"] = quantize
        _backend["compare_configs"] = compare_configs
        _backend["compile_model"] = compile_model
        _backend["run_and_measure"] = run_and_measure
        _backend["ExecutionEnvironment"] = ExecutionEnvironment
        _backend["classify_failure"] = classify_failure
        _backend["explain"] = explain

        callback(True, "")
    except Exception as e:
        callback(False, str(e))


class SplashWindow(tk.Toplevel):
    """
    Futuristic Deep-Tech Splash Screen for TATVA RISC-V Optimization Studio.
    """

    def __init__(self, parent: tk.Tk, on_complete_callback: Any) -> None:
        super().__init__(parent)
        self.parent = parent
        self.on_complete_callback = on_complete_callback

        self.title("TATVA RISC-V Optimization Studio")
        self.geometry("580x360")
        self.resizable(False, False)
        self.overrideredirect(True)
        self.configure(bg="#131314")

        # Center splash window on screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

        self.canvas = tk.Canvas(self, width=580, height=220, bg="#131314", highlightthickness=0)
        self.canvas.pack(fill=tk.X, pady=(20, 0))

        self._draw_tatva_logo()

        self.lbl_status = tk.Label(
            self,
            text="Initializing deep-tech compiler pipeline & loading dependencies...",
            font=("JetBrains Mono", 8),
            fg="#9CA3AF",
            bg="#131314",
        )
        self.lbl_status.pack(pady=(5, 10))

        style = ttk.Style(self)
        style.configure(
            "GreenGlow.Horizontal.TProgressbar",
            troughcolor="#1E1E20",
            background="#EBB85F",   # brand gold, matching the web splash
            bordercolor="#282A2C",
            thickness=4,
        )

        self.progress = ttk.Progressbar(
            self, mode="indeterminate", length=400, style="GreenGlow.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=(0, 15))
        self.progress.start(12)

        lbl_copyright = tk.Label(
            self,
            text="© 2026 TATVA RISC-V Optimization Studio. All rights reserved.",
            font=("Inter", 8),
            fg="#6B7280",
            bg="#131314",
        )
        lbl_copyright.pack(side=tk.BOTTOM, pady=(0, 15))

        threading.Thread(target=self._run_loader, daemon=True).start()

    def _logo_path(self) -> str:
        """Locate the wordmark, whether running frozen, installed, or from a checkout."""
        candidates = [
            os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "logo-dark.png"),
            os.path.join(_CURRENT_DIR, "assets", "logo-dark.png"),
            os.path.join(_CURRENT_DIR, "..", "..", "assets", "logo-dark.png"),
            os.path.join(_CURRENT_DIR, "..", "assets", "logo-dark.png"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return ""

    def _draw_tatva_logo(self) -> None:
        """
        Show the real wordmark.

        This used to hand-draw five letterforms out of canvas line segments, in the old
        blue. That meant the splash carried a different logo from every other surface,
        and a brand change had to be made twice. Tk 8.6 reads PNG natively, so the
        shipped asset can just be displayed -- with the drawn version kept only as a
        fallback for the case where the asset is genuinely missing.
        """
        c = self.canvas
        c.delete("all")

        path = self._logo_path()
        if path:
            try:
                image = tk.PhotoImage(file=path)
                # Integer subsampling only -- Tk has no smooth scaler, and a non-integer
                # factor would alias the thin strokes in the letterforms.
                factor = max(1, round(image.width() / 320))
                if factor > 1:
                    image = image.subsample(factor, factor)
                # Held on the instance: Tk keeps only a weak reference and a local would
                # be collected before the canvas paints, leaving an empty box.
                self._logo_image = image
                c.create_image(290, 110, image=image)
                return
            except Exception:
                pass

        self._draw_tatva_logo_fallback()

    def _draw_tatva_logo_fallback(self) -> None:
        """Vector stand-in used only when the logo asset cannot be loaded."""
        c = self.canvas
        gold = "#EBB85F"
        cx, cy = 110, 110

        def letter_t(x: int) -> None:
            c.create_line(x - 20, cy - 25, x + 20, cy - 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)
            c.create_line(x, cy - 25, x, cy + 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)

        def letter_a(x: int) -> None:
            c.create_line(x - 22, cy + 25, x, cy - 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)
            c.create_line(x, cy - 25, x + 22, cy + 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)
            c.create_polygon(x - 8, cy + 18, x + 8, cy + 18, x, cy + 8, fill=gold, outline=gold)

        letter_t(cx)
        letter_a(cx + 70)
        letter_t(cx + 140)

        # V
        vx = cx + 210
        c.create_line(vx - 22, cy - 25, vx, cy + 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)
        c.create_line(vx, cy + 25, vx + 22, cy - 25, fill="#FFFFFF", width=3, capstyle=tk.ROUND)

        letter_a(cx + 280)

        # The comet above the final A.
        tx, ty = cx + 302, cy - 25
        c.create_line(tx, ty, tx + 60, ty, fill="#7A5A18", width=2)
        c.create_line(tx, ty, tx + 35, ty, fill=gold, width=2)
        c.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill="#F6D08A", outline=gold)

    def _run_loader(self) -> None:
        load_backend_libraries(self._on_loaded)

    def _on_loaded(self, success: bool, error_msg: str) -> None:
        def main_thread_update() -> None:
            self.progress.stop()
            self.destroy()
            self.on_complete_callback(success, error_msg)

        self.after(0, main_thread_update)


class TatvaApp(tk.Tk):
    """
    Main Application Window — Gemini / Claude Modern Dark Theme.
    Features:
      - Left Panel (60-65% width): Scrollable Chat Thread with inline interactive cards & Autonomous Engine
      - Right Panel (35-40% width): Collapsible Slide-Out Artifacts Panel for structured output & code viewer
    """

    def __init__(self, is_test_mode: bool = False) -> None:
        super().__init__()
        self.is_test_mode = is_test_mode
        self.title("TATVA — Bare-Metal RISC-V Transformer Optimization Studio")
        self.geometry("1380x880")
        self.minsize(1080, 720)

        self.model_path: str | None = None
        self.imported_ir: Any = None
        self.graph_stats: Any = None
        self.cancel_requested = False
        self.msg_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.scaffolding_agent = None
        self.scaffolding_files_data = []

        self._configure_styles()
        self._build_layout()
        self._init_scaffolding_agent()

    def _ui_call(self, fn: Callable[[], None]) -> None:
        """
        Run `fn` on the Tk main loop from a worker thread.

        Every background task here ends by touching widgets, and `after` raises
        RuntimeError("main thread is not in main loop") once the window has gone away --
        during shutdown, or in a test that constructs the app without calling mainloop.
        That exception surfaced as an unhandled thread crash rather than anything
        actionable, so swallow exactly that case and let everything else through.
        """
        with contextlib.suppress(RuntimeError, tk.TclError):
            self.after(0, fn)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        with contextlib.suppress(Exception):
            style.theme_use("clam")

        # Gemini / Claude Modern Dark Color Palette
        bg_dark = "#131314"
        bg_card = "#1E1E20"
        bg_elevated = "#2E3033"
        fg_white = "#E3E3E3"
        fg_muted = "#9AA0A6"
        accent_blue = "#A8C7FA"
        border_col = "#282A2C"

        self.configure(bg=bg_dark)

        style.configure(".", background=bg_dark, foreground=fg_white, font=("Inter", 10))
        style.configure("TFrame", background=bg_dark)
        style.configure(
            "TLabelframe", background=bg_card, foreground=fg_white, bordercolor=border_col, relief="solid"
        )
        style.configure(
            "TLabelframe.Label", background=bg_card, foreground=accent_blue, font=("Inter", 11, "bold")
        )
        style.configure("TLabel", background=bg_dark, foreground=fg_white)
        style.configure("Header.TLabel", font=("Inter", 15, "bold"), foreground="#FFFFFF")
        style.configure(
            "TButton",
            font=("Inter", 10, "bold"),
            background=bg_elevated,
            foreground=fg_white,
            bordercolor=border_col,
            padding=[10, 6],
        )
        style.map(
            "TButton",
            background=[("active", accent_blue)],
            foreground=[("active", "#040C18")],
        )

        style.configure(
            "Primary.TButton",
            font=("Inter", 10, "bold"),
            background=accent_blue,
            foreground="#040C18",
            bordercolor=accent_blue,
            padding=[12, 7],
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#C2E7FF")],
            foreground=[("active", "#040C18")],
        )

        style.configure("TNotebook", background=bg_dark, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=bg_card,
            foreground=fg_muted,
            padding=[14, 7],
            font=("Inter", 9, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", bg_elevated)],
            foreground=[("selected", accent_blue)],
        )

    def _build_layout(self) -> None:
        # Header banner
        header_frame = ttk.Frame(self, padding=8, relief=tk.RAISED)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        lbl_hdr = ttk.Label(
            header_frame,
            text="TATVA Optimization Studio",
            style="Header.TLabel",
        )
        lbl_hdr.pack(side=tk.LEFT, padx=(5, 0))

        lbl_build = ttk.Label(
            header_frame,
            text=BUILD_LABEL,
            font=("JetBrains Mono", 8),
            foreground="#9AA0A6",
        )
        lbl_build.pack(side=tk.LEFT, padx=(12, 0))

        btn_toggle_art = ttk.Button(
            header_frame,
            text="Toggle Artifacts Panel 📑",
            command=self._toggle_artifacts_panel,
        )
        btn_toggle_art.pack(side=tk.RIGHT, padx=5)

        self.lbl_badge = ttk.Label(
            header_frame,
            text="[QEMU SIMULATION]",
            font=("Helvetica", 9, "bold"),
            foreground="#A8C7FA",
        )
        self.lbl_badge.pack(side=tk.RIGHT, padx=10)

        # Paned Window (Left: Chat Thread ~62%, Right: Artifacts Panel ~38%)
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Panel (Chat Thread)
        self.left_frame = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.left_frame, weight=6)

        # Right Panel (Slide-out Artifacts Panel)
        self.right_frame = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.right_frame, weight=4)

        self._build_artifacts_panel()
        self._build_chat_thread()

    def _build_chat_thread(self) -> None:
        # Chat log display container
        chat_container = ttk.Frame(self.left_frame)
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.txt_chat = tk.Text(
            chat_container,
            wrap=tk.WORD,
            font=("Inter", 10),
            bg="#131314",
            fg="#E3E3E3",
            insertbackground="#FFFFFF",
            padx=12,
            pady=12,
            relief=tk.FLAT,
        )
        self.txt_chat.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(chat_container, orient=tk.VERTICAL, command=self.txt_chat.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_chat.config(yscrollcommand=scrollbar.set)

        self._append_chat_message(
            "Assistant",
            "Welcome to TATVA RISC-V Optimization Studio — Antigravity Engine! 🚀\n\n"
            "Interact via natural language or use the interactive engineering cards below:\n"
            " • Upload & analyze ONNX / PyTorch models\n"
            " • Target selection (RV64GCV Vector, RV64GC, RV32IMC)\n"
            " • Schraudolph Softmax Fusion & INT8 Quantization passes\n"
            " • Autonomous Closed-Loop Engineering Engine (Ollama / Claude + GCC + QEMU self-correction)",
        )
        self._build_inline_cards()

        # Fixed Chat Input Box at Bottom
        input_container = ttk.Frame(self.left_frame, padding=5)
        input_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.txt_user_input = tk.Text(
            input_container,
            height=3,
            font=("Inter", 10),
            bg="#1E1E20",
            fg="#E3E3E3",
            insertbackground="#FFFFFF",
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.txt_user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.txt_user_input.bind("<Return>", self._on_enter_pressed)

        btn_send = ttk.Button(input_container, text="Send ➔", style="Primary.TButton", command=self._send_chat_message)
        btn_send.pack(side=tk.RIGHT)

    def _build_inline_cards(self) -> None:
        card_box = ttk.LabelFrame(self.left_frame, text=" Interactive Control Cards ", padding=10)
        card_box.pack(fill=tk.X, pady=(0, 8))

        notebook_cards = ttk.Notebook(card_box)
        notebook_cards.pack(fill=tk.X, expand=True)

        # Card 1: Model Setup & Hardware Target
        frame_config = ttk.Frame(notebook_cards, padding=10)
        notebook_cards.add(frame_config, text=" 1. Model & Target Setup ")

        lbl_m = ttk.Label(frame_config, text="Model File:")
        lbl_m.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.entry_model = ttk.Entry(frame_config, font=("Consolas", 10), width=35)
        self.entry_model.grid(row=0, column=1, padx=5, sticky=tk.EW)

        btn_br = ttk.Button(frame_config, text="Browse...", command=self._browse_model)
        btn_br.grid(row=0, column=2, padx=2)

        btn_an = ttk.Button(frame_config, text="Analyze Graph", command=self._analyze_model)
        btn_an.grid(row=0, column=3, padx=2)

        lbl_t = ttk.Label(frame_config, text="Target:")
        lbl_t.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.cb_allow_exp = tk.BooleanVar(value=False)
        chk_exp = ttk.Checkbutton(
            frame_config,
            text="Experimental",
            variable=self.cb_allow_exp,
            command=self._update_target_dropdown,
        )
        chk_exp.grid(row=1, column=2, padx=2)

        self.cbo_targets = ttk.Combobox(frame_config, state="readonly", width=24)
        self.cbo_targets.grid(row=1, column=1, padx=5, sticky=tk.W)

        btn_vf = ttk.Button(frame_config, text="Verify Toolchain", command=self._verify_toolchain_configuration)
        btn_vf.grid(row=1, column=3, padx=2)

        # Card 2: Optimization Passes & Pipeline Execution
        frame_run = ttk.Frame(notebook_cards, padding=10)
        notebook_cards.add(frame_run, text=" 2. Passes & Pipeline Run ")

        self.cb_fuse_softmax = tk.BooleanVar(value=True)
        chk_fuse = ttk.Checkbutton(
            frame_run, text="Softmax Fusion (Schraudolph)", variable=self.cb_fuse_softmax
        )
        chk_fuse.pack(side=tk.LEFT, padx=(0, 10))

        self.cb_quantize = tk.BooleanVar(value=False)
        chk_quant = ttk.Checkbutton(
            frame_run, text="Dynamic INT8 Quantization", variable=self.cb_quantize, command=self._on_quantize_toggled
        )
        chk_quant.pack(side=tk.LEFT, padx=(0, 15))

        btn_run = ttk.Button(frame_run, text="🚀 Run Optimization Pipeline", style="Primary.TButton", command=self._start_pipeline_run)
        btn_run.pack(side=tk.LEFT, padx=(0, 5))

        btn_cancel = ttk.Button(frame_run, text="Cancel", command=self._cancel_run)
        btn_cancel.pack(side=tk.LEFT)

        # Card 3: Autonomous Closed-Loop Agent Engine ("Tatva Antigravity Engine")
        frame_scaffold = ttk.Frame(notebook_cards, padding=10)
        notebook_cards.add(frame_scaffold, text=" 3. Antigravity Engine ")

        # Provider & Model Configuration Control Panel (Scaffolding View)
        config_panel = ttk.LabelFrame(frame_scaffold, text=" Provider & Model Configuration ", padding=8)
        config_panel.pack(fill=tk.X, pady=(0, 6))

        # Row 1: Provider Selector, NVIDIA Key Input, Password Toggle, Sync Button, Toolchain Badge
        row1 = ttk.Frame(config_panel)
        row1.pack(fill=tk.X, pady=(0, 4))

        lbl_prov = ttk.Label(row1, text="Provider:")
        lbl_prov.pack(side=tk.LEFT, padx=(0, 3))

        self.cbo_provider = ttk.Combobox(
            row1,
            values=["Local (Ollama)", "NVIDIA NIM", "OpenAI", "Anthropic"],
            state="readonly",
            width=14,
        )
        self.cbo_provider.pack(side=tk.LEFT, padx=(0, 8))
        self.cbo_provider.set("NVIDIA NIM")
        self.cbo_provider.bind("<<ComboboxSelected>>", self._on_provider_changed)

        lbl_nv = ttk.Label(row1, text="NVIDIA Key:")
        lbl_nv.pack(side=tk.LEFT, padx=(4, 2))

        self.entry_nvidia_key = ttk.Entry(row1, font=("Consolas", 8), width=18, show="*")
        self.entry_nvidia_key.pack(side=tk.LEFT, padx=(0, 2))
        try:
            from tatva.config import get_nvidia_api_key
            env_nv_key = get_nvidia_api_key() or ""
            if env_nv_key:
                self.entry_nvidia_key.insert(0, env_nv_key)
        except Exception:
            pass

        self.btn_toggle_key = ttk.Button(row1, text="👁", width=3, command=self._toggle_key_visibility)
        self.btn_toggle_key.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_sync_nv = ttk.Button(row1, text="🔄 Sync / Refresh Models", command=self._sync_nvidia_models)
        self.btn_sync_nv.pack(side=tk.LEFT, padx=(0, 5))

        self.lbl_toolchain_badge = tk.Label(
            row1, text=" 🟢 Toolchain Ready ", font=("JetBrains Mono", 8, "bold"), bg="#1E3A5F", fg="#60A5FA", padx=6, pady=2
        )
        self.lbl_toolchain_badge.pack(side=tk.RIGHT)

        # Row 2: Dynamic Model Picker Dropdown, Search Filter, Inline Status Error
        row2 = ttk.Frame(config_panel)
        row2.pack(fill=tk.X)

        lbl_backend = ttk.Label(row2, text="Model Picker:")
        lbl_backend.pack(side=tk.LEFT, padx=(0, 3))

        self.cbo_llm_backend = ttk.Combobox(row2, width=32)
        self.cbo_llm_backend.pack(side=tk.LEFT, padx=(0, 8))
        self.cbo_llm_backend.bind("<<ComboboxSelected>>", lambda e: self._save_scaffolding_state())

        lbl_filter = ttk.Label(row2, text="Search Filter:")
        lbl_filter.pack(side=tk.LEFT, padx=(4, 2))

        self.entry_model_filter = ttk.Entry(row2, font=("Consolas", 8), width=16)
        self.entry_model_filter.pack(side=tk.LEFT, padx=(0, 8))
        self.entry_model_filter.bind("<KeyRelease>", self._filter_llm_models)

        self.lbl_nv_status = ttk.Label(row2, text="", font=("Inter", 8), foreground="#F87171")
        self.lbl_nv_status.pack(side=tk.LEFT)

        lbl_prompt = ttk.Label(frame_scaffold, text="Autonomous Specification:")
        lbl_prompt.pack(anchor=tk.W, pady=(0, 2))

        self.txt_scaf_prompt = tk.Text(frame_scaffold, height=2, font=("Consolas", 9), wrap=tk.WORD, bg="#131314", fg="#E3E3E3")
        self.txt_scaf_prompt.pack(fill=tk.X, pady=(0, 6))
        self.txt_scaf_prompt.insert(
            tk.END, "Keyword-spotting classifier for RV64GCV with 8-bit quantization, C harness, and QEMU verification."
        )

        scaf_btn_bar = ttk.Frame(frame_scaffold)
        scaf_btn_bar.pack(fill=tk.X)

        btn_gen_scaf = ttk.Button(
            scaf_btn_bar, text="🚀 Run Autonomous Loop", style="Primary.TButton", command=self._generate_scaffold
        )
        btn_gen_scaf.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_scaf_cost = ttk.Label(
            scaf_btn_bar, text="$0.00000 (Local Engine Enabled)", font=("JetBrains Mono", 8), foreground="#9AA0A6"
        )
        self.lbl_scaf_cost.pack(side=tk.LEFT, padx=5)

        self._populate_llm_backends()
        self._load_scaffolding_state()
        self._update_target_dropdown()
        self.cbo_targets.bind("<<ComboboxSelected>>", self._on_target_selected)

    def _toggle_key_visibility(self) -> None:
        if self.entry_nvidia_key.cget("show") == "*":
            self.entry_nvidia_key.config(show="")
        else:
            self.entry_nvidia_key.config(show="*")

    def _on_provider_changed(self, event: Any = None) -> None:
        prov = self.cbo_provider.get().strip()
        if prov == "NVIDIA NIM":
            self.entry_nvidia_key.config(state="normal")
            self.btn_sync_nv.config(state="normal")
            self.btn_toggle_key.config(state="normal")
            key = self.entry_nvidia_key.get().strip()
            if key:
                self._sync_nvidia_models()
            else:
                defaults = ["Click Sync to load models", "nvidia/llama-3.1-nemotron-70b-instruct", "deepseek-ai/deepseek-r1"]
                self._all_models_cache = defaults
                self.cbo_llm_backend["values"] = defaults
                self.cbo_llm_backend.current(0)
        elif prov == "Local (Ollama)":
            self.entry_nvidia_key.config(state="disabled")
            self.btn_sync_nv.config(state="disabled")
            self.btn_toggle_key.config(state="disabled")
            self.lbl_nv_status.config(text="")
            try:
                from scaffolding.llm_provider import get_local_ollama_models
                local_models = get_local_ollama_models()
                models = [f"Ollama: {m} (Local / Free)" for m in local_models] if local_models else ["Ollama: qwen2.5-coder:7b (Local / Free)", "Ollama: deepseek-coder-v2 (Local / Free)"]
            except Exception:
                models = ["Ollama: qwen2.5-coder:7b (Local / Free)"]
            self._all_models_cache = models
            self.cbo_llm_backend["values"] = models
            if models:
                self.cbo_llm_backend.current(0)
        elif prov == "OpenAI":
            self.entry_nvidia_key.config(state="disabled")
            self.btn_sync_nv.config(state="disabled")
            self.btn_toggle_key.config(state="disabled")
            self.lbl_nv_status.config(text="")
            models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o3-mini"]
            self._all_models_cache = models
            self.cbo_llm_backend["values"] = models
            self.cbo_llm_backend.current(0)
        elif prov == "Anthropic":
            self.entry_nvidia_key.config(state="disabled")
            self.btn_sync_nv.config(state="disabled")
            self.btn_toggle_key.config(state="disabled")
            self.lbl_nv_status.config(text="")
            from tatva.config import ANTHROPIC_MODEL_LABEL

            models = [ANTHROPIC_MODEL_LABEL]
            self._all_models_cache = models
            self.cbo_llm_backend["values"] = models
            self.cbo_llm_backend.current(0)

        self._save_scaffolding_state()

    def _sync_nvidia_models(self) -> None:
        key = self.entry_nvidia_key.get().strip()
        if not key:
            try:
                from tatva.config import get_nvidia_api_key
                key = get_nvidia_api_key() or ""
            except Exception:
                pass

        if not key:
            self.lbl_nv_status.config(
                text="⚠️ Failed to fetch NVIDIA model catalog. Verify your nvapi-... key.",
                foreground="#F87171"
            )
            return

        self.btn_sync_nv.config(state="disabled")
        self.lbl_nv_status.config(text="⏳ Fetching NVIDIA catalog...", foreground="#60A5FA")

        def worker():
            try:
                from scaffolding.llm_provider import fetch_nvidia_models
                models, err = fetch_nvidia_models(key)
                if err or not models:
                    msg = err or "Failed to fetch NVIDIA model catalog. Verify your nvapi-... key."
                    def update_err():
                        self.lbl_nv_status.config(text=f"⚠️ {msg}", foreground="#F87171")
                        self.btn_sync_nv.config(state="normal")
                    self._ui_call(update_err)
                else:
                    formatted_nvidia = [f"NVIDIA: {m}" for m in models]
                    self._all_models_cache = formatted_nvidia

                    def update_ui():
                        self.cbo_llm_backend["values"] = formatted_nvidia
                        if formatted_nvidia:
                            self.cbo_llm_backend.current(0)
                        self.lbl_nv_status.config(
                            text=f"✅ Synced {len(models)} live models!",
                            foreground="#34D399"
                        )
                        self.btn_sync_nv.config(state="normal")
                        self._save_scaffolding_state()
                    self._ui_call(update_ui)
            except Exception as exc:
                # Bind the text now. Python clears the `as e` name at the end of the
                # except block, so a closure that read it fired NameError instead of
                # showing the user why the sync failed.
                detail = f"⚠️ Could not fetch the NVIDIA model catalog: {exc}"

                def update_ex():
                    self.lbl_nv_status.config(text=detail, foreground="#F87171")
                    self.btn_sync_nv.config(state="normal")
                self._ui_call(update_ex)

        threading.Thread(target=worker, daemon=True).start()

    def _filter_llm_models(self, event: Any = None) -> None:
        query = self.entry_model_filter.get().strip().lower()
        all_models = getattr(self, "_all_models_cache", list(self.cbo_llm_backend["values"]))
        if not query:
            self.cbo_llm_backend["values"] = all_models
        else:
            filtered = [m for m in all_models if query in m.lower()]
            self.cbo_llm_backend["values"] = filtered
            if filtered:
                self.cbo_llm_backend.current(0)

    def _save_scaffolding_state(self) -> None:
        try:
            from scaffolding.config import ScaffoldingConfig
            cfg = ScaffoldingConfig.load()
            if hasattr(self, "cbo_provider"):
                cfg.provider = self.cbo_provider.get().strip()
            if hasattr(self, "entry_nvidia_key"):
                cfg.nvidia_api_key = self.entry_nvidia_key.get().strip()
            if hasattr(self, "cbo_llm_backend"):
                cfg.selected_model = self.cbo_llm_backend.get().strip()
            cfg.save()
        except Exception:
            pass

    def _load_scaffolding_state(self) -> None:
        try:
            from scaffolding.config import ScaffoldingConfig
            cfg = ScaffoldingConfig.load()
            if hasattr(self, "cbo_provider") and cfg.provider:
                self.cbo_provider.set(cfg.provider)
            if hasattr(self, "entry_nvidia_key"):
                from tatva.config import get_nvidia_api_key
                key_val = cfg.nvidia_api_key or get_nvidia_api_key() or ""
                if key_val:
                    self.entry_nvidia_key.insert(0, key_val)
            self._on_provider_changed()
            if hasattr(self, "cbo_llm_backend") and cfg.selected_model:
                vals = list(self.cbo_llm_backend["values"])
                if cfg.selected_model in vals:
                    self.cbo_llm_backend.set(cfg.selected_model)
        except Exception:
            pass


    def _populate_llm_backends(self) -> None:
        try:
            from scaffolding.llm_provider import LLMProvider
            nv_key = getattr(self, "entry_nvidia_key", None)
            key_str = nv_key.get().strip() if nv_key else None
            provider = LLMProvider()
            models = provider.get_available_models(key_str)
            self._all_models_cache = models
            self.cbo_llm_backend["values"] = models
            if models:
                self.cbo_llm_backend.current(0)
        except Exception:
            from tatva.config import ANTHROPIC_MODEL_LABEL

            self.cbo_llm_backend["values"] = [ANTHROPIC_MODEL_LABEL, "Ollama: qwen2.5-coder (Local)"]
            self.cbo_llm_backend.current(0)


    def _build_artifacts_panel(self) -> None:
        lbl_art_hdr = ttk.Label(self.right_frame, text="📑 Artifacts & Code Viewer", style="Header.TLabel")
        lbl_art_hdr.pack(anchor=tk.W, pady=(0, 5))

        self.notebook_art = ttk.Notebook(self.right_frame)
        self.notebook_art.pack(fill=tk.BOTH, expand=True)

        # Tab A: Model Summary
        self.tab_model_art = ttk.Frame(self.notebook_art, padding=5)
        self.notebook_art.add(self.tab_model_art, text=" Model Summary ")
        self.txt_model_summary = tk.Text(self.tab_model_art, wrap=tk.WORD, font=("Consolas", 9), bg="#131314", fg="#E3E3E3")
        self.txt_model_summary.pack(fill=tk.BOTH, expand=True)
        self.txt_model_summary.insert(tk.END, "No model loaded. Click 'Analyze Graph' in Setup card.")

        # Tab B: Target Specifications
        self.tab_target_art = ttk.Frame(self.notebook_art, padding=5)
        self.notebook_art.add(self.tab_target_art, text=" Target Specs ")
        self.txt_target_notes = tk.Text(self.tab_target_art, wrap=tk.WORD, font=("Consolas", 9), bg="#131314", fg="#E3E3E3")
        self.txt_target_notes.pack(fill=tk.BOTH, expand=True)
        self.txt_target_notes.insert(tk.END, "Select a target variant to view architecture specifications.")

        # Tab C: Benchmark Results
        self.tab_results_art = ttk.Frame(self.notebook_art, padding=5)
        self.notebook_art.add(self.tab_results_art, text=" Results & Parity ")
        self.txt_results = tk.Text(self.tab_results_art, wrap=tk.WORD, font=("Consolas", 9), bg="#131314", fg="#E3E3E3")
        self.txt_results.pack(fill=tk.BOTH, expand=True)
        self.txt_results.insert(tk.END, "No benchmark run metrics available yet.")

        # Tab D: Diagnostics
        self.tab_diag_art = ttk.Frame(self.notebook_art, padding=5)
        self.notebook_art.add(self.tab_diag_art, text=" Diagnostics ")
        self.txt_diag = tk.Text(self.tab_diag_art, wrap=tk.WORD, font=("Consolas", 9), bg="#131314", fg="#E3E3E3")
        self.txt_diag.pack(fill=tk.BOTH, expand=True)
        self.txt_diag.insert(tk.END, "No compiler failures reported.")

        # Tab E: Scaffolding Generator Inspector
        self.tab_scaf_art = ttk.Frame(self.notebook_art, padding=5)
        self.notebook_art.add(self.tab_scaf_art, text=" Workspace Files ")

        scaf_top = ttk.Frame(self.tab_scaf_art)
        scaf_top.pack(fill=tk.X, pady=(0, 5))

        lbl_fn = ttk.Label(scaf_top, text="Select File:")
        lbl_fn.pack(side=tk.LEFT, padx=(0, 5))

        self.cbo_scaf_files = ttk.Combobox(scaf_top, state="readonly", width=25)
        self.cbo_scaf_files.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.cbo_scaf_files.bind("<<ComboboxSelected>>", self._on_scaf_file_selected)

        self.lbl_ast_badge = tk.Label(
            scaf_top, text=" AST: N/A ", font=("JetBrains Mono", 8, "bold"), bg="#282A2C", fg="white", padx=4
        )
        self.lbl_ast_badge.pack(side=tk.RIGHT)

        self.txt_scaf_content = tk.Text(self.tab_scaf_art, wrap=tk.NONE, font=("Consolas", 9), bg="#131314", fg="#E3E3E3")
        self.txt_scaf_content.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        scaf_bottom = ttk.Frame(self.tab_scaf_art)
        scaf_bottom.pack(fill=tk.X)

        btn_write_disk = ttk.Button(scaf_bottom, text="💾 Approve & Write Project to Disk", command=self._write_scaffold_disk)
        btn_write_disk.pack(side=tk.LEFT, padx=(0, 5))

        btn_handoff = ttk.Button(scaf_bottom, text="⚡ Handoff to Tatva Pipeline", command=self._handoff_scaffold_to_pipeline)
        btn_handoff.pack(side=tk.LEFT)

    def _toggle_artifacts_panel(self) -> None:
        if self.right_frame.winfo_viewable():
            self.paned.forget(self.right_frame)
        else:
            self.paned.add(self.right_frame, weight=4)

    def _append_chat_message(self, sender: str, text: str) -> None:
        self.txt_chat.insert(tk.END, f"\n[{sender}]:\n", "bold_sender")
        self.txt_chat.insert(tk.END, f"{text}\n")
        self.txt_chat.see(tk.END)

    def _on_enter_pressed(self, event: Any) -> str:
        if not event.state & 0x0001:
            self._send_chat_message()
            return "break"
        return ""

    def _send_chat_message(self) -> None:
        msg = self.txt_user_input.get("1.0", tk.END).strip()
        if not msg:
            return
        self.txt_user_input.delete("1.0", tk.END)

        self._append_chat_message("User", msg)

        msg_lower = msg.lower()
        if "analyze" in msg_lower or "model" in msg_lower:
            path = self.entry_model.get().strip()
            if path and os.path.exists(path):
                self._analyze_model()
            else:
                self._append_chat_message(
                    "Assistant", "Please select a model file path in the Setup Card and click 'Analyze Graph'."
                )
        elif "run" in msg_lower or "pipeline" in msg_lower or "optimize" in msg_lower:
            self._start_pipeline_run()
        elif "scaffold" in msg_lower or "generate" in msg_lower or "loop" in msg_lower:
            self.txt_scaf_prompt.delete("1.0", tk.END)
            self.txt_scaf_prompt.insert(tk.END, msg)
            self._generate_scaffold()
        else:
            self._append_chat_message(
                "Assistant",
                f"Received request: '{msg}'.\n\n"
                "I can assist you with:\n"
                " • Analyzing your model graph ('analyze')\n"
                " • Running the compilation & QEMU pipeline ('run')\n"
                " • Autonomous Closed-Loop Engineering Engine ('generate')",
            )

    # --- Setup & Target Selection ---
    def _browse_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select ONNX Model File",
            filetypes=[("ONNX Models", "*.onnx"), ("All Files", "*.*")],
        )
        if filename:
            self.entry_model.delete(0, tk.END)
            self.entry_model.insert(0, filename)
            self.model_path = filename
            self._append_chat_message("Assistant", f"Selected model file: `{filename}`")

    def _analyze_model(self) -> None:
        path = self.entry_model.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Invalid Model", "Please select a valid ONNX model file.")
            return

        self.model_path = path
        try:
            import_fn = _backend.get("import_model")
            analyze_fn = _backend.get("analyze_graph")

            if import_fn and analyze_fn:
                self.imported_ir = import_fn(path)
                self.graph_stats = analyze_fn(self.imported_ir)

                self.txt_model_summary.delete("1.0", tk.END)
                self.txt_model_summary.insert(tk.END, f"Model File: {os.path.basename(path)}\n")
                self.txt_model_summary.insert(tk.END, f"File Size:  {os.path.getsize(path) / 1024:.1f} KB\n")
                self.txt_model_summary.insert(tk.END, f"Total Ops:  {self.graph_stats.total_ops}\n\n")
                self.txt_model_summary.insert(tk.END, "Operator Histogram:\n")
                for op, count in self.graph_stats.op_histogram.items():
                    self.txt_model_summary.insert(tk.END, f"  - {op:<20}: {count}\n")
                self.txt_model_summary.insert(
                    tk.END,
                    f"\nTransformer Bottleneck Detected: {'YES' if self.graph_stats.has_transformer_bottleneck else 'NO'}\n",
                )

                self.notebook_art.select(self.tab_model_art)
                self._append_chat_message(
                    "Assistant",
                    f"Analyzed model graph `{os.path.basename(path)}`.\n"
                    f"Total Ops: {self.graph_stats.total_ops} | Bottleneck: {'YES' if self.graph_stats.has_transformer_bottleneck else 'NO'}.\n"
                    "Summary opened in the Artifacts Panel on the right.",
                )
            else:
                self._append_chat_message("Assistant", f"Loaded model file path: {path}")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze model:\n{e}")

    def _update_target_dropdown(self) -> None:
        targets_dict = _backend.get("TARGETS", {})
        allow_exp = self.cb_allow_exp.get()

        options = []
        for name, variant in targets_dict.items():
            if variant.experimental and not allow_exp:
                continue
            display_name = f"{name} [EXPERIMENTAL]" if variant.experimental else name
            options.append(display_name)

        self.cbo_targets["values"] = options
        if options and not self.cbo_targets.get():
            self.cbo_targets.current(0)
            self._on_target_selected(None)

    def _on_target_selected(self, event: Any) -> None:
        if not hasattr(self, "txt_target_notes"):
            return
        val = self.cbo_targets.get()
        if not val:
            return
        tgt_name = val.split()[0]
        targets_dict = _backend.get("TARGETS", {})
        if tgt_name in targets_dict:
            variant = targets_dict[tgt_name]
            self.txt_target_notes.delete("1.0", tk.END)
            self.txt_target_notes.insert(tk.END, f"Target Name:   {variant.name}\n")
            self.txt_target_notes.insert(tk.END, f"GCC Architecture: {variant.gcc_march}\n")
            self.txt_target_notes.insert(tk.END, f"GCC ABI:          {variant.gcc_mabi}\n")
            self.txt_target_notes.insert(tk.END, f"Bitness:          {variant.bitness}-bit\n")
            self.txt_target_notes.insert(
                tk.END,
                f"Experimental:     {'YES (Gated)' if variant.experimental else 'NO (Baseline)'}\n\n",
            )
            self.txt_target_notes.insert(tk.END, f"Description:\n  {variant.notes}\n")

    def _verify_toolchain_configuration(self) -> None:
        val = self.cbo_targets.get()
        if not val:
            return
        tgt_name = val.split()[0]
        targets_dict = _backend.get("TARGETS", {})
        verify_fn = _backend.get("verify_target")

        if tgt_name in targets_dict and verify_fn:
            variant = targets_dict[tgt_name]
            try:
                res = verify_fn(variant)
                if res["status"] == "ok":
                    messagebox.showinfo(
                        "Toolchain Verified",
                        f"✅ Toolchain for '{tgt_name}' is correctly configured:\n"
                        f"  GCC march: {variant.gcc_march}\n"
                        f"  GCC mabi:  {variant.gcc_mabi}\n"
                        f"  QEMU:      qemu-system-riscv64 ready",
                    )
                    self._append_chat_message("Assistant", f"✅ Verified toolchain configuration for `{tgt_name}`.")
                else:
                    messagebox.showwarning("Toolchain Issue", f"Toolchain issue:\n{res['error']}")
            except Exception as e:
                messagebox.showerror("Verification Exception", str(e))

    def _on_quantize_toggled(self) -> None:
        pass

    # --- Execution Pipeline ---
    def _start_pipeline_run(self) -> None:
        path = self.entry_model.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Missing Model", "Please select a valid ONNX model file.")
            return

        val = self.cbo_targets.get()
        if not val:
            messagebox.showwarning("Missing Target", "Please select a target architecture.")
            return

        tgt_name = val.split()[0]
        targets_dict = _backend.get("TARGETS", {})
        variant = targets_dict.get(tgt_name)

        if not variant:
            messagebox.showerror("Error", f"Unknown target '{tgt_name}'")
            return

        self.cancel_requested = False
        self._append_chat_message(
            "Assistant",
            f"🚀 Starting TATVA Optimization Pipeline...\n"
            f"  Model:  {os.path.basename(path)}\n"
            f"  Target: {variant.name} ({variant.gcc_march})\n"
            f"  Softmax Fusion: {'ENABLED' if self.cb_fuse_softmax.get() else 'DISABLED'}\n"
            f"  INT8 Quantize:  {'ENABLED' if self.cb_quantize.get() else 'DISABLED'}",
        )

        threading.Thread(
            target=self._pipeline_worker,
            args=(path, variant, self.cb_fuse_softmax.get(), self.cb_quantize.get()),
            daemon=True,
        ).start()

    def _cancel_run(self) -> None:
        self.cancel_requested = True
        self._append_chat_message("Assistant", "⚠️ Pipeline run cancellation requested by user.")

    def _pipeline_worker(self, path: str, variant: Any, fuse_softmax: bool, do_quantize: bool) -> None:
        try:
            import_fn = _backend["import_model"]
            establish_baseline_fn = _backend["establish_baseline"]
            fuse_fn = _backend["select_fast_softmax_kernel"]
            quantize_fn = _backend["quantize"]
            compile_fn = _backend["compile_model"]
            run_measure_fn = _backend["run_and_measure"]

            baseline_res = establish_baseline_fn(path, variant)
            if self.cancel_requested:
                return

            ir = import_fn(path)
            opt_ir = ir
            if fuse_softmax:
                opt_ir = fuse_fn(opt_ir)
            if do_quantize:
                opt_ir = quantize_fn(opt_ir)

            if self.cancel_requested:
                return

            # Same root the CLI uses, rather than a path relative to this source file --
            # which for an installed wheel points inside site-packages. See
            # tatva.runner.build_root.
            from tatva.runner import build_root

            build_dir = os.path.join(build_root(), f"gui_build_{variant.name}")
            artifact = compile_fn(opt_ir, variant, build_dir, warmup_count=2, timed_count=5)

            if self.cancel_requested:
                return

            measurement = run_measure_fn(artifact)

            target_logits = []
            for line in measurement.raw_output.splitlines():
                if "FIRST_LOGITS:" in line:
                    parts = line.strip().split(":")[1].strip().split()
                    target_logits = [float(x) for x in parts]
                    break

            # Accuracy is measured against the host reference, not the QEMU baseline.
            mse = 0.0
            if target_logits and baseline_res.ref_logits:
                min_l = min(len(target_logits), len(baseline_res.ref_logits))
                import numpy as np

                mse = float(np.mean((np.array(target_logits[:min_l]) - np.array(baseline_res.ref_logits[:min_l])) ** 2))

            self._ui_call(
                lambda: self._update_results_ui(
                    baseline_res.latency_result.mean_ms,
                    measurement.mean_ms,
                    mse,
                    measurement.environment,
                )
            )
        except Exception as exc:
            classify_fn = _backend.get("classify_failure")
            explain_fn = _backend.get("explain")

            if classify_fn and explain_fn:
                diag_msg = explain_fn(classify_fn(exc))
                self._ui_call(lambda: self._update_diagnostics_ui(diag_msg))
            else:
                # Format now: `exc` is unbound by the time the lambda runs on the UI thread.
                failure = f"❌ Pipeline Failed: {exc}"
                self._ui_call(lambda: self._append_chat_message("Assistant", failure))

    def _update_results_ui(self, base_ms: float, opt_ms: float, mse: float, env: str) -> None:
        speedup = ((base_ms - opt_ms) / base_ms) * 100.0 if base_ms > 0 else 0.0
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.insert(tk.END, "=======================================================\n")
        self.txt_results.insert(tk.END, "            TATVA BENCHMARK RESULTS & PARITY           \n")
        self.txt_results.insert(tk.END, "=======================================================\n\n")
        self.txt_results.insert(tk.END, f"Environment:             [{env}] (Instruction-Accurate QEMU)\n")
        self.txt_results.insert(tk.END, f"Baseline Mean Latency:  {base_ms:.4f} ms\n")
        self.txt_results.insert(tk.END, f"Optimized Mean Latency: {opt_ms:.4f} ms\n")
        self.txt_results.insert(tk.END, f"Execution Speedup:      {speedup:+.2f}%\n")
        self.txt_results.insert(tk.END, f"Numerical MSE Parity:   {mse:.6f}\n")
        self.txt_results.insert(tk.END, f"Parity Verdict:         {'PASS [OK]' if mse < 0.05 else 'FAIL [ACCURACY DROP]'}\n")

        self.notebook_art.select(self.tab_results_art)
        self._append_chat_message(
            "Assistant",
            f"🎉 Benchmark execution completed successfully!\n"
            f"  Baseline:  {base_ms:.4f} ms\n"
            f"  Optimized: {opt_ms:.4f} ms\n"
            f"  Speedup:   {speedup:+.2f}%\n"
            f"  Parity:    {'PASS [OK]' if mse < 0.05 else 'FAIL'}\n"
            "Full comparison table opened in the Artifacts Panel.",
        )

    def _update_diagnostics_ui(self, diag_msg: str) -> None:
        self.txt_diag.delete("1.0", tk.END)
        self.txt_diag.insert(tk.END, "=======================================================\n")
        self.txt_diag.insert(tk.END, "              TATVA DIAGNOSTICS REPORT                 \n")
        self.txt_diag.insert(tk.END, "=======================================================\n\n")
        self.txt_diag.insert(tk.END, diag_msg)

        self.notebook_art.select(self.tab_diag_art)
        self._append_chat_message(
            "Assistant",
            "⚠️ Compiler pipeline error encountered.\n"
            "Plain-English explanation and fix recommendations opened in the Diagnostics Artifact Panel.",
        )

    # --- Project Scaffolding Assistant Integration ---
    def _init_scaffolding_agent(self) -> None:
        try:
            from scaffolding.agent import ScaffoldingAgent
            from scaffolding.config import ScaffoldingConfig

            cfg = ScaffoldingConfig.load()
            self.scaffolding_agent = ScaffoldingAgent(cfg)
        except Exception as e:
            print(f"Scaffolding agent initialization note: {e}")

    def _generate_scaffold(self) -> None:
        prompt = self.txt_scaf_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a project description prompt.")
            return

        model_name = self.cbo_llm_backend.get() or "Ollama: qwen2.5-coder (Local)"
        self._append_chat_message(
            "Assistant", f"🤖 Running Autonomous Closed-Loop Verification for prompt: '{prompt}' using {model_name}..."
        )

        def worker() -> None:
            try:
                if not self.scaffolding_agent:
                    self._init_scaffolding_agent()

                res = self.scaffolding_agent.loop_agent.run_autonomous_loop(
                    prompt_text=prompt,
                    model_name=model_name,
                    log_callback=lambda msg: self._ui_call(
                        lambda: self._append_chat_message("Antigravity Engine", msg)
                    ),
                )
                self._ui_call(lambda: self._on_scaffold_generated(res))
            except Exception as exc:
                failure = f"❌ Scaffolding Error: {exc}"
                self._ui_call(lambda: self._append_chat_message("Assistant", failure))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scaffold_generated(self, res: dict[str, Any]) -> None:
        self.scaffolding_files_data = res.get("files", [])
        cost = res.get("cumulative_cost_usd", 0.0)

        options = [f["path"] for f in self.scaffolding_files_data]
        self.cbo_scaf_files["values"] = options
        if options:
            self.cbo_scaf_files.current(0)
            self._on_scaf_file_selected(None)

        self.notebook_art.select(self.tab_scaf_art)
        # Say which one the user actually got. A template fallback that reads as
        # "generated from your prompt" is the whole problem this line exists to avoid.
        source = res.get("workspace_source")
        origin = {
            "llm": "Generated from your prompt by the LLM.",
            "template": "⚠ Built-in starter template — no LLM was reachable, so your prompt did not shape the code.",
        }.get(source, "Origin not reported.")
        self._append_chat_message(
            "Assistant",
            f"✨ Autonomous closed-loop generation finished for '{res.get('project_name', 'RISC-V Antigravity Starter')}'!\n"
            f"  Source:        {origin}\n"
            f"  Files created: {len(self.scaffolding_files_data)}\n"
            f"  Attempts used: {res.get('attempts_used', 1)}/5\n"
            f"  Cost:          ${cost:.5f}\n"
            f"  Safety Gate:   In-memory only (click 'Approve & Write Project to Disk' in Artifacts panel to save).",
        )

    def _on_scaf_file_selected(self, event: Any) -> None:
        val = self.cbo_scaf_files.get()
        if not val:
            return

        for item in self.scaffolding_files_data:
            if item["path"] == val:
                self.txt_scaf_content.delete("1.0", tk.END)
                self.txt_scaf_content.insert(tk.END, item["content"])

                ast_info = item.get("ast_check", {})
                passed = ast_info.get("passed", True)
                msg = ast_info.get("message", "PASSED")
                if passed:
                    self.lbl_ast_badge.config(text=f" AST: {msg} ", bg="#10B981", fg="white")
                else:
                    self.lbl_ast_badge.config(text=f" AST: {msg} ", bg="#EF4444", fg="white")
                break

    def _write_scaffold_disk(self) -> None:
        if not self.scaffolding_files_data or not self.scaffolding_agent:
            messagebox.showwarning("No Files", "Generate a project scaffold first.")
            return

        target_dir = filedialog.askdirectory(title="Select Output Directory for Project Files")
        if target_dir:
            created = self.scaffolding_agent.write_to_disk(target_dir, self.scaffolding_files_data)
            messagebox.showinfo("Files Written", f"Successfully wrote {len(created)} files to:\n{target_dir}")
            self._append_chat_message("Assistant", f"💾 Approved disk write: saved {len(created)} files to `{target_dir}`.")

            model_cand = os.path.join(target_dir, "models", "model.onnx")
            if os.path.exists(model_cand):
                self._load_model_from_scaffolding(model_cand)

    def _handoff_scaffold_to_pipeline(self) -> None:
        path = self.entry_model.get().strip()
        if path and os.path.exists(path):
            self._start_pipeline_run()
        else:
            messagebox.showwarning(
                "No Model", "No valid ONNX model currently selected. Please write files to disk first."
            )

    def _load_model_from_scaffolding(self, onnx_path: str) -> None:
        self.model_path = onnx_path
        self.entry_model.delete(0, tk.END)
        self.entry_model.insert(0, onnx_path)
        self._append_chat_message("Assistant", f"⚡ Loaded scaffolded model into setup card: `{onnx_path}`.")

    # --- API Helper methods for tests & external callers ---
    def validate_model_file(self, model_path: str) -> dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {"valid": False, "error": f"Path not found: '{model_path}'"}
        return {"valid": True, "path": model_path, "filename": os.path.basename(model_path)}

    def scan_hardware_boards(self) -> dict[str, Any]:
        return {"found": False, "status": "SIMULATION MODE — QEMU"}

    def analyze_model(self, model_path: str) -> dict[str, Any]:
        return self._analyze_model() or {}


def _diagnose(exc: BaseException) -> str:
    """
    Turn a caught exception into the plain-English cause-and-recommendation text the beta
    scope lists as the key differentiator.

    Every bridge method that can fail routes through here so the front end never has to
    show a bare `str(exception)`. `explain` prefers the Claude API when a key is
    configured and otherwise uses the deterministic offline rule engine, so this works
    with no network. Diagnosis is strictly additive — a failure to diagnose must never
    suppress the underlying error, so anything raised here is swallowed and the caller
    still reports the raw message alongside.
    """
    try:
        from tatva.diagnostics import classify_failure, explain
        return explain(classify_failure(exc))
    except Exception:
        return ""


def _validation_label(structural: str | None, numerical: str | None) -> str:
    """
    Describe how a single rewrite was checked, in the words of what actually happened.

    The repair engine records each of its two checks as passed / failed / skipped /
    not_run. A rewrite whose numerical comparison never ran must not read the same as one
    that ran and passed, so "not checked" is spelled out rather than collapsed into the
    structural result.
    """
    parts = []
    if structural == "passed":
        parts.append("structure re-verified")
    elif structural == "failed":
        parts.append("structural check FAILED")
    else:
        parts.append("structure not checked")

    if numerical == "passed":
        parts.append("output compared on the host and unchanged")
    elif numerical == "failed":
        parts.append("output changed — rewrite discarded")
    elif numerical == "skipped":
        parts.append("host comparison could not run")
    else:
        parts.append("output not compared")
    return "; ".join(parts).capitalize() + "."


class TatvaPyBridge:
    """
    Bidirectional Python-JavaScript PyBridge for PyWebView Edge WebView2 Desktop Window.
    Exposes compiler, runner, optimizer, and Antigravity closed-loop engine to website/index.html frontend.
    """

    def __init__(self) -> None:
        # Guards the toolchain-install state, which a worker thread writes and the
        # UI thread polls.
        self._install_lock = threading.Lock()
        self._install_state: dict[str, Any] = {
            "running": False,
            "done": False,
            "ok": False,
            "component": "",
            "label": "",
            "read": 0,
            "total": 0,
            "log": [],
            "error": "",
        }
        # Repaired graphs from `attempt_auto_fix`, keyed on (absolute model path, target
        # name). The rewritten module lives only in memory, so without this the build
        # that follows a successful auto-fix would re-import the original file and
        # quietly compile the graph the user was just told had been repaired.
        self._repair_lock = threading.Lock()
        self._repaired: dict[tuple[str, str], Any] = {}

    def select_file(self) -> str:
        """
        Open a native file browser for model selection.

        Cancelling has to be distinguishable from failing. The previous version returned
        early only when a path came back, so pressing Cancel on the native dialog fell
        through to the Tkinter block below and immediately opened a *second* dialog --
        in a different toolkit, with a different title, that the user then had to cancel
        as well. Tk is a fallback for "pywebview is not there", not for "the user said no".
        """
        try:
            import webview

            if webview.windows:
                res = webview.windows[0].create_file_dialog(
                    webview.OPEN_DIALOG,
                    file_types=('Model Files (*.onnx;*.pt;*.pth;*.tflite;*.keras;*.h5)', 'All Files (*.*)')
                )
                # res is None when cancelled, a sequence of paths when chosen. Either
                # way pywebview handled it and there is nothing left to fall back to.
                return res[0] if res else ""
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Native file dialog unavailable; falling back to Tk", exc_info=True)

        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Model Files", "*.onnx *.pt *.pth *.tflite *.keras *.h5"), ("All Files", "*.*")]
            )
            root.destroy()
            return path or ""
        except Exception:
            return ""

    def select_model_file(self) -> dict[str, Any]:
        """Wrapper for selecting model file returning dict with path."""
        path = self.select_file()
        return {"path": path}

    def get_build_info(self) -> dict[str, str]:
        return {"version": BUILD_VERSION, "label": BUILD_LABEL}

    def list_targets(self) -> list[dict[str, Any]]:
        """
        The real target registry, so the UI cannot drift from the compiler.

        The page used to hardcode three cards, one of which (RV64GCV) it presented
        without the experimental warning the CLI attaches to it.
        """
        try:
            from tatva.compiler import TARGETS

            return [
                {
                    "name": v.name,
                    "march": v.gcc_march,
                    "mabi": v.gcc_mabi,
                    "bitness": v.bitness,
                    "experimental": v.experimental,
                    "notes": v.notes,
                    "default": v.name == "RV64GC",
                }
                for v in TARGETS.values()
            ]
        except Exception:
            return []

    def list_sample_models(self) -> list[dict[str, Any]]:
        """
        Models that ship with the repo, so a first-time user has something to compile.

        Without this the "try it" path starts with "go find an ONNX file", which is a
        poor first five minutes. Only files that are actually on disk are returned --
        the list is built by looking, not by hardcoding names into the page.
        """
        roots = [
            os.path.join(getattr(sys, "_MEIPASS", ""), "models"),
            os.path.join(_CURRENT_DIR, "..", "..", "models"),
            os.path.join(_CURRENT_DIR, "..", "..", "..", "models"),
            os.path.join(os.getcwd(), "models"),
        ]

        # Ordered smallest-first: the nano model compiles in a fraction of the time the
        # pretrained one does, and a first run should not take twenty minutes.
        known = [
            ("model_mlp.onnx", "MLP fixture", "Two dense layers. The fastest end-to-end run."),
            ("model.onnx", "Tiny transformer block", "Attention pattern present — shows the fusion pass working."),
            ("model_nano.onnx", "Nano", "Small but realistic; a few minutes per build."),
            ("model_medium.onnx", "Medium", "Slower. Use once the small ones work."),
        ]

        out: list[dict[str, Any]] = []
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for filename, label, note in known:
                path = os.path.abspath(os.path.join(root, filename))
                if os.path.exists(path) and not any(s["path"] == path for s in out):
                    out.append(
                        {
                            "path": path,
                            "name": filename,
                            "label": label,
                            "note": note,
                            "size_kb": round(os.path.getsize(path) / 1024, 1),
                        }
                    )
            if out:
                break
        return out

    def get_toolchain_health(self) -> dict[str, Any]:
        try:
            from scaffolding.executor import ToolchainManager
            return ToolchainManager.get_health_status()
        except Exception as e:
            # Never claim a healthy toolchain we could not actually probe.
            return {
                "gcc": False,
                "gcc_name": "Unknown",
                "gcc_path": "",
                "qemu": False,
                "qemu_name": "Unknown",
                "qemu_path": "",
                "cmake": False,
                "make": False,
                "status_badge": "🔴 Toolchain probe failed",
                "error": f"Could not probe toolchain: {e}",
            }

    # ── in-app toolchain install ──────────────────────────────────────────────
    #
    # Without this, anyone who receives the zip gets through stages 01-04 and then
    # hits "RISC-V GCC cross-compiler binary not found" at stage 05, with the only
    # remedy being `tatva setup` in a terminal -- a command they do not have,
    # because they have an exe and not a checkout. The machinery already existed
    # in tatva.toolchain; it just had no button.

    def get_toolchain_plan(self) -> dict[str, Any]:
        """What an install would fetch, from where, to where. Touches no network."""
        try:
            from tatva.toolchain import COMPONENTS, plan_install, tools_dir

            items = []
            for key in COMPONENTS:
                plan = plan_install(key)
                items.append(
                    {
                        "key": key,
                        "label": plan.component.label,
                        "version": plan.component.version,
                        "url": plan.url,
                        "dest": plan.dest,
                        "size_mb": plan.component.approx_size_mb,
                        "installed": plan.already_installed,
                    }
                )
            return {"supported": True, "tools_dir": tools_dir(), "components": items, "error": None}
        except Exception as e:
            # Unsupported CPU/OS raises here. Say so plainly rather than offering a
            # button that can only fail.
            return {"supported": False, "tools_dir": "", "components": [], "error": str(e)}

    def start_toolchain_install(self, force: bool = False) -> dict[str, Any]:
        """
        Kick the download off on a worker thread and return immediately.

        A synchronous 520 MB download would block the js_api call, and with it the
        whole window, for several minutes with nothing on screen.
        """
        with self._install_lock:
            if self._install_state.get("running"):
                return {"started": False, "error": "An install is already running."}
            self._install_state = {
                "running": True,
                "done": False,
                "ok": False,
                "component": "",
                "label": "",
                "read": 0,
                "total": 0,
                "log": ["Starting…"],
                "error": "",
            }

        threading.Thread(target=self._install_worker, args=(bool(force),), daemon=True).start()
        return {"started": True, "error": None}

    def _install_worker(self, force: bool) -> None:
        from tatva.toolchain import COMPONENTS, install_component, plan_install

        def note(line: str) -> None:
            with self._install_lock:
                self._install_state["log"].append(line)

        ok = True
        try:
            for key in COMPONENTS:
                plan = plan_install(key)
                with self._install_lock:
                    self._install_state.update(
                        component=key, label=plan.component.label, read=0, total=0
                    )

                if plan.already_installed and not force:
                    note(f"{plan.component.label} — already installed, skipped")
                    continue

                note(f"Downloading {plan.component.label} (~{plan.component.approx_size_mb} MB)…")

                def on_progress(read: int, total: int) -> None:
                    with self._install_lock:
                        self._install_state["read"] = read
                        self._install_state["total"] = total

                path = install_component(key, force=force, progress=False, on_progress=on_progress)
                note(f"Installed → {path}")
        except Exception as e:
            ok = False
            with self._install_lock:
                self._install_state["error"] = str(e)
            note(f"FAILED: {e}")

        with self._install_lock:
            self._install_state["running"] = False
            self._install_state["done"] = True
            self._install_state["ok"] = ok
        if ok:
            note("Toolchain ready. Stages 04 and 05 can now run.")

    def get_toolchain_install_status(self) -> dict[str, Any]:
        with self._install_lock:
            return dict(self._install_state)

    def get_ollama_models(self) -> list[str]:
        """Return models the local Ollama server actually reports. Empty if it is not running."""
        try:
            from scaffolding.llm_provider import get_local_ollama_models
            return get_local_ollama_models()
        except Exception:
            return []

    def fetch_nvidia_models(self, api_key: str = "") -> dict[str, Any]:
        """Fetch live NVIDIA model catalog via PyBridge."""
        try:
            from scaffolding.llm_provider import fetch_nvidia_models
            models, err = fetch_nvidia_models(api_key)
            if err:
                return {"success": False, "models": [], "error": err}
            return {"success": True, "models": models, "error": None}
        except Exception as e:
            return {"success": False, "models": [], "error": f"Failed to fetch NVIDIA model catalog. Verify your nvapi-... key. ({e})"}

    def get_nvidia_models(self, api_key: str = "") -> list[str]:
        """Return list of NVIDIA model IDs or empty list on failure."""
        res = self.fetch_nvidia_models(api_key)
        return res.get("models", [])

    def ask_assistant(self, prompt: str, model_name: str = "", api_key: str = "") -> dict[str, Any]:
        """
        Send a question to whichever LLM the user has actually configured.

        The assistant panel used to print one canned sentence about "Gemini" for any
        input; there is no Gemini integration in this project and never was. This
        routes to the same LLMProvider the scaffolding engine uses, and when nothing
        is reachable it says so instead of inventing an answer.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return {"success": False, "reply": "", "error": "Empty prompt."}

        system_prompt = (
            "You are the TATVA compiler assistant. TATVA compiles ONNX transformer "
            "models to standalone C99 bare-metal RISC-V binaries via Apache TVM Relax "
            "and benchmarks them under QEMU system-mode emulation. Targets are "
            "RV32IMC, RV32IMAC, RV64GC, RV64IMAFDC, RV64GCV and RV32EMC. Answer "
            "concisely and say plainly when you do not know."
        )

        try:
            from scaffolding.llm_provider import LLMProvider

            provider = LLMProvider()
            model = (model_name or "").strip()
            if not model:
                local = self.get_ollama_models()
                if not local:
                    return {
                        "success": False,
                        "reply": "",
                        "error": (
                            "No LLM is configured. Start a local Ollama server "
                            "(`ollama serve`), or pick an NVIDIA/Anthropic model and "
                            "supply its key."
                        ),
                    }
                model = f"Ollama: {local[0]} (Local / Free)"

            reply, cost = provider.query(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=[],
                model_name=model,
                api_key=api_key or None,
            )
            return {"success": True, "reply": reply, "model": model, "cost_usd": cost, "error": None}
        except Exception as e:
            return {"success": False, "reply": "", "error": str(e)}

    def validate_model_file(self, model_path: str) -> dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {"valid": False, "error": f"Path not found: '{model_path}'"}
        
        import hashlib
        size_bytes = os.path.getsize(model_path)
        size_mb = round(size_bytes / (1024.0 * 1024.0), 2)
        ext = os.path.splitext(model_path)[1].lower()
        framework_map = {
            ".onnx": "ONNX IR Model",
            ".pt": "PyTorch TorchScript",
            ".pth": "PyTorch Weights",
            ".tflite": "TensorFlow Lite",
            ".keras": "Keras Model",
            ".h5": "HDF5 Model",
        }
        framework = framework_map.get(ext, "Neural Net Model")

        try:
            with open(model_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()[:12]
        except Exception as e:
            return {"valid": False, "error": f"Could not read '{model_path}': {e}"}

        # Real op count / bottleneck detection, read straight from the ONNX graph.
        # Unknown stays unknown -- we do not invent a layer count.
        layer_count = "unknown"
        has_bottleneck: bool | None = None
        parse_error = ""
        if ext == ".onnx":
            try:
                import onnx

                graph = onnx.load(model_path).graph
                op_types = [n.op_type for n in graph.node]
                layer_count = f"{len(op_types)} Ops"
                has_bottleneck = "Softmax" in op_types and any(
                    t in op_types for t in ("MatMul", "Gemm")
                )
            except Exception as e:
                parse_error = f"ONNX graph could not be parsed: {e}"

        return {
            "valid": True,
            "filename": os.path.basename(model_path),
            "framework": framework,
            "size_mb": size_mb,
            # Raw bytes as well as megabytes. Every bundled sample is well under a
            # megabyte, so size_mb alone renders "0 MB" for a file that is really
            # 15 KB. The UI picks the unit from this.
            "size_bytes": size_bytes,
            "status": "Ready for Step 1" if not parse_error else "Loaded (graph unreadable)",
            "layer_count": layer_count,
            "sha256": f"0x{sha256}...",
            "has_bottleneck": has_bottleneck,
            "error": parse_error,
        }

    def verify_toolchain_configuration(self, target_name: str) -> dict[str, Any]:
        """Actually probe GCC and QEMU for the requested target."""
        try:
            from tatva.compiler import TARGETS
            from tatva.runner import find_qemu, find_riscv_gcc

            variant = TARGETS.get(target_name)
            if variant is None:
                return {
                    "status": "error",
                    "error": f"Unknown target '{target_name}'. Known targets: {', '.join(sorted(TARGETS))}.",
                }

            missing = []
            _, gcc_path = find_riscv_gcc()
            if not gcc_path:
                missing.append("RISC-V GCC")

            bitness = 32 if "32" in variant.name else 64
            _, qemu_path = find_qemu(bitness)
            if not qemu_path:
                missing.append(f"qemu-system-riscv{bitness}")

            if missing:
                return {
                    "status": "error",
                    "error": f"Missing toolchain component(s): {', '.join(missing)}. Run 'tatva doctor' for details.",
                }

            return {"status": "ok", "error": "", "gcc_path": gcc_path, "qemu_path": qemu_path}
        except Exception as e:
            return {"status": "error", "error": f"Toolchain verification failed: {e}"}

    def scan_hardware_boards(self) -> dict[str, Any]:
        return {"found": False, "status": "SIMULATION MODE — QEMU results only"}

    def analyze_model(self, model_path: str) -> dict[str, Any]:
        if not os.path.exists(model_path):
            return {"error": f"File not found: {model_path}"}
        try:
            from tatva.compiler import analyze_graph, import_model
            ir = import_model(model_path)
            stats = analyze_graph(ir)
            return {
                "filename": os.path.basename(model_path),
                "size_kb": round(os.path.getsize(model_path) / 1024.0, 1),
                "total_ops": getattr(stats, "total_ops", 0),
                "op_histogram": getattr(stats, "op_histogram", {}),
                "has_transformer_bottleneck": getattr(stats, "has_transformer_bottleneck", False),
            }
        except Exception as e:
            return {
                "filename": os.path.basename(model_path),
                "error": f"Analysis failed: {e}",
                "diagnosis": _diagnose(e),
            }

    def map_operators(self, model_path: str, target_name: str) -> dict[str, Any]:
        """
        Stage 03 (MAP): match every operator in the graph against the chosen chip.

        This is the stage the GUI previously had no screen for at all -- the numbering
        jumped from 2 to 4. It answers one question per operator: is there a lowering
        for it on this target, what will that lowering actually be, and if there isn't
        one, why not and can TATVA do anything about it. Nothing here is estimated; the
        operator list comes from the imported Relax module and every verdict comes from
        the capability database, which is built on the same SUPPORTED_OPS set the
        compiler enforces.

        The per-operator descriptions used to be a chain of substring tests written
        inline here ("softmax" in op -> fused kernel). That put the UI's account of the
        backend in a different file from the backend, with nothing keeping them in step.
        They now come from `tatva.capabilities`, which is also what the CLI reads.
        """
        if not model_path or not os.path.exists(model_path):
            return {"success": False, "error": f"Model file not found: '{model_path}'"}

        try:
            from tatva.capabilities import capability_for
            from tatva.compiler import TARGETS, analyze_graph, import_model

            variant = TARGETS.get(target_name)
            if variant is None:
                return {
                    "success": False,
                    "error": f"Unknown target '{target_name}'. Known targets: {', '.join(sorted(TARGETS))}.",
                }

            report = analyze_graph(import_model(model_path))
            histogram = getattr(report, "op_histogram", {}) or {}

            operators = []
            for op, count in sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0])):
                cap = capability_for(op, variant, count=count)
                row = cap.to_json()
                row["count"] = count
                operators.append(row)

            unsupported = [o["op"] for o in operators if not o["supported"]]
            fixable = [o["op"] for o in operators if o.get("auto_fix_available")]
            blocking = [o for o in unsupported if o not in fixable]

            # Target-level caveats. These are properties of the chip/codegen pairing, not
            # of the model, and they are the reason a "supported" verdict is not the whole
            # story -- RV64GCV maps everything and still emits scalar code.
            warnings: list[str] = []
            if variant.experimental:
                warnings.append(f"{variant.name} is experimental: {variant.notes}")
            if "v" in variant.gcc_march.rsplit("rv64", 1)[-1] or variant.gcc_march.endswith("v"):
                warnings.append(
                    "Vector extension is enabled in the ABI but not targeted by code "
                    "generation. Expect the same cycle count as the non-vector target."
                )
            if variant.bitness == 32:
                warnings.append(
                    "32-bit target: double-precision tensors are emulated in software. "
                    "Prefer FP32 models."
                )
            if getattr(report, "has_transformer_bottleneck", False):
                warnings.append(
                    "Attention pattern detected (softmax feeding matmul). This is the "
                    "subgraph the fusion pass in stage 04 rewrites."
                )

            return {
                "success": True,
                "error": "",
                "target": variant.name,
                "march": variant.gcc_march,
                "mabi": variant.gcc_mabi,
                "bitness": variant.bitness,
                "experimental": variant.experimental,
                "notes": variant.notes,
                "total_ops": getattr(report, "total_ops", 0),
                "distinct_ops": len(operators),
                "operators": operators,
                "unsupported": unsupported,
                # Operators with no lowering that a validated rewrite can express, and
                # those that nothing can. The distinction is what decides whether the
                # Attempt Auto-Fix control does anything, so the frontend is told rather
                # than left to infer it.
                "fixable": fixable,
                "blocking": blocking,
                "auto_fix_available": bool(fixable),
                "warnings": warnings,
                "ready": not unsupported,
            }
        except Exception as e:
            return {"success": False, "error": f"Operator mapping failed: {e}", "diagnosis": _diagnose(e)}

    def attempt_auto_fix(self, model_path: str, target_name: str) -> dict[str, Any]:
        """
        Run the graph-repair engine over a model and report exactly what it did.

        This is the backend behind the Attempt Auto-Fix control on the mapping table. It
        rewrites the unsupported operators it has exact rules for, then validates the
        result twice -- structurally, and by executing the original and rewritten graphs
        on the host and comparing outputs. A rewrite that fails either check is discarded
        and the original graph is kept.

        Three outcomes are possible and all three are reported plainly: the graph is
        fully mapped now, part of it is and the rest still blocks code generation, or
        nothing could be rewritten. The repaired module is held for the subsequent build
        so the pipeline compiles the graph the user was shown, not the original.
        """
        if not model_path or not os.path.exists(model_path):
            return {"success": False, "error": f"Model file not found: '{model_path}'"}

        try:
            from tatva.compiler import TARGETS, analyze_graph, import_model
            from tatva.repair import repair_graph

            variant = TARGETS.get(target_name)
            if variant is None:
                return {
                    "success": False,
                    "error": f"Unknown target '{target_name}'. Known targets: {', '.join(sorted(TARGETS))}.",
                }

            before = analyze_graph(import_model(model_path))
            result = repair_graph(import_model(model_path), variant)

            payload = result.to_json()
            payload.update({"success": True, "error": "", "target": variant.name})

            if result.applied:
                # Hold the repaired module so run_pipeline compiles it. Keyed on the
                # model and target together: a repair is validated against one target's
                # operator set and must not leak to another.
                with self._repair_lock:
                    self._repaired[(os.path.abspath(model_path), variant.name)] = result

                after = analyze_graph(result.model_ir)
                payload["before"] = {
                    "total_ops": before.total_ops,
                    "distinct_ops": len(before.op_histogram),
                    "unsupported": sorted(before.unsupported_ops),
                }
                payload["after"] = {
                    "total_ops": after.total_ops,
                    "distinct_ops": len(after.op_histogram),
                    "unsupported": sorted(after.unsupported_ops),
                }

            payload["ready"] = result.applied and not result.remaining_unsupported
            return payload
        except Exception as e:
            return {"success": False, "error": f"Auto-fix failed: {e}", "diagnosis": _diagnose(e)}

    def get_target_capabilities(self, target_name: str) -> dict[str, Any]:
        """
        The full operator table for a target, so the studio can show what the chip
        supports without a model loaded. The page holds no operator list of its own.
        """
        try:
            from tatva.capabilities import capability_table, repairable_ops
            from tatva.compiler import TARGETS

            variant = TARGETS.get(target_name)
            if variant is None:
                return {"success": False, "error": f"Unknown target '{target_name}'."}
            return {
                "success": True,
                "error": "",
                "target": variant.name,
                "march": variant.gcc_march,
                "mabi": variant.gcc_mabi,
                "experimental": variant.experimental,
                "notes": variant.notes,
                "operators": capability_table(variant),
                "repairable": repairable_ops(),
            }
        except Exception as e:
            return {"success": False, "error": f"Capability lookup failed: {e}"}

    def get_model_formats(self) -> dict[str, Any]:
        """
        Which model formats TATVA can take, resolved against this installation.

        Statuses are computed, not declared: each format is checked for whether TATVA
        implements the adapter, whether the installed TVM carries the frontend, and
        whether the framework's package is importable. Anything short of all three
        reports Coming Soon with the specific reason.
        """
        try:
            from tatva.frontends import format_table

            return {"success": True, "error": "", "formats": format_table()}
        except Exception as e:
            return {"success": False, "error": f"Format registry unavailable: {e}", "formats": []}

    def import_model_info(self, model_path: str) -> dict[str, Any]:
        """
        Describe a model the moment it is selected -- framework, format, size, parameter
        count, input and output shapes, precision, operator count and detected family.

        Reads the file directly and does not import it into TVM, so it is fast enough to
        run on selection. Every field is read out of the model; the family classifier is
        allowed to answer "Unable to determine" rather than guess.
        """
        try:
            from tatva.frontends import inspect_model

            info = inspect_model(model_path)
            payload = info.to_json()
            payload["success"] = info.ok
            return payload
        except Exception as e:
            return {"success": False, "ok": False, "error": f"Could not inspect the model: {e}"}

    def interpret_config(
        self,
        text: str,
        target_name: str = "",
        fuse_softmax: bool = True,
        do_quantize: bool = False,
    ) -> dict[str, Any]:
        """
        Resolve a plain-English priority into a build configuration.

        Deterministic and offline -- see nl_config for why that matters. This only ever
        proposes the same switches the checkboxes set, and it returns the reasoning so
        the front end can show what it decided and why before anything is built.
        """
        try:
            from tatva.compiler import DEFAULT_TARGET
            from tatva.nl_config import interpret, summarise

            intent = interpret(
                text,
                target=target_name or DEFAULT_TARGET,
                fuse=bool(fuse_softmax),
                quantize=bool(do_quantize),
            )
            return {
                "success": True,
                "error": "",
                "target": intent.target,
                "fuse": intent.fuse,
                "quantize": intent.quantize,
                "accuracy_tolerance": intent.accuracy_tolerance,
                "reasons": intent.reasons,
                "conflicts": intent.conflicts,
                "matched": intent.matched,
                "summary": summarise(intent),
            }
        except Exception as e:
            return {"success": False, "error": f"Could not read that as a configuration: {e}"}

    def run_pipeline(
        self,
        model_path: str,
        target_name: str,
        fuse_softmax: bool = True,
        do_quantize: bool = False,
        auto_fix: bool = True,
    ) -> dict[str, Any]:
        """
        Compile and measure BOTH the baseline and the optimized configuration under QEMU.

        Every number returned here comes from an actual emulated run. There is no
        estimation path and no simulated fallback: if the pipeline cannot run, this
        reports the failure instead of inventing a result.

        The run is now recorded as it happens. Analysis, mapping, repair, artifacts,
        validation, effort and the audit trail are all written to a `RunRecord` under a
        `run_id`, which the Artifacts, Validation and Evidence pages read afterwards
        without recompiling. The overall status -- SUCCESS, PARTIAL, BLOCKED or FAILED --
        is derived in one place from what the stages recorded, so a build that stopped
        cannot present itself as one that finished.
        """
        if not model_path or not os.path.exists(model_path):
            return {"success": False, "status": "FAILED", "error": f"Model file not found: '{model_path}'"}

        passes: list[str] = []
        if fuse_softmax:
            passes.append("fuse")
        if do_quantize:
            passes.append("quantize")

        from tatva import artifacts as artifacts_mod
        from tatva import audit as audit_mod
        from tatva import effort as effort_mod
        from tatva import runs, validation
        from tatva.compiler import TARGETS

        variant = TARGETS.get(target_name)
        if variant is None:
            return {
                "success": False,
                "status": "FAILED",
                "error": f"Unknown target '{target_name}'. Known targets: {', '.join(sorted(TARGETS))}.",
            }

        record = runs.REGISTRY.new_run(
            model_path=os.path.abspath(model_path),
            model_name=os.path.basename(model_path),
            target_name=variant.name,
            march=variant.gcc_march,
            mabi=variant.gcc_mabi,
            passes=list(passes),
        )
        trail = record.trail
        trail.info("Run", "Pipeline started", detail=f"{record.model_name} -> {variant.name}", evidence={
            "model": record.model_path, "target": variant.name, "march": variant.gcc_march,
            "passes": passes, "auto_fix": bool(auto_fix),
        })

        try:
            return self._run_pipeline_inner(record, variant, passes, auto_fix,
                                            artifacts_mod, audit_mod, effort_mod, runs, validation)
        except Exception as e:
            # The beta scope calls plain-English diagnostics the key differentiator, and
            # this is the one place a user actually meets a compiler failure. Returning
            # str(e) here handed them the raw exception -- "Memory limit exceeded: 720896
            # bytes" -- which is exactly the raw-compiler-error experience the rule engine
            # in diagnostics.py exists to replace. The CLI (`tatva diagnose`) and the
            # legacy Tk front end already route through it; the web GUI did not.
            diagnosis = _diagnose(e)
            record.error = f"Pipeline failed: {e}"
            record.diagnosis = diagnosis
            trail.error("Run", "Pipeline failed", detail=str(e), evidence={"exception": type(e).__name__})
            runs.REGISTRY.finish(record)
            return {
                "success": False,
                "measured": False,
                "run_id": record.run_id,
                "error": record.error,
                "diagnosis": diagnosis,
                "config_digest": (
                    f"=== TATVA RISC-V OPTIMIZATION PIPELINE — FAILED ===\n"
                    f"Target Architecture : {target_name}\n"
                    f"Model File Path     : {model_path}\n"
                    f"Failure             : {e}"
                ),
                "status": record.status,
                "status_reason": record.status_reason,
            }

    def _run_pipeline_inner(
        self, record: Any, variant: Any, passes: list[str], auto_fix: bool,
        artifacts_mod: Any, audit_mod: Any, effort_mod: Any, runs: Any, validation: Any,
    ) -> dict[str, Any]:
        """
        The pipeline body, stage by stage.

        Split out of `run_pipeline` so the failure handler there wraps every stage
        uniformly -- an exception anywhere below lands in one place, gets diagnosed once,
        and is recorded on the run rather than escaping as a bare string to the frontend.
        """
        from tatva.compiler import analyze_graph, import_model
        from tatva.frontends import inspect_model
        from tatva.optimizer import compare_configs

        trail = record.trail
        model_path = record.model_path

        # ---- model description
        info = inspect_model(model_path)
        record.model_info = info.to_json()
        if info.ok:
            trail.ok("Import", "Model inspected", detail=f"{info.format}, {info.parameter_count:,} parameters",
                     evidence={"format": info.format, "parameters": info.parameter_count,
                               "precision": info.precision, "onnx_ops": info.op_count,
                               "family": info.family.get("family", "")})

        # ---- graph import and analysis
        model_ir = import_model(model_path)
        report = analyze_graph(model_ir)
        record.analysis = {
            "total_ops": report.total_ops,
            "distinct_ops": len(report.op_histogram),
            "op_histogram": dict(report.op_histogram),
            "unsupported_ops": sorted(report.unsupported_ops),
            "has_transformer_bottleneck": report.has_transformer_bottleneck,
        }
        trail.ok("Analyze", "Graph imported and analysed",
                 detail=f"{report.total_ops} operator calls, {len(report.op_histogram)} distinct kinds",
                 evidence={"total_ops": report.total_ops, "distinct": len(report.op_histogram),
                           "unsupported": sorted(report.unsupported_ops)})

        # ---- mapping against the target
        record.mapping = self.map_operators(model_path, variant.name)
        unsupported = sorted(report.unsupported_ops)
        trail.record("Map", "Operators mapped against the target",
                     outcome="ok" if not unsupported else "warn",
                     detail=(f"{len(unsupported)} operator kind(s) have no lowering" if unsupported
                             else "Every operator has a lowering"),
                     evidence={"unsupported": unsupported,
                               "fixable": record.mapping.get("fixable", []),
                               "blocking": record.mapping.get("blocking", [])})

        # ---- repair, if the graph needs it and the user has not opted out
        build_ir = None
        if unsupported:
            key = (model_path, variant.name)
            with self._repair_lock:
                cached = self._repaired.get(key)

            result = cached
            if result is None and auto_fix:
                from tatva.repair import repair_graph

                trail.info("Repair", "Attempting graph repair",
                           detail=f"{len(unsupported)} unsupported operator kind(s)",
                           evidence={"unsupported": unsupported})
                result = repair_graph(import_model(model_path), variant)
                if result.applied:
                    with self._repair_lock:
                        self._repaired[key] = result
            elif result is not None:
                trail.info("Repair", "Reusing the repair from Attempt Auto-Fix",
                           detail=result.message, evidence={"repaired_ops": result.repaired_ops})

            if result is not None:
                record.repair = result.to_json()
                trail.record(
                    "Repair", f"Repair {result.status.lower()}",
                    outcome=("ok" if result.status == "REPAIRED" else
                             "warn" if result.status == "PARTIAL" else "blocked"),
                    detail=result.message,
                    evidence={
                        "repaired_ops": result.repaired_ops,
                        "remaining_unsupported": result.remaining_unsupported,
                        "structural_validation": result.structural_validation,
                        "numerical_validation": result.numerical_validation,
                        "max_abs_diff": result.max_abs_diff,
                    },
                )
                if result.applied:
                    build_ir = result.model_ir
                    # Carried so the optimizer's cache key can tell a repaired build from
                    # an unrepaired one for the same file.
                    build_ir.metadata["repaired_ops"] = list(result.repaired_ops)
                remaining = result.remaining_unsupported
            else:
                remaining = unsupported
                trail.blocked("Repair", "Automatic repair not attempted",
                              detail="Auto-fix is off for this run.", evidence={"unsupported": unsupported})

            if remaining:
                # Nothing further is honest here: the backend has no kernel for these and
                # no validated rewrite. Stop, name them, and do not report a latency.
                trail.blocked("Codegen", "Code generation not started",
                              detail=f"{len(remaining)} operator kind(s) still have no lowering.",
                              evidence={"remaining": remaining})
                record.validation = validation.evaluate(self._validation_input(record)).to_json()
                record.effort = effort_mod.compute(self._effort_input(record)).to_json()
                runs.REGISTRY.finish(record)
                return self._pipeline_payload(record, measured=False)

        # ---- build and measure
        configs = ["baseline"] + (["optimized"] if passes else [])
        trail.info("Build", "Compiling and measuring",
                   detail=f"configurations: {', '.join(configs)}",
                   evidence={"configs": configs, "passes": passes,
                             "graph": "repaired" if build_ir is not None else "original"})

        res = compare_configs(model_path, variant, configs, passes=passes, model_ir=build_ir)
        comp = res["comparison"]
        record.build_dirs = {name: data.get("build_dir", "") for name, data in res["results"].items()}

        base_ms = res["results"]["baseline"]["latency"].mean_ms
        has_optimized = "optimized" in res["results"]
        if has_optimized:
            opt_ms = comp["opt_mean_ms"]
            mse = comp["opt_accuracy_delta_mse"]
            accuracy_ok = comp["opt_accuracy_ok"]
        else:
            # No optimization pass was selected. The baseline is still compared against
            # the host reference -- this used to report mse=0.0 and accuracy_ok=True,
            # which announced a parity check that had not been performed.
            opt_ms = base_ms
            mse = comp.get("baseline_accuracy_delta_mse", float("nan"))
            accuracy_ok = bool(comp.get("baseline_accuracy_ok", False))

        speedup = round(((base_ms - opt_ms) / base_ms) * 100.0, 2) if base_ms else 0.0

        trail.ok("Measure", "Emulated run completed",
                 detail=f"baseline {base_ms:.4f} ms" + (f", optimized {opt_ms:.4f} ms" if has_optimized else ""),
                 evidence={"baseline_ms": base_ms, "optimized_ms": opt_ms if has_optimized else None,
                           "speedup_pct": speedup, "environment": "QEMU_SIM",
                           "build_dirs": record.build_dirs})

        record.benchmark = {
            "measured": True,
            "has_optimized": has_optimized,
            "base_ms": round(base_ms, 4),
            "opt_ms": round(opt_ms, 4) if has_optimized else None,
            "speedup_pct": speedup,
            "mse": mse,
            "accuracy_ok": accuracy_ok,
            "accuracy_reference": comp.get("accuracy_reference", "host onnxruntime"),
            "tolerance": comp.get("tolerance", 0.05),
            "environment": "QEMU_SIM",
            "nominal_clock_mhz": 100,
            "baseline_median_ms": comp.get("baseline_median_ms"),
            "baseline_p95_ms": comp.get("baseline_p95_ms"),
            "opt_median_ms": comp.get("opt_median_ms"),
            "opt_p95_ms": comp.get("opt_p95_ms"),
            "reference_logits": comp.get("reference_logits", []),
        }

        # ---- artifacts actually on disk
        manifest = artifacts_mod.build_manifest(
            run_id=record.run_id, model_path=model_path, target_name=variant.name,
            march=variant.gcc_march, mabi=variant.gcc_mabi, configs=record.build_dirs,
            passes=passes, repaired_ops=(record.repair or {}).get("repaired_ops", []),
            # Both land in the same directory after this manifest is written: the effort
            # estimate needs the file counts this manifest produces, and the audit trail
            # is only complete once every stage below has recorded its outcome.
            written_after=[effort_mod.EFFORT_FILENAME, audit_mod.AUDIT_FILENAME],
        )
        record.manifest = manifest
        for build_dir in record.build_dirs.values():
            artifacts_mod.write_manifest(manifest, build_dir)
        totals = manifest.get("totals", {})
        trail.ok("Artifacts", "Generated files enumerated",
                 detail=f"{totals.get('file_count', 0)} file(s), {totals.get('total_bytes', 0):,} bytes",
                 evidence=totals)

        # ---- optimization history, validation, effort
        record.optimization_history = self._optimization_history(record, comp, passes)
        record.validation = validation.evaluate(self._validation_input(record)).to_json()
        trail.record("Validate", f"Validation {record.validation.get('verdict', '').lower()}",
                     outcome=("ok" if record.validation.get("verdict") in ("PASSED", "PARTIAL") else "error"),
                     detail=record.validation.get("summary", ""),
                     evidence={k: record.validation.get(k) for k in
                               ("passed", "failed", "skipped", "not_implemented")})

        effort_result = effort_mod.compute(self._effort_input(record))
        record.effort = effort_result.to_json()
        for build_dir in record.build_dirs.values():
            effort_mod.write_effort(effort_result, build_dir, run_id=record.run_id)
        trail.record("Effort", "Engineering-effort estimate",
                     outcome="ok" if effort_result.available else "warn",
                     detail=(f"{effort_result.total_hours} h ({effort_result.total_days} days), "
                             f"model {effort_result.model_version}" if effort_result.available
                             else effort_result.reason),
                     evidence=effort_result.inputs)

        runs.REGISTRY.finish(record)
        for build_dir in record.build_dirs.values():
            trail.write(build_dir)

        return self._pipeline_payload(record, measured=True)

    def _validation_input(self, record: Any) -> dict[str, Any]:
        """Flatten a run into the shape `validation.evaluate` reads. One place, so the
        validation page and the report cannot be looking at different runs."""
        from tatva import artifacts as artifacts_mod

        stats = artifacts_mod.generated_source_stats(record.build_dirs)
        bench = record.benchmark or {}
        return {
            "analysis": record.analysis,
            "unsupported_ops": ((record.repair or {}).get("remaining_unsupported")
                                if record.repair else (record.analysis or {}).get("unsupported_ops", [])),
            "target_name": record.target_name,
            "repair": record.repair,
            "build_dirs": record.build_dirs,
            "build_attempted": bool(record.build_dirs) or bool(record.benchmark),
            "build_error": record.error,
            "import_error": record.error,
            "generated_files": stats["generated_files"],
            "generated_lines": stats["generated_lines"],
            "measured": bool(bench.get("measured")),
            "environment": bench.get("environment", ""),
            "base_ms": bench.get("base_ms"),
            "nominal_clock_mhz": bench.get("nominal_clock_mhz", 100),
            "parity_applicable": bool(bench.get("measured")),
            "accuracy_ok": bench.get("accuracy_ok"),
            "accuracy_reference": bench.get("accuracy_reference", "host ONNX Runtime"),
            "mse": bench.get("mse"),
            "tolerance": bench.get("tolerance"),
        }

    def _effort_input(self, record: Any) -> Any:
        """Counted quantities for the effort model -- all of them read from the run."""
        from tatva import artifacts as artifacts_mod
        from tatva.effort import EffortInputs

        stats = artifacts_mod.generated_source_stats(record.build_dirs)
        analysis = record.analysis or {}
        return EffortInputs(
            distinct_operator_kinds=int(analysis.get("distinct_ops") or 0),
            total_operator_calls=int(analysis.get("total_ops") or 0),
            generated_files=stats["generated_files"],
            generated_lines=stats["generated_lines"],
            repaired_op_kinds=len((record.repair or {}).get("repaired_ops") or []),
            build_configs=len(record.build_dirs),
            benchmark_runs=len(record.build_dirs) if (record.benchmark or {}).get("measured") else 0,
        )

    def _optimization_history(self, record: Any, comp: dict[str, Any], passes: list[str]) -> list[dict[str, Any]]:
        """
        What each transformation changed, why, and how it was checked.

        The measured effect is reported once, for the passes as a group, because that is
        how they were measured: `compare_configs` builds one optimized configuration with
        every selected pass applied. Splitting one measurement into a per-pass attribution
        would be arithmetic on a number that was never collected that way.
        """
        history: list[dict[str, Any]] = []

        for rec in (record.repair or {}).get("records", []):
            history.append({
                "stage": "Graph repair",
                "name": f"Rewrite {rec['original_op']}",
                "before": f"{rec['original_op']} x{rec['occurrences']} (no lowering on {record.target_name})",
                "after": " -> ".join(rec["replacement_ops"]) or "(unchanged)",
                "reason": rec.get("reason", ""),
                "impact": "Unblocked code generation for this operator." if rec.get("mapping_result") == "MAPPED"
                          else "Rewritten, but the result is still unmapped.",
                "validation": _validation_label(rec.get("structural_validation"), rec.get("numerical_validation")),
                "detail": rec.get("identity", ""),
            })

        skipped_note = comp.get("opt_optimization_skipped", "")
        if "fuse" in passes:
            applied = not skipped_note
            history.append({
                "stage": "Optimization",
                "name": "Softmax kernel selection",
                "before": "TVM's default softmax kernel (exact expf per element).",
                "after": ("Single-pass kernel using a Schraudolph fast exponential." if applied
                          else "Unchanged."),
                "reason": ("Softmax feeding matmul is the dominant transcendental cost on a scalar target."
                           if applied else skipped_note),
                "impact": "Measured together with the other selected passes; see below." if applied
                          else "No effect: the pass did not apply to this graph.",
                "validation": ("Covered by the end-to-end parity check against host ONNX Runtime."
                               if applied else "Not applicable."),
                "detail": skipped_note,
            })

        if "quantize" in passes:
            history.append({
                "stage": "Optimization",
                "name": "INT8 quantization",
                "before": "FP32 weights and activations.",
                "after": "Values round-tripped through INT8, then computed in FP32.",
                "reason": (
                    "Measures what INT8 would cost in accuracy. It does not reduce "
                    "footprint or latency: the weights stay FP32 on device and the "
                    "matmuls stay FP32, so this pass only adds work."
                ),
                "impact": "Measured together with the other selected passes; see below.",
                "validation": "Covered by the end-to-end parity check against host ONNX Runtime.",
                "detail": "Activation scale calibrated on the host at the p99.9 percentile.",
            })

        bench = record.benchmark or {}
        if bench.get("measured") and bench.get("has_optimized"):
            delta = bench["base_ms"] - (bench["opt_ms"] or 0.0)
            history.append({
                "stage": "Measurement",
                "name": f"Combined effect of: {', '.join(passes)}",
                "before": f"{bench['base_ms']} ms (baseline, no passes)",
                "after": f"{bench['opt_ms']} ms",
                "reason": "Both configurations were built and run under the same emulator settings.",
                "impact": f"{delta:+.4f} ms ({bench['speedup_pct']:+.2f}%) under QEMU at a nominal 100 MHz.",
                "validation": (f"Parity MSE {bench['mse']} against a tolerance of {bench['tolerance']} "
                               f"({'within' if bench['accuracy_ok'] else 'outside'} tolerance)."),
                "detail": "Emulated cycle counts, not silicon.",
            })

        return history

    def _pipeline_payload(self, record: Any, *, measured: bool) -> dict[str, Any]:
        """
        The object the frontend receives.

        Carries the legacy field names the report page already reads, plus the run id
        and the derived status. Nothing here recomputes a number: every value is copied
        from what the stages recorded.
        """
        bench = record.benchmark or {}
        passes = record.passes

        if measured:
            digest = (
                f"=== TATVA RISC-V OPTIMIZATION PIPELINE EXECUTION ===\n"
                f"Run ID              : {record.run_id}\n"
                f"Target Architecture : {record.target_name} ({record.march})\n"
                f"Model File Path     : {record.model_path}\n"
                f"Softmax Fusion Pass : {'ENABLED' if 'fuse' in passes else 'DISABLED'}\n"
                f"INT8 Quant Pass     : {'ENABLED' if 'quantize' in passes else 'DISABLED'}\n"
                f"Graph Repair        : "
                + (", ".join((record.repair or {}).get("repaired_ops", [])) or "not required") + "\n"
                "Compiler Backend    : TVM Relax -> C -> riscv-none-elf-gcc\n"
                "Measurement         : QEMU system-mode, rdcycle, -icount shift=0"
            )
        else:
            digest = (
                f"=== TATVA RISC-V OPTIMIZATION PIPELINE — {record.status} ===\n"
                f"Run ID              : {record.run_id}\n"
                f"Target Architecture : {record.target_name} ({record.march})\n"
                f"Model File Path     : {record.model_path}\n"
                f"Reason              : {record.status_reason}"
            )

        diagnosis = record.diagnosis
        if measured and not bench.get("accuracy_ok"):
            # A run that completes but lands outside the parity tolerance is a failure the
            # user has to act on, and a bare status line does not say what to do about it.
            # compare_configs returns rather than raises here, so build the same context
            # classify_failure would have produced from an AccuracyDropError.
            from tatva.diagnostics import AccuracyDropError
            diagnosis = _diagnose(AccuracyDropError(mse=float(bench.get("mse") or 0.0),
                                                    tolerance=float(bench.get("tolerance") or 0.05)))

        payload: dict[str, Any] = {
            "success": measured,
            "run_id": record.run_id,
            "status": record.status,
            "status_reason": record.status_reason,
            "error": record.error,
            "diagnosis": diagnosis,
            "config_digest": digest,
            "measured": measured,
            "has_optimized": bool(bench.get("has_optimized")),
            "repair": record.repair,
            # After a repair, what still blocks is the repair's leftovers -- an empty list
            # there means "nothing is unsupported any more", not "fall back to the list
            # from before the repair ran".
            "unsupported": ((record.repair or {}).get("remaining_unsupported")
                            if record.repair else (record.analysis or {}).get("unsupported_ops", [])),
            "validation_verdict": (record.validation or {}).get("verdict", ""),
            "validation_summary": (record.validation or {}).get("summary", ""),
            "effort_available": bool((record.effort or {}).get("available")),
            "artifact_count": (record.manifest or {}).get("totals", {}).get("file_count", 0),
        }

        if measured:
            # Four decimals, not two. A small model runs in well under a millisecond
            # under QEMU, and rounding to 2 dp turned a real baseline/optimized gap into
            # "0.06 ms" against "0.06 ms" -- two identical bars on the chart for a run
            # that did measure a difference.
            payload.update({
                "base_ms": bench["base_ms"],
                "opt_ms": bench["opt_ms"] if bench.get("has_optimized") else bench["base_ms"],
                "speedup": bench["speedup_pct"],
                "baseline_ms": bench["base_ms"],
                "optimized_ms": bench["opt_ms"] if bench.get("has_optimized") else bench["base_ms"],
                "speedup_pct": bench["speedup_pct"],
                "mse": bench["mse"],
                "accuracy_ok": bench["accuracy_ok"],
                "accuracy_reference": bench["accuracy_reference"],
                "tolerance": bench["tolerance"],
            })
        return payload

    # ------------------------------------------------------------------ run readback
    #
    # Everything below reads a finished run out of the registry. They exist so the
    # Artifacts, Validation, Evidence and Effort pages do not have to hold the whole
    # result in the page and hand it back, and so none of them recomputes anything.

    def get_run(self, run_id: str = "") -> dict[str, Any]:
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first."}
        return {"success": True, "error": "", **record.to_json()}

    def list_runs(self, limit: int = 10) -> dict[str, Any]:
        from tatva import runs

        return {"success": True, "error": "", "runs": runs.REGISTRY.recent(limit)}

    def get_artifacts(self, run_id: str = "") -> dict[str, Any]:
        """
        The files a run actually produced, read back off disk.

        Re-enumerating rather than replaying the manifest is deliberate: it means the
        page cannot show a file that has since been deleted, and it picks up anything
        the build wrote that the manifest was assembled before.
        """
        from tatva import artifacts as artifacts_mod
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first.", "builds": []}
        if not record.build_dirs:
            return {
                "success": True, "error": "", "run_id": record.run_id, "builds": [],
                "totals": {"file_count": 0, "total_bytes": 0},
                "reason": record.status_reason or "This run produced no files.",
                **record.summary(),
            }

        builds = []
        for config_name, build_dir in record.build_dirs.items():
            found = artifacts_mod.discover(build_dir)
            builds.append({
                "config": config_name,
                "build_dir": build_dir,
                "exists": os.path.isdir(build_dir),
                "file_count": len(found),
                "total_bytes": sum(a.size_bytes for a in found),
                "artifacts": [a.to_json() for a in found],
            })
        return {
            "success": True, "error": "", "builds": builds,
            "totals": {
                "file_count": sum(b["file_count"] for b in builds),
                "total_bytes": sum(b["total_bytes"] for b in builds),
            },
            "manifest_filename": artifacts_mod.MANIFEST_FILENAME,
            **record.summary(),
        }

    def read_artifact(self, path: str, max_bytes: int = 200_000) -> dict[str, Any]:
        """
        Read one generated file for the in-app viewer.

        Restricted to files inside a build directory of a recorded run -- the frontend
        passes back a path this bridge handed it, and this makes that the only thing it
        can ask for. Binary files are not decoded; their size is reported instead.
        """
        from tatva import runs

        target = os.path.abspath(path or "")
        allowed = any(
            target.startswith(os.path.abspath(d) + os.sep)
            for record in [runs.REGISTRY.get(rid["run_id"]) for rid in runs.REGISTRY.recent(runs.MAX_RUNS)]
            if record for d in record.build_dirs.values() if d
        )
        if not allowed:
            return {"success": False, "error": "That file is not part of a build from this session."}
        if not os.path.isfile(target):
            return {"success": False, "error": "The file is no longer on disk."}

        size = os.path.getsize(target)
        if os.path.splitext(target)[1].lower() in (".elf", ".bin", ".o"):
            return {
                "success": True, "error": "", "binary": True, "path": target,
                "size_bytes": size, "content": "",
                "note": "Binary file. Size and hash are listed above; TATVA does not disassemble it in-app.",
            }
        try:
            with open(target, encoding="utf-8", errors="replace") as fh:
                content = fh.read(max_bytes)
        except OSError as e:
            return {"success": False, "error": f"Could not read the file: {e}"}
        return {
            "success": True, "error": "", "binary": False, "path": target, "size_bytes": size,
            "content": content, "truncated": size > max_bytes,
        }

    def get_validation(self, run_id: str = "") -> dict[str, Any]:
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first."}
        if not record.validation:
            return {"success": True, "error": "", "checks": [], "verdict": "NOT_RUN",
                    "summary": "Validation has not run for this build.", **record.summary()}
        return {"success": True, "error": "", **record.validation, **record.summary()}

    def get_effort(self, run_id: str = "") -> dict[str, Any]:
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first."}
        if not record.effort:
            return {"success": True, "error": "", "available": False,
                    "reason": "No estimate: this run produced nothing to count.", **record.summary()}
        return {"success": True, "error": "", **record.effort, **record.summary()}

    def get_optimization_history(self, run_id: str = "") -> dict[str, Any]:
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first."}
        return {"success": True, "error": "", "history": record.optimization_history, **record.summary()}

    def get_audit_trail(self, run_id: str = "") -> dict[str, Any]:
        from tatva import runs

        record = runs.REGISTRY.get(run_id)
        if record is None:
            return {"success": False, "error": "No run to show yet. Compile a model first."}
        return {
            "success": True, "error": "",
            **(record.trail.to_json() if record.trail else {"events": [], "counts": {}}),
            **record.summary(),
        }

    def run_autonomous_loop(self, prompt_text: str, target_name: str, model_name: str) -> dict[str, Any]:
        try:
            from scaffolding.loop_agent import LoopAgent
            agent = LoopAgent()
            return agent.run_autonomous_loop(
                prompt_text=prompt_text, target=target_name, model_name=model_name
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Autonomous loop failed: {e}",
                "project_name": "tatva_riscv_antigravity_starter",
                "attempts_used": 0,
                "cumulative_cost_usd": 0.0,
                "files": [],
            }


def launch_gui() -> None:
    """
    Application entry point launching PyWebView Edge WebView2 Desktop UI
    with fallback to Tkinter.
    """
    from tatva.config import load_dotenv_file

    load_dotenv_file()

    if getattr(sys, "frozen", False):
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(base_dir, "website", "index.html")]
    else:
        candidates = [
            # Installed wheel: pyproject force-includes `website` at `tatva/website`.
            # This case was missing, so `pip install tatva-compiler && tatva gui` opened
            # a file:// URL that did not exist and rendered a blank window.
            os.path.join(_CURRENT_DIR, "website", "index.html"),
            # Source checkout.
            os.path.join(_CURRENT_DIR, "..", "..", "website", "index.html"),
            os.path.join(_CURRENT_DIR, "..", "website", "index.html"),
        ]

    html_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            "Could not find the GUI's index.html. Looked in:\n  " + "\n  ".join(os.path.abspath(p) for p in candidates)
        )

    _normalized_html_path = os.path.abspath(html_path).replace(os.sep, "/")
    target_url = f"file:///{_normalized_html_path}"

    try:
        import webview
        bridge = TatvaPyBridge()
        webview.create_window(
            "TATVA — Bare-Metal RISC-V Transformer Optimization Studio",
            url=target_url,
            js_api=bridge,
            width=1380,
            height=880,
            min_size=(1080, 720),
            resizable=True,
        )
        webview.start()
    except Exception as e:
        print(f"PyWebView launch note ({e}), falling back to Tkinter...")
        app = TatvaApp()
        app.withdraw()

        def on_loaded(success: bool, err: str) -> None:
            if success:
                app.deiconify()
                app.lift()
                app.focus_force()
            else:
                messagebox.showerror("Initialization Error", f"Failed to load backend libraries:\n{err}")
                app.destroy()

        SplashWindow(app, on_loaded)
        app.mainloop()


def _main() -> None:
    """
    Entry point for the frozen TATVA.exe.

    The packaged build runs windowed, with no console attached, so an exception on the
    way up has nowhere to print to and the app just vanishes from the taskbar with no
    explanation. Catching it here means a failed launch produces a dialog the user can
    read and a log file they can send on, instead of silence.
    """
    try:
        launch_gui()
    except Exception:
        import traceback

        detail = traceback.format_exc()

        log_path = ""
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".tatva")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "startup-error.log")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(detail)
        except Exception:
            log_path = ""

        message = (
            "TATVA could not start.\n\n"
            f"{detail.strip().splitlines()[-1] if detail.strip() else 'Unknown error'}\n\n"
            + (f"Full details were written to:\n{log_path}" if log_path else detail)
        )

        if getattr(sys, "frozen", False):
            with contextlib.suppress(Exception):
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("TATVA — startup failed", message)
                root.destroy()
        print(detail, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    _main()

