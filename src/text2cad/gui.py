"""Tkinter GUI for the Text-to-CadQuery prototype."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from .agent import generate_execute_repair
from .env import load_dotenv
from .renderer import render_stl_to_png


class Text2CadApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        load_dotenv()

        self.title("Text2CAD CadQuery Prototype")
        self.geometry("1180x780")
        self.preview_image: ImageTk.PhotoImage | None = None
        self.latest_run_dir: Path | None = None
        self.latest_code_path: Path | None = None
        self.latest_cq_editor_view_path: Path | None = None
        self.latest_cq_editor_launcher_path: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        ttk.Label(left, text="CAD description").pack(anchor=tk.W)
        self.prompt = scrolledtext.ScrolledText(left, width=48, height=7, wrap=tk.WORD)
        self.prompt.insert(
            "1.0",
            "a 60 mm by 40 mm by 8 mm rectangular plate with four 5 mm corner holes",
        )
        self.prompt.pack(fill=tk.X, pady=(4, 10))

        options = ttk.Frame(left)
        options.pack(fill=tk.X, pady=(0, 10))

        self.mock_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Mock mode (no API)", variable=self.mock_var).pack(anchor=tk.W)

        self.auto_open_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Open 3D in CQ-editor after success",
            variable=self.auto_open_var,
        ).pack(anchor=tk.W)

        ttk.Label(options, text="Knowledge mode").pack(anchor=tk.W, pady=(8, 0))
        self.rag_mode_var = tk.StringVar(value="Baseline (no RAG)")
        self.rag_mode_selector = ttk.Combobox(
            options,
            state="readonly",
            width=34,
            textvariable=self.rag_mode_var,
            values=(
                "Baseline (no RAG)",
                "Lightweight RAG (8 entries)",
                "Full RAG (all entries)",
            ),
        )
        self.rag_mode_selector.pack(anchor=tk.W)

        ttk.Label(options, text="Total repair budget (all failure types)").pack(
            anchor=tk.W, pady=(8, 0)
        )
        self.repairs_var = tk.IntVar(value=2)
        ttk.Spinbox(options, from_=0, to=5, textvariable=self.repairs_var, width=8).pack(anchor=tk.W)

        self.run_button = ttk.Button(left, text="Generate and render", command=self._start_run)
        self.run_button.pack(fill=tk.X, pady=(0, 8))

        self.open_code_button = ttk.Button(left, text="Open 3D in CQ-editor", command=self._open_in_cq_editor)
        self.open_code_button.pack(fill=tk.X, pady=(0, 8))
        self.open_code_button.state(["disabled"])

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status).pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(left, text="Log").pack(anchor=tk.W)
        self.log = scrolledtext.ScrolledText(left, width=48, height=24, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Rendered preview").pack(anchor=tk.W)
        self.preview = ttk.Label(right, anchor=tk.CENTER, background="white")
        self.preview.pack(fill=tk.BOTH, expand=True, pady=(4, 10))

        ttk.Label(right, text="Generated CadQuery code").pack(anchor=tk.W)
        self.code_view = scrolledtext.ScrolledText(right, height=12, wrap=tk.NONE)
        self.code_view.pack(fill=tk.BOTH, expand=False)

    def _start_run(self) -> None:
        description = self.prompt.get("1.0", tk.END).strip()
        if not description:
            messagebox.showwarning("Missing prompt", "Please enter a CAD description.")
            return

        self.run_button.state(["disabled"])
        self.open_code_button.state(["disabled"])
        self.status.set("Running...")
        self._set_log("Starting generation...\n")
        self.code_view.delete("1.0", tk.END)
        self.preview.configure(image="", text="")

        thread = threading.Thread(target=self._run_worker, args=(description,), daemon=True)
        thread.start()

    def _run_worker(self, description: str) -> None:
        try:
            result = generate_execute_repair(
                description,
                mock=self.mock_var.get(),
                max_repairs=self.repairs_var.get(),
                python_executable=os.getenv("CADQUERY_PYTHON") or sys.executable,
                progress_callback=self._ui_progress,
                rag_mode=self._selected_rag_mode(),
            )
            final = result.final_attempt
            if not result.success:
                telemetry = result.telemetry
                self._ui_error(
                    "Generation failed within the total repair budget.\n\n"
                    f"Run directory: {result.run_dir}\n\n{final.stderr}"
                    f"\nExecution repairs: {telemetry['execution_repairs']}"
                    f"\nConstraint repairs: {telemetry['constraint_repairs']}"
                )
                return

            preview_path = final.run_dir / "preview.png"
            render_stl_to_png(final.stl_path, preview_path)
            cq_editor_path = self._write_cq_editor_viewer(final.code_path)
            cq_editor_launcher_path = self._write_cq_editor_launcher(cq_editor_path)
            code = final.code_path.read_text(encoding="utf-8")
            self.latest_run_dir = final.run_dir
            self.latest_code_path = final.code_path
            self.latest_cq_editor_view_path = cq_editor_path
            self.latest_cq_editor_launcher_path = cq_editor_launcher_path
            self._ui_success(result, code, preview_path, cq_editor_path, cq_editor_launcher_path)
        except Exception as exc:
            self._ui_error(str(exc))

    def _ui_success(
        self,
        result,
        code: str,
        preview_path: Path,
        cq_editor_path: Path,
        cq_editor_launcher_path: Path,
    ) -> None:
        def update() -> None:
            final = result.final_attempt
            telemetry = result.telemetry
            usage = telemetry["token_usage"]
            cost = telemetry["cost"]
            timing = telemetry["timing"]
            token_text = usage["total_tokens"] if usage["total_tokens"] is not None else "not returned"
            cost_text = (
                f"{cost['estimated_total']} {cost['currency'] or ''}".rstrip()
                if cost["estimated_total"] is not None
                else "not available"
            )
            self.status.set("Done.")
            self._set_log(
                "Done.\n"
                f"Run directory: {result.run_dir}\n"
                f"Attempts: {len(result.attempts)}\n"
                f"Total repair budget: {telemetry['total_repair_budget']}\n"
                f"Execution repairs: {telemetry['execution_repairs']}\n"
                f"Constraint repairs: {telemetry['constraint_repairs']}\n"
                f"Execution pass: {telemetry['execution_pass_within_repair_budget']}\n"
                f"Constraint pass: {telemetry['constraint_pass_within_repair_budget']}\n"
                f"End-to-end pass: {telemetry['end_to_end_pass_within_repair_budget']}\n"
                f"Setting: {result.rag_mode}\n"
                f"Total tokens: {token_text}\n"
                f"Estimated cost: {cost_text}\n"
                f"LLM time: {timing['llm_seconds']} s\n"
                f"CadQuery execution time: {timing['cad_execution_seconds']} s\n"
                f"End-to-end time: {timing['end_to_end_seconds']} s\n"
                f"References: {', '.join(result.reference_ids) if result.reference_ids else 'none'}\n"
                f"Generated code: {final.code_path}\n"
                f"STEP: {final.step_path}\n"
                f"STL: {final.stl_path}\n"
                f"Preview: {preview_path}\n"
                f"CQ-editor viewer: {cq_editor_path}\n"
                f"CQ-editor autorun launcher: {cq_editor_launcher_path}\n"
            )
            self.code_view.delete("1.0", tk.END)
            self.code_view.insert("1.0", code)
            self._show_preview(preview_path)
            self.run_button.state(["!disabled"])
            self.open_code_button.state(["!disabled"])
            if self.auto_open_var.get():
                self._open_in_cq_editor()

        self.after(0, update)

    def _ui_error(self, message: str) -> None:
        def update() -> None:
            self.status.set("Failed.")
            self._set_log(message)
            self.run_button.state(["!disabled"])

        self.after(0, update)

    def _ui_progress(self, message: str) -> None:
        def update() -> None:
            self.status.set(message)
            self.log.insert(tk.END, message + "\n")
            self.log.see(tk.END)

        self.after(0, update)

    def _selected_rag_mode(self) -> str:
        label = self.rag_mode_var.get()
        if label.startswith("Lightweight"):
            return "lightweight"
        if label.startswith("Full"):
            return "full"
        return "off"

    def _set_log(self, text: str) -> None:
        self.log.delete("1.0", tk.END)
        self.log.insert("1.0", text)

    def _show_preview(self, path: Path) -> None:
        image = Image.open(path)
        image.thumbnail((760, 430))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_image)

    def _write_cq_editor_viewer(self, code_path: Path) -> Path:
        viewer_path = code_path.parent / "cq_editor_view.py"
        viewer_path.write_text(
            f"""from pathlib import Path
import runpy

namespace = runpy.run_path(str(Path({str(code_path)!r})))
result = namespace.get("result")
if result is None:
    for fallback_name in ("model", "part", "assembly"):
        if fallback_name in namespace:
            result = namespace[fallback_name]
            break
if result is None:
    raise ValueError("Generated code did not define `result`.")

show_object(result)
""",
            encoding="utf-8",
        )
        return viewer_path

    def _write_cq_editor_launcher(self, viewer_path: Path) -> Path:
        launcher_path = viewer_path.parent / "launch_cq_editor_autorun.py"
        log_path = viewer_path.parent / "cq_editor_autorun.log"
        launcher_path.write_text(
            f"""import traceback
import sys

from PyQt5.QtCore import QTimer

sys.argv = ["cq-editor-autorun", {str(viewer_path)!r}]

from cq_editor.__main__ import app
from cq_editor.main_window import MainWindow

LOG_PATH = {str(log_path)!r}


def log(message):
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(message + "\\n")


def autorun_render():
    try:
        log("Starting CQ-editor autorun render.")
        win.components["debugger"].render()
        log("CQ-editor autorun render finished.")
    except Exception:
        log(traceback.format_exc())


try:
    app.setQuitOnLastWindowClosed(False)
    win = MainWindow(filename={str(viewer_path)!r})
    win.show()
    win.raise_()
    win.activateWindow()
    QTimer.singleShot(1800, autorun_render)
    log("CQ-editor window opened.")
    app.exec_()
except Exception:
    log(traceback.format_exc())
    raise
""",
            encoding="utf-8",
        )
        return launcher_path

    def _open_in_cq_editor(self) -> None:
        if not self.latest_cq_editor_launcher_path:
            return
        subprocess.Popen(
            [
                os.getenv("CADQUERY_PYTHON") or sys.executable,
                str(self.latest_cq_editor_launcher_path),
            ],
            stdout=(self.latest_run_dir / "cq_editor_stdout.log").open("a", encoding="utf-8")
            if self.latest_run_dir
            else subprocess.DEVNULL,
            stderr=(self.latest_run_dir / "cq_editor_stderr.log").open("a", encoding="utf-8")
            if self.latest_run_dir
            else subprocess.DEVNULL,
        )


def main() -> None:
    app = Text2CadApp()
    app.mainloop()


if __name__ == "__main__":
    main()
