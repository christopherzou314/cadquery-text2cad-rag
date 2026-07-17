"""Command line entrypoint for the Text-to-CadQuery prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from .env import load_dotenv
from .agent import generate_execute_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CadQuery Python from text and export STEP/STL files."
    )
    parser.add_argument("description", help="Natural-language CAD description.")
    parser.add_argument("--mock", action="store_true", help="Use offline deterministic demo code.")
    parser.add_argument("--model", help="OpenAI-compatible model name.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL, e.g. https://api.openai.com/v1.")
    parser.add_argument("--api-key", help="API key. Prefer OPENAI_API_KEY for normal use.")
    parser.add_argument("--python", help="Python interpreter with cadquery installed.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for run artifacts.")
    parser.add_argument("--max-repairs", type=int, default=2, help="Number of traceback repair attempts.")
    parser.add_argument("--rag", action="store_true", help="Inject retrieved CadQuery references.")
    parser.add_argument(
        "--rag-mode",
        choices=("off", "lightweight", "full"),
        help="Select no RAG, the original 8-entry library, or the full library.",
    )
    parser.add_argument("--rag-top-k", type=int, default=3, help="Number of references to retrieve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    rag_mode = args.rag_mode or ("full" if args.rag else "off")

    result = generate_execute_repair(
        description=args.description,
        mock=args.mock,
        max_repairs=args.max_repairs,
        output_root=Path(args.output_dir),
        python_executable=args.python,
        model=args.model,
        api_key=args.api_key,
        base_url=args.api_base,
        rag_mode=rag_mode,
        rag_top_k=args.rag_top_k,
    )

    final = result.final_attempt
    if not result.success:
        raise SystemExit(
            "Failed after repair attempts.\n"
            f"Run directory: {result.run_dir}\n"
            f"Last stderr:\n{final.stderr}"
        )

    print("Done.")
    print(f"Run directory: {result.run_dir}")
    print(f"Attempts: {len(result.attempts)}")
    print(f"Generation setting: {result.rag_mode}")
    if result.reference_ids:
        print(f"References: {', '.join(result.reference_ids)}")
    print(f"Generated code: {final.code_path}")
    print(f"STEP: {final.step_path}")
    print(f"STL: {final.stl_path}")


if __name__ == "__main__":
    main()
