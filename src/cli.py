"""Command-line interface: sqlconsensus {prepare, generate, score, evaluate, run}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.report import evaluate
from src.generation.sampler import DEFAULTS, generate
from src.pipeline import prepare, run
from src.scoring.confidence import AGREE_WEIGHT
from src.scoring.execution import EXECUTION_TIMEOUT
from src.scoring.scorer import score


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    """Options of the generation step with their default values."""
    defaults = DEFAULTS
    parser.add_argument("--num-samples", type=int, default=defaults["num_samples"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--tensor-parallel-size", type=int, default=defaults["tensor_parallel_size"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])


def _generation_options(args: argparse.Namespace) -> dict:
    """Generation options of parsed arguments as keyword arguments."""
    return {key: getattr(args, key) for key in DEFAULTS}


def _parser() -> argparse.ArgumentParser:
    """Parser with one subcommand per experiment step."""
    parser = argparse.ArgumentParser(prog="sqlconsensus")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build benchmark records")
    prepare.add_argument("benchmark", choices=("bird", "spider", "ehrsql"))
    prepare.add_argument("--dataset", type=Path, required=True, help="Question file of the benchmark")
    prepare.add_argument("--db-dir", type=Path, help="Directory of the BIRD or Spider SQLite databases")
    prepare.add_argument("--db-file", type=Path, help="EHRSQL SQLite database")
    prepare.add_argument("--database", choices=("mimic_iii", "eicu"))
    prepare.add_argument("--omnisql-repo", type=Path, help="Clone of the OmniSQL repository with the EHRSQL index")
    prepare.add_argument("--out", type=Path, required=True)

    generate = commands.add_parser("generate", help="Generate predictions, sampled queries and Verify")
    generate.add_argument("--data", type=Path, required=True)
    generate.add_argument("--backend", choices=("vllm", "openai"), required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--out", type=Path, required=True)
    _add_generation_options(generate)

    score = commands.add_parser("score", help="Compute Agree, Consist, Verify and the confidence")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--generations", type=Path, required=True)
    score.add_argument("--agree-weight", type=float, default=AGREE_WEIGHT)
    score.add_argument("--timeout", type=float, default=EXECUTION_TIMEOUT)
    score.add_argument("--ehrsql-repo", type=Path, help="Apply the official EHRSQL SQL rewrite")
    score.add_argument("--out", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Report accuracy, AUROC, R@10%% and R@20%%")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--scored", type=Path, required=True)
    evaluate.add_argument("--ehrsql-repo", type=Path, help="Apply the official EHRSQL SQL rewrite")
    evaluate.add_argument("--out", type=Path)

    run = commands.add_parser("run", help="Generate, score and evaluate one model on one benchmark")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--backend", choices=("vllm", "openai"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--ehrsql-repo", type=Path, help="Apply the official EHRSQL SQL rewrite")
    run.add_argument("--out-dir", type=Path, required=True)
    _add_generation_options(run)
    return parser


def _print_metrics(metrics: dict) -> None:
    """Print metrics rounded to two decimals."""
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()}))


def main(argv: list[str] | None = None) -> None:
    """Entry point of the sqlconsensus command."""
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        count = prepare(args.benchmark, args.out, dataset=args.dataset, db_dir=args.db_dir,
                                 db_file=args.db_file, database=args.database, omnisql_repo=args.omnisql_repo)
        print(f"{count} records -> {args.out}")
    elif args.command == "generate":
        generate(args.data, args.backend, args.model, args.out, **_generation_options(args))
    elif args.command == "score":
        count = score(args.data, args.generations, args.out, agree_weight=args.agree_weight,
                               timeout=args.timeout, ehrsql_repo=args.ehrsql_repo)
        print(f"{count} scored -> {args.out}")
    elif args.command == "evaluate":
        _print_metrics(evaluate(args.data, args.scored, args.out, ehrsql_repo=args.ehrsql_repo))
    elif args.command == "run":
        _print_metrics(run(args.data, args.backend, args.model, args.out_dir, ehrsql_repo=args.ehrsql_repo,
                                    **_generation_options(args)))


if __name__ == "__main__":
    main()
