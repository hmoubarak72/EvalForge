"""Export the active evaluation dataset as JSON Lines.

Each line is one test case with its current labels, ready to be consumed by
another eval harness or checked into a data repository.

Usage:
    python scripts/export_dataset.py --output data/eval_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evalforge.config import get_settings  # noqa: E402
from evalforge.container import build_container  # noqa: E402
from evalforge.db.session import session_scope  # noqa: E402
from evalforge.repositories import DatasetRepository  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="data/eval_dataset.jsonl")
    args = parser.parse_args(argv)

    container = build_container(get_settings())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        session_scope(container.session_factory) as session,
        output.open("w", encoding="utf-8") as handle,
    ):
        dataset = DatasetRepository(session)
        version = dataset.current_version()
        count = 0
        for case in dataset.active_test_cases():
            label = case.current_label
            if label is None:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": case.id,
                        "dataset_version": version,
                        "category": case.category,
                        "difficulty": case.difficulty,
                        "case_type": case.case_type,
                        "source": case.source,
                        "application_feature": case.application_feature,
                        "system_prompt": case.system_prompt,
                        "user_prompt": case.user_prompt,
                        "expected_behavior": label.expected_behavior,
                        "expected_quality_score": label.expected_quality_score,
                        "question_type": label.question_type,
                        "reference_answer": label.reference_answer,
                        "rubric": label.rubric,
                        "required_assertions": label.required_assertions,
                        "forbidden_assertions": label.forbidden_assertions,
                        "label_confidence": label.confidence,
                        "label_version": label.version,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    print(f"Exported {count} test case(s) from dataset v{version} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
