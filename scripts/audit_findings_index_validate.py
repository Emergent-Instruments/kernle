#!/usr/bin/env python3
"""Validate consistency between findings index and per-pass finding listings."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

import yaml


def fail(message: str, errors: Optional[List[str]] = None) -> None:
    if errors is None:
        print(f"FAIL: {message}")
        return
    errors.append(f"FAIL: {message}")


def parse_yaml(path: pathlib.Path, errors: List[str]) -> Mapping[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"ERROR: unable to read {path}: {e}") from e

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise SystemExit(f"ERROR: YAML parse failed for {path}: {e}") from e

    if data is None:
        raise SystemExit(f"ERROR: empty YAML in {path}")
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected mapping root in {path}")
    return data


def collect_pass_findings(pass_file: pathlib.Path, errors: List[str]) -> List[Dict[str, Any]]:
    data = parse_yaml(pass_file, errors)
    return collect_pass_findings_from_data(pass_file, data, errors)


def collect_pass_findings_from_data(
    pass_file: pathlib.Path, data: Mapping[str, Any], errors: List[str]
) -> List[Dict[str, Any]]:
    findings_block = data.get("audit", {}).get("findings")
    if findings_block is None:
        return []
    if not isinstance(findings_block, dict):
        fail(f"{pass_file}: findings must be a mapping", errors)
        return []

    findings: List[Dict[str, Any]] = []
    for bucket_name, items in findings_block.items():
        if not isinstance(items, list):
            fail(f"{pass_file}: findings[{bucket_name}] must be a list", errors)
            continue
        for item in items:
            if not isinstance(item, dict):
                fail(f"{pass_file}: findings item in {bucket_name} must be a mapping", errors)
                continue
            item = dict(item)
            item["bucket"] = bucket_name
            findings.append(item)
    return findings


def validate_pass_findings_shape(
    pass_file: pathlib.Path, findings: List[Dict[str, Any]], errors: List[str]
) -> None:
    allowed_buckets = {"critical", "high", "medium", "low", "docs_only"}
    allowed_statuses = {"resolved", "mitigated", "open"}
    for item in findings:
        item_id = item.get("id")
        bucket = item.get("bucket")
        status = item.get("status")
        severity = item.get("severity")

        if bucket not in allowed_buckets:
            fail(f"{pass_file}: unexpected findings bucket '{bucket}'", errors)
            continue
        if severity != bucket:
            fail(
                f"{pass_file}: finding {item_id} severity '{severity}' does not match bucket '{bucket}'",
                errors,
            )
        if status not in allowed_statuses:
            fail(f"{pass_file}: finding {item_id} has invalid status '{status}'", errors)
        if not item_id:
            fail(f"{pass_file}: finding missing id")


def validate_calls_graph(
    pass_file: pathlib.Path, data: Mapping[str, Any], errors: List[str]
) -> None:
    """Ensure all call graph entries reference canonical function IDs declared in the pass."""
    audit = data.get("audit", {})
    if not isinstance(audit, dict):
        return
    calls_graph = audit.get("calls_graph", [])
    if calls_graph is None:
        return
    if not isinstance(calls_graph, list):
        fail(f"{pass_file}: calls_graph must be a list", errors)
        return

    function_ids = set()
    for func in audit.get("functions", []) or []:
        if isinstance(func, dict) and isinstance(func.get("id"), str):
            function_ids.add(func["id"])

    id_pattern = re.compile(r"^F-[A-Z0-9-]+(?:-[0-9]{3})?[a-z]?$")
    for idx, edge in enumerate(calls_graph, 1):
        if not isinstance(edge, dict):
            fail(f"{pass_file}: calls_graph[{idx}] must be a mapping", errors)
            continue
        source = edge.get("from")
        if not isinstance(source, str):
            fail(f"{pass_file}: calls_graph[{idx}].from must be a string", errors)
            continue
        if not id_pattern.match(source):
            fail(
                f"{pass_file}: calls_graph[{idx}].from is not a canonical function ID ({source})",
                errors,
            )
            continue
        if source not in function_ids:
            fail(
                f"{pass_file}: calls_graph[{idx}].from references unknown function '{source}'",
                errors,
            )
            continue

        targets = edge.get("to", [])
        if not isinstance(targets, list):
            fail(f"{pass_file}: calls_graph[{idx}].to must be a list", errors)
            continue
        for target in targets:
            if not isinstance(target, str):
                fail(
                    f"{pass_file}: calls_graph[{idx}].to contains non-string target {target!r}",
                    errors,
                )
                continue
            if not id_pattern.match(target):
                fail(
                    f"{pass_file}: calls_graph[{idx}] target '{target}' is not a canonical function ID",
                    errors,
                )
                continue
            if target not in function_ids:
                fail(
                    f"{pass_file}: calls_graph[{idx}] target '{target}' references unknown function",
                    errors,
                )


def collect_index_findings(index_path: pathlib.Path, errors: List[str]) -> List[Dict[str, Any]]:
    data = parse_yaml(index_path, errors)
    findings = data.get("findings")
    if findings is None:
        return []
    if not isinstance(findings, list):
        raise SystemExit(f"ERROR: {index_path}: findings must be a list")

    normalized = []
    for item in findings:
        if not isinstance(item, dict):
            fail(f"{index_path}: findings item must be a mapping", errors)
            continue
        normalized.append(dict(item))
    return normalized


def tally(findings: Iterable[Mapping[str, Any]]) -> Dict[str, Counter]:
    status = Counter()
    severity = Counter()
    ids = []
    for item in findings:
        ids.append(item.get("id"))
        status[item.get("status", "missing")] += 1
        severity[item.get("severity", "missing")] += 1
    return {
        "status": status,
        "severity": severity,
        "ids": Counter(ids),
        "count": Counter({"total": len(ids)}),
    }


def validate(
    index_path: pathlib.Path, pass_paths: Iterable[pathlib.Path], require: Optional[Set[str]] = None
) -> bool:
    errors: List[str] = []
    index_data = parse_yaml(index_path, errors)

    pass_findings: List[Dict[str, Any]] = []
    seen_finding_ids: Set[str] = set()
    for pass_path in pass_paths:
        pass_data = parse_yaml(pass_path, errors)
        findings = collect_pass_findings_from_data(pass_path, pass_data, errors)
        validate_pass_findings_shape(pass_path, findings, errors)
        validate_calls_graph(pass_path, pass_data, errors)
        for finding in findings:
            finding_id = finding.get("id")
            if not finding_id:
                continue
            if finding_id in seen_finding_ids:
                fail(f"duplicate finding ID across pass files: {finding_id}")
            seen_finding_ids.add(finding_id)
        pass_findings.extend(findings)

    index_findings = collect_index_findings(index_path, errors)

    index_summary = index_data.get("summary", {})
    if not isinstance(index_summary, dict):
        fail(f"{index_path}: summary section must be a mapping", errors)
        index_summary = {}

    expected_statuses = {"resolved", "mitigated", "open"}
    expected_severities = {"high", "medium", "low", "docs_only"}

    pass_tally = tally(pass_findings)
    index_tally = tally(index_findings)

    def expect_int(value: Any, key: str, fallback: int = 0) -> int:
        if isinstance(value, int):
            return value
        if value is None:
            return fallback
        fail(f"{index_path}: summary.{key} should be integer, got {value!r}")
        return fallback

    if pass_tally["count"]["total"] != index_tally["count"]["total"]:
        fail(
            "pass occurrence mismatch: "
            f"passes={pass_tally['count']['total']}, "
            f"index={index_tally['count']['total']}"
        )

    if (
        expect_int(index_summary.get("pass_occurrences"), "pass_occurrences", 0)
        != pass_tally["count"]["total"]
    ):
        fail(
            "summary.pass_occurrences mismatch: "
            f"declared={index_summary.get('pass_occurrences')}, "
            f"actual={pass_tally['count']['total']}"
        )

    if expect_int(index_summary.get("unique_findings"), "unique_findings", 0) != len(
        index_tally["ids"]
    ):
        fail(
            "summary.unique_findings mismatch: "
            f"declared={index_summary.get('unique_findings')}, "
            f"actual={len(index_tally['ids'])}"
        )

    if require is not None:
        pass_ids = set(index_tally["ids"])
        missing = require - pass_ids
        if missing:
            fail(f"{index_path}: missing findings IDs in pass files: {', '.join(sorted(missing))}")

    for status in expected_statuses:
        declared = expect_int(index_summary.get(status), status, 0)
        actual = pass_tally["status"].get(status, 0)
        if declared != actual:
            fail(f"summary.{status} mismatch: declared={declared}, actual={actual}")

    # Optional severity checks can fail if the index does not track this dimension.
    for severity in expected_severities:
        declared = expect_int(index_summary.get(severity), severity, None)
        if declared is None:
            continue
        actual = pass_tally["severity"].get(severity, 0)
        if declared != actual:
            fail(f"summary.{severity} mismatch: declared={declared}, actual={actual}")

    duplicate = [id_ for id_, cnt in index_tally["ids"].items() if cnt > 1]
    if duplicate:
        fail(f"duplicate finding IDs in index: {', '.join(sorted(duplicate))}")

    index_lookup = {item.get("id"): item for item in index_findings}
    for item in pass_findings:
        item_id = item.get("id")
        if not item_id:
            fail("pass finding missing id")
            continue
        indexed = index_lookup.get(item_id)
        if indexed is None:
            fail(f"finding {item_id} missing from index")
            continue
        pass_status = item.get("status")
        pass_severity = item.get("severity")
        if pass_status and indexed.get("status") != pass_status:
            fail(
                f"finding {item_id} status mismatch: "
                f"pass={pass_status}, index={indexed.get('status')}"
            )
        if pass_severity and indexed.get("severity") != pass_severity:
            fail(
                f"finding {item_id} severity mismatch: "
                f"pass={pass_severity}, index={indexed.get('severity')}"
            )

    residual = index_data.get("residual_risk")
    if residual:
        if not isinstance(residual, dict):
            fail("residual_risk must be a mapping")
        else:
            backlog = residual.get("mitigation_backlog", [])
            if backlog and not isinstance(backlog, list):
                fail("residual_risk.mitigation_backlog must be a list")
            elif isinstance(backlog, list):
                for item in backlog:
                    if not isinstance(item, dict):
                        fail("residual_risk.mitigation_backlog entries must be mappings")
                        continue
                    backlog_id = item.get("id")
                    if backlog_id is None:
                        fail("residual_risk.mitigation_backlog entry missing id")
                        continue
                    if backlog_id not in index_tally["ids"]:
                        fail(f"residual risk references finding id not in index: {backlog_id}")

            heatmap = residual.get("residual_risk_heatmap")
            if heatmap and not isinstance(heatmap, dict):
                fail("residual_risk.residual_risk_heatmap must be a mapping")
            if isinstance(heatmap, dict):
                total = sum(
                    v for key, v in heatmap.items() if key != "total" and isinstance(v, int)
                )
                if total != pass_tally["status"].get("mitigated", 0):
                    fail(
                        "residual_risk.residual_risk_heatmap total mismatch with mitigated findings: "
                        f"declared={total}, mitigated={pass_tally['status'].get('mitigated', 0)}"
                    )

    if pass_tally["status"].get("open", 0) != 0:
        fail("passes contain open findings; expected zero in this governance snapshot")

    if errors:
        print(f"Validation failed for {index_path}:")
        for err in errors:
            print(f"  - {err}")
        return False

    print(f"{index_path}: OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="audits/logic-audit-findings-index.yaml",
        help="Path to the findings index YAML",
    )
    parser.add_argument(
        "--pass-dir",
        default="audits",
        help="Directory containing logic-audit-pass-*.yaml files",
    )
    args = parser.parse_args()

    index_path = pathlib.Path(args.index)
    pass_dir = pathlib.Path(args.pass_dir)
    pass_files = sorted(pass_dir.glob("logic-audit-pass-*.yaml"))
    if not pass_files:
        raise SystemExit(f"ERROR: no pass files found in {pass_dir}")

    index_data = parse_yaml(index_path, [])
    required_ids = {item.get("id") for item in index_data.get("findings", []) if item.get("id")}

    if not validate(index_path, pass_files, required_ids):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
