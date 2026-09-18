#!/usr/bin/env python3
"""Create a non-deployable, public documentation snapshot of the AppBuilder YAML.

The source export contains deployment metadata (signed asset URLs, application
identifiers, editor metadata, and model endpoint identifiers).  This script
keeps the workflow logic, schemas, prompts, and local Python code while removing
those deployment-specific details.  The output is intentionally a documentation
blueprint and must not be treated as an importable platform export.
"""

from __future__ import annotations

import argparse
import copy
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml


ID_KEY = re.compile(r"(?i)(?:^id$|_id$|^id_|node_id$|param_id$|handle_id$)")
URL_KEY = re.compile(r"(?i)(?:url|uri|path)$")
ROLE_REPLACEMENTS = (
    ("学校心理学家的身份", "FBA指导的教育决策支持助手身份"),
    ("学校心理学家", "FBA指导的教育决策支持助手"),
)


class PublicWorkflowSanitizer:
    def __init__(self, source: dict[str, Any]) -> None:
        self.source = source
        self.id_map: dict[str, str] = {}
        self.node_id_map: dict[str, str] = {}
        nodes = source.get("workflow_detail", {}).get("workflow_schema", {}).get("nodes", [])
        for index, node in enumerate(nodes, start=1):
            original = node.get("id")
            if isinstance(original, str):
                self.node_id_map[original] = f"node_{index:03d}"

    def sanitize_text(self, value: str) -> str:
        result = value
        for old, new in ROLE_REPLACEMENTS:
            result = result.replace(old, new)
        return result

    def safe_id(self, value: str) -> str:
        if value in self.node_id_map:
            return self.node_id_map[value]
        if value not in self.id_map:
            self.id_map[value] = f"ref_{len(self.id_map) + 1:04d}"
        return self.id_map[value]

    def safe_scalar(self, key: str, value: Any) -> Any:
        if isinstance(value, str):
            if URL_KEY.search(key):
                return "${MODEL_ENDPOINT}" if key == "model_url" else None
            if ID_KEY.search(key):
                return self.safe_id(value) if value else value
            return self.sanitize_text(value)
        return value

    def sanitize_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in settings.items():
            if key in {"model_id", "model_value"}:
                result[key] = "${MODEL_ID}"
                continue
            if key == "model_name":
                result[key] = "${MODEL_NAME}"
                continue
            if key == "model_url":
                result[key] = "${MODEL_ENDPOINT}"
                continue
            if key == "llm_advanced_parameters" and not value:
                continue
            if isinstance(value, dict):
                nested = self.sanitize_settings(value)
                if nested:
                    result[key] = nested
            elif isinstance(value, list):
                result[key] = [self.sanitize_value(key, item) for item in value]
            else:
                safe = self.safe_scalar(key, value)
                if safe is not None:
                    result[key] = safe
        return result

    def sanitize_value(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return self.sanitize_settings(value)
        if isinstance(value, list):
            return [self.sanitize_value(key, item) for item in value]
        return self.safe_scalar(key, value)

    def schema_item(self, item: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("name", "type", "required"):
            if key in item:
                result[key] = self.sanitize_text(item[key]) if isinstance(item[key], str) else item[key]
        description = item.get("desc")
        if description:
            result["description"] = self.sanitize_text(description)
        value = item.get("value")
        if isinstance(value, dict) and value.get("type") == "literal":
            content = value.get("content")
            if isinstance(content, (str, int, float, bool)):
                result["literal_value"] = self.sanitize_text(content) if isinstance(content, str) else content
        return result

    def node(self, node: dict[str, Any]) -> dict[str, Any]:
        data = node.get("data", {})
        settings = data.get("settings", {})
        original_id = node.get("id")
        public_id = self.node_id_map.get(original_id)
        if public_id is None:
            public_id = self.safe_id(str(original_id or ""))
        result: dict[str, Any] = {
            "id": public_id,
            "type": node.get("type", "unknown"),
            "name": self.sanitize_text(str(node.get("name", ""))),
            "inputs": [self.schema_item(item) for item in data.get("inputs", []) if isinstance(item, dict)],
            "outputs": [self.schema_item(item) for item in data.get("outputs", []) if isinstance(item, dict)],
        }
        if isinstance(settings, dict) and settings:
            result["settings"] = self.sanitize_settings(settings)
        return result

    def memory_variables(self) -> list[dict[str, Any]]:
        schema = (
            self.source.get("app_instance_config", {})
            .get("memory", {})
            .get("variable", {})
            .get("schema", [])
        )
        nodes_text = repr(self.source.get("workflow_detail", {}).get("workflow_schema", {}).get("nodes", []))
        result = []
        for item in schema:
            name = item.get("name")
            if not name or name not in nodes_text:
                continue
            result.append(
                {
                    "name": name,
                    "description": self.sanitize_text(str(item.get("description", ""))),
                    "entity_life_cycle": item.get("entity_life_cycle", "user"),
                }
            )
        return result

    def build(self) -> dict[str, Any]:
        workflow_schema = self.source.get("workflow_detail", {}).get("workflow_schema", {})
        edges = []
        for edge in workflow_schema.get("edges", []):
            source = edge.get("source_node_id")
            target = edge.get("target_node_id")
            if source not in self.node_id_map or target not in self.node_id_map:
                continue
            item = {
                "source": self.node_id_map[source],
                "target": self.node_id_map[target],
            }
            if "source_port_id" in edge:
                item["source_port"] = edge["source_port_id"]
            if "target_port_id" in edge:
                item["target_port"] = edge["target_port_id"]
            edges.append(item)

        global_config = workflow_schema.get("global_config", {})
        safe_global = self.sanitize_settings(global_config) if isinstance(global_config, dict) else {}
        result: dict[str, Any] = {
            "public_release": {
                "schema": "fba_workflow_public_blueprint/v1",
                "generated_date": date.today().isoformat(),
                "deployable": False,
                "source_format": "Baidu AppBuilder workflow export",
                "purpose": "Public documentation of the research workflow structure and prompt/code logic.",
                "redactions": [
                    "signed asset URLs and live share links",
                    "application, workflow, node, port, and deployment identifiers",
                    "platform editor metadata and duplicated rich-text templates",
                    "model service endpoints and deployment-specific model identifiers",
                    "unreferenced test variables",
                ],
                "safety_boundary": {
                    "intended_use": "Research documentation and code review only.",
                    "not_a": [
                        "medical or clinical diagnosis",
                        "treatment or crisis-management service",
                        "autonomous or final high-stakes decision tool",
                        "replacement for educator or qualified professional review",
                    ],
                    "review_required": "Educators and, when appropriate, qualified professionals must review every hypothesis and recommendation.",
                    "risk_escalation": "Self-harm, harm-to-others, or other serious-risk indications require immediate escalation under the responsible institution's procedures.",
                    "privacy": "Do not place names, school identifiers, raw child narratives, chat logs, access keys, or signed URLs in this public file.",
                },
            },
            "application": {
                "name": "FBA-guided behavior-assessment decision-support workflow",
                "description": self.sanitize_text(str(self.source.get("app_desc", ""))),
                "export_version": self.source.get("ab_version"),
                "platform_configuration": {
                    "provider": "redacted",
                    "endpoint": "${MODEL_ENDPOINT}",
                    "model": "${MODEL_ID}",
                    "retrieval_corpus": "Not included in this public snapshot.",
                },
                "memory_variables": self.memory_variables(),
            },
            "workflow": {
                "history_chat_rounds": self.source.get("app_instance_config", {}).get("history_chat_rounds"),
                "global_config": safe_global,
                "nodes": [self.node(node) for node in workflow_schema.get("nodes", [])],
                "edges": edges,
            },
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original AppBuilder YAML export (kept outside the public repository).")
    parser.add_argument("output", type=Path, help="Path for the sanitized public YAML blueprint.")
    args = parser.parse_args()

    source = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("The source YAML root must be a mapping.")
    public = PublicWorkflowSanitizer(copy.deepcopy(source)).build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(public, allow_unicode=True, sort_keys=False, width=120, default_flow_style=False),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
