# Public Workflow Blueprint

`functional_behavior_assessment_workflow_public.yaml` is a sanitized documentation
snapshot of the research workflow. It is not an importable Baidu AppBuilder export
and it does not contain the live application, deployment identifiers, model
credentials, signed asset URLs, or retrieval corpus.

The accompanying `../tools/sanitize_public_workflow.py` script creates the file
from the private source export. The source export remains outside this repository.
The public snapshot keeps node types, input/output schemas, Python code, prompt
logic, and high-level edges while replacing deployment-specific model settings
with `${MODEL_ENDPOINT}`, `${MODEL_ID}`, and `${MODEL_NAME}` placeholders.

The agent's model components were updated after study data collection. This
sanitized snapshot is derived from the latest agent version, whereas the
manuscript reports the study-time models documented for the evaluated workflow.
The retrieval knowledge base is not included because it is linked to specific
case materials collected for the study and is withheld for participant privacy
and system security.

The workflow is research decision support. It does not diagnose, treat, or make
autonomous high-stakes decisions. Educators and, when appropriate, qualified
professionals must review every generated hypothesis or recommendation, and
serious-risk indications must be escalated under the responsible institution's
procedures.
