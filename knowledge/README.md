# CadQuery RAG Knowledge Base

This directory contains three searchable layers:

- `cadquery_reference.json`: small manually curated modelling patterns and examples.
- `cadquery_official_guides.json`: topic-level coverage of the official Workplane, Sketch, Assembly, and Examples guides.
- `cadquery_api_reference.json`: generated API entries for public CadQuery classes, methods, properties, import/export functions, and free shape functions.

The API index is generated from the signatures and docstrings of the CadQuery version installed in the project environment. This keeps method signatures aligned with the code that will actually execute while every entry links back to the official documentation.

Regenerate after upgrading CadQuery:

```bash
python scripts/build_cadquery_reference.py
```

The retriever searches all `cadquery_*.json` list files. `cadquery_reference_manifest.json` records the installed version, entry counts, and official source pages but is not searched.
