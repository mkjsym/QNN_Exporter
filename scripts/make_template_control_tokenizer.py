#!/usr/bin/env python3
"""Create a vocab-only GGUF with another model's chat template for a controlled experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from gguf import GGUFReader, GGUFWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh path")
    source, template_source = GGUFReader(args.source), GGUFReader(args.template_source)
    template = template_source.fields["tokenizer.chat_template"].contents()
    architecture = source.fields["general.architecture"].contents()
    writer = GGUFWriter(args.output, architecture)
    for name, field in source.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        value = template if name == "tokenizer.chat_template" else field.contents()
        writer.add_key_value(name, value, field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    check = GGUFReader(args.output)
    assert not check.tensors
    changed = [name for name, field in source.fields.items() if not name.startswith("GGUF.")
        and field.contents() != check.fields[name].contents()]
    assert changed == ["tokenizer.chat_template"]
    manifest = {"source": str(args.source), "template_source": str(args.template_source),
        "output": str(args.output), "metadata_changed": changed, "tensors": 0,
        "scope": "tokenizer metadata only; QNN target weights and external embeddings are unchanged",
        "sha256": {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in (args.source, args.template_source, args.output)}}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
