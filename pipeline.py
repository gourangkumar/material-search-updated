from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from text_utils import (
    clean_scalar,
    load_json_map,
    normalize_material_record,
    normalize_text,
    numeric_conflicts,
    query_features,
)


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_path(value: str, data_dir: Path) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else data_dir / path
    if not resolved.exists() and resolved.suffix.lower() == ".csv":
        zipped = Path(str(resolved) + ".zip")
        if zipped.exists():
            return zipped
    return resolved


def load_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for vector mode. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name, device=device)


def load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for cross-encoder reranking."
        ) from exc
    return CrossEncoder(model_name)


def iter_master_chunks(path: Path, source: str, required: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    if not path.exists():
        return
    input_source: Any = path
    archive = None
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        candidates = [
            info for info in archive.infolist()
            if info.filename.lower().endswith(".csv") and not info.filename.startswith("__MACOSX/")
        ]
        if not candidates:
            archive.close()
            raise ValueError(f"No CSV file found inside {path}")
        member = max(candidates, key=lambda info: info.file_size)
        input_source = archive.open(member)
    try:
        iterator = pd.read_csv(
            input_source,
            usecols=required,
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            low_memory=False,
        )
    except ValueError as exc:
        if archive is not None:
            archive.close()
        header = []
        missing = [column for column in required if column not in header]
        raise ValueError(f"{path.name} is missing required columns: {missing}") from exc
    try:
        for chunk in iterator:
            chunk["source"] = source
            yield chunk
    finally:
        if archive is not None:
            archive.close()


def build_index_jsonl(
    config: dict[str, Any],
    data_dir: Path,
    output: Path,
    corpus_ids_output: Path,
    brand_lexicon_output: Path,
    include_embeddings: bool,
    require_secondary: bool,
    max_documents: int | None = None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    embed_cfg = config["embedding"]
    normal_cfg = config["normalization"]
    required = data_cfg["required_columns"]
    primary = resolve_data_path(data_cfg["primary_master"], data_dir)
    secondary = resolve_data_path(data_cfg["secondary_master"], data_dir)

    if not primary.exists():
        raise FileNotFoundError(f"Primary master not found: {primary}")
    if require_secondary and not secondary.exists():
        raise FileNotFoundError(f"Secondary master not found: {secondary}")
    if not secondary.exists():
        print(f"WARNING: secondary master not found; continuing with {primary.name} only", file=sys.stderr)

    abbreviations = load_json_map(ROOT / "abbreviations.json")
    aliases = load_json_map(ROOT / "brand_aliases.json")
    model = None
    if include_embeddings:
        model = load_sentence_transformer(embed_cfg["model"], embed_cfg.get("device", "cpu"))

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    brands: set[str] = set()
    source_counts: dict[str, int] = defaultdict(int)
    skipped_duplicate = skipped_invalid = 0

    sources = [(primary, "master")]
    if secondary.exists():
        sources.append((secondary, "temp"))

    with output.open("w", encoding="utf-8") as out:
        for path, source in sources:
            for chunk_no, chunk in enumerate(
                iter_master_chunks(path, source, required, int(data_cfg["chunksize"])), start=1
            ):
                documents: list[dict[str, Any]] = []
                for raw in chunk.to_dict(orient="records"):
                    if max_documents is not None and len(seen_ids) >= max_documents:
                        break
                    material_id = clean_scalar(raw.get("materialId"))
                    if not material_id or material_id in seen_ids:
                        skipped_duplicate += int(bool(material_id))
                        skipped_invalid += int(not bool(material_id))
                        continue
                    product_name = clean_scalar(raw.get("productName"))
                    erp = clean_scalar(raw.get("companyERPCode"))
                    specification = clean_scalar(raw.get("productSpecification"))
                    if not product_name and not erp and not specification:
                        skipped_invalid += 1
                        continue
                    raw["source"] = source
                    document = normalize_material_record(
                        raw,
                        abbreviations,
                        aliases,
                        remove_commercial_noise=bool(normal_cfg.get("remove_commercial_noise", True)),
                    )
                    seen_ids.add(material_id)
                    if document["brandNameNormalized"]:
                        brands.add(str(document["brandNameNormalized"]))
                    documents.append(document)

                if include_embeddings and documents:
                    vectors = model.encode(
                        [document["embeddingText"] for document in documents],
                        batch_size=int(embed_cfg["batch_size"]),
                        normalize_embeddings=bool(embed_cfg.get("normalize_embeddings", True)),
                        show_progress_bar=False,
                    )
                    if vectors.shape[1] != int(embed_cfg["dimension"]):
                        raise ValueError(
                            f"Embedding dimension {vectors.shape[1]} does not match configured "
                            f"dimension {embed_cfg['dimension']}"
                        )
                    for document, vector in zip(documents, vectors, strict=True):
                        document["embedding"] = np.asarray(vector, dtype=np.float32).tolist()

                for document in documents:
                    document.pop("embeddingText", None)
                    out.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
                source_counts[source] += len(documents)
                print(
                    f"{source}: chunk={chunk_no:,}, indexed={source_counts[source]:,}, "
                    f"unique_total={len(seen_ids):,}",
                    file=sys.stderr,
                )
                if max_documents is not None and len(seen_ids) >= max_documents:
                    break
            if max_documents is not None and len(seen_ids) >= max_documents:
                break

    with corpus_ids_output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(seen_ids)) + "\n")
    with brand_lexicon_output.open("w", encoding="utf-8") as handle:
        json.dump({"brands": sorted(brands), "aliases": aliases}, handle, ensure_ascii=False, indent=2)

    summary = {
        "documents": len(seen_ids),
        "source_counts": dict(source_counts),
        "skipped_duplicate": skipped_duplicate,
        "skipped_invalid": skipped_invalid,
        "embeddings_included": include_embeddings,
        "secondary_loaded": secondary.exists(),
        "max_documents": max_documents,
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


def upgrade_index_jsonl(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    max_documents: int | None = None,
) -> dict[str, Any]:
    """Rebuild normalized fields while preserving vectors from an existing JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"Existing JSONL not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")
    abbreviations = load_json_map(ROOT / "abbreviations.json")
    aliases = load_json_map(ROOT / "brand_aliases.json")
    normal_cfg = config["normalization"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = embeddings_preserved = invalid = 0
    source_counts: dict[str, int] = defaultdict(int)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, start=1):
            if max_documents is not None and written >= max_documents:
                break
            if not line.strip():
                continue
            try:
                previous = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                raise ValueError(f"Invalid JSON on line {line_number:,} of {input_path}") from exc
            rebuilt = normalize_material_record(
                previous,
                abbreviations,
                aliases,
                remove_commercial_noise=bool(normal_cfg.get("remove_commercial_noise", True)),
            )
            rebuilt.pop("embeddingText", None)
            if "embedding" in previous:
                rebuilt["embedding"] = previous["embedding"]
                embeddings_preserved += 1
            destination.write(json.dumps(rebuilt, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            source_counts[str(rebuilt.get("source", ""))] += 1
            if written % 50_000 == 0:
                print(f"upgraded={written:,}, embeddings_preserved={embeddings_preserved:,}", file=sys.stderr)
    summary = {
        "documents": written,
        "source_counts": dict(source_counts),
        "embeddings_preserved": embeddings_preserved,
        "invalid": invalid,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


class TypesenseREST:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        env_name = str(config["api_key_env"])
        self.api_key = os.environ.get(env_name, "")
        if not self.api_key:
            raise RuntimeError(f"Environment variable {env_name} is not set")
        base_url_env = str(config.get("base_url_env", "TYPESENSE_URL"))
        direct_url = os.environ.get(base_url_env, "").strip()
        if direct_url:
            self.base_url = direct_url.rstrip("/")
        else:
            protocol = os.environ.get("TYPESENSE_PROTOCOL", str(config.get("protocol", "http")))
            host = os.environ.get("TYPESENSE_HOST", str(config.get("host", "localhost")))
            port = os.environ.get("TYPESENSE_PORT", str(config.get("port", 8108)))
            if "://" in host:
                self.base_url = host.rstrip("/")
            else:
                self.base_url = f"{protocol}://{host}:{port}"
        self.timeout = float(config.get("connection_timeout_seconds", 10))

    def request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        content_type: str = "application/json",
        retries: int = 3,
        timeout_seconds: float | None = None,
        retry_delay_seconds: float = 0.25,
        max_retry_delay_seconds: float = 8.0,
    ) -> Any:
        data: bytes | None
        if payload is None:
            data = None
        elif content_type == "application/json":
            data = json.dumps(payload).encode("utf-8")
        else:
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
        request_timeout = self.timeout if timeout_seconds is None else float(timeout_seconds)
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                self.base_url + path,
                data=data,
                method=method,
                headers={
                    "X-TYPESENSE-API-KEY": self.api_key,
                    "Content-Type": content_type,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    body = response.read().decode("utf-8")
                    if not body:
                        return None
                    if content_type == "text/plain":
                        return body
                    return json.loads(body)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                    raise RuntimeError(f"Typesense HTTP {exc.code}: {details}") from exc
                print(
                    f"Typesense HTTP {exc.code} ({details.strip() or 'temporary server error'}); "
                    f"retrying attempt {attempt + 2}/{retries + 1}",
                    file=sys.stderr,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                http.client.IncompleteRead,
            ) as exc:
                if attempt == retries:
                    raise RuntimeError(f"Could not reach Typesense at {self.base_url}: {exc}") from exc
                print(
                    f"Typesense request interrupted ({type(exc).__name__}); "
                    f"retrying attempt {attempt + 2}/{retries + 1}",
                    file=sys.stderr,
                )
            delay = min(float(max_retry_delay_seconds), float(retry_delay_seconds) * (2**attempt))
            time.sleep(delay)
        raise AssertionError("unreachable")

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def multi_search(self, searches: list[dict[str, Any]]) -> dict[str, Any]:
        return self.request("POST", "/multi_search", {"searches": searches})


def import_jsonl_into_collection(
    config: dict[str, Any],
    jsonl_path: Path,
    name: str,
    client: TypesenseREST,
    skip_documents: int = 0,
    batch_size_override: int | None = None,
    publish_alias: bool = True,
) -> dict[str, Any]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    batch_size = int(batch_size_override or config["typesense"].get("import_batch_size", 500))
    import_timeout = float(config["typesense"].get("import_timeout_seconds", 300))
    import_retries = int(config["typesense"].get("import_retries", 5))
    retry_delay = float(config["typesense"].get("import_retry_delay_seconds", 15))
    max_retry_delay = float(config["typesense"].get("import_max_retry_delay_seconds", 60))
    progress_every = int(config["typesense"].get("import_progress_every", 5000))
    imported = failed = 0
    last_report = 0
    batch: list[str] = []

    def flush(lines: list[str]) -> tuple[int, int]:
        if not lines:
            return 0, 0
        path = f"/collections/{urllib.parse.quote(name)}/documents/import?action=upsert"
        raw_response = client.request(
            "POST",
            path,
            "\n".join(lines),
            "text/plain",
            retries=import_retries,
            timeout_seconds=import_timeout,
            retry_delay_seconds=retry_delay,
            max_retry_delay_seconds=max_retry_delay,
        )
        ok = bad = 0
        for line in raw_response.splitlines():
            result = json.loads(line)
            if result.get("success"):
                ok += 1
            else:
                bad += 1
                if bad <= 5:
                    print(f"IMPORT ERROR: {result}", file=sys.stderr)
        return ok, bad

    source_documents = 0
    if skip_documents:
        print(f"resuming collection={name}, skipping_already_imported={skip_documents:,}", file=sys.stderr)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_documents += 1
            if source_documents <= skip_documents:
                continue
            batch.append(line.rstrip("\n"))
            if len(batch) >= batch_size:
                ok, bad = flush(batch)
                imported += ok
                failed += bad
                batch.clear()
                if imported + failed - last_report >= progress_every:
                    print(f"imported={imported:,}, failed={failed:,}", file=sys.stderr)
                    last_report = imported + failed
        ok, bad = flush(batch)
        imported += ok
        failed += bad
    print(f"imported={imported:,}, failed={failed:,}", file=sys.stderr)

    if failed:
        raise RuntimeError(f"Index import completed with {failed:,} failed documents")
    metadata = client.request(
        "GET",
        f"/collections/{urllib.parse.quote(name)}",
        retries=import_retries,
        retry_delay_seconds=retry_delay,
        max_retry_delay_seconds=max_retry_delay,
    )
    final_documents = int(metadata.get("num_documents", 0))
    if final_documents != source_documents:
        raise RuntimeError(
            f"Collection verification failed: Typesense has {final_documents:,} documents but "
            f"the JSONL has {source_documents:,}. The alias was not changed."
        )
    alias = config["typesense"]["collection_alias"]
    if publish_alias:
        client.request("PUT", f"/aliases/{urllib.parse.quote(alias)}", {"collection_name": name})
    return {
        "collection_name": name,
        "alias": alias if publish_alias else None,
        "skipped_existing": skip_documents,
        "imported_this_run": imported,
        "final_documents": final_documents,
        "failed": failed,
    }


def create_and_import(
    config: dict[str, Any],
    schema_path: Path,
    jsonl_path: Path,
    collection_name: str | None,
    batch_size_override: int | None = None,
) -> dict[str, Any]:
    client = TypesenseREST(config["typesense"])
    if not client.health().get("ok"):
        raise RuntimeError("Typesense health check failed")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    name = collection_name or f"materials_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    schema["name"] = name
    collection = client.request("POST", "/collections", schema)
    result = import_jsonl_into_collection(
        config,
        jsonl_path,
        name,
        client,
        batch_size_override=batch_size_override,
    )
    result["collection"] = collection
    return result


def resume_import(
    config: dict[str, Any],
    jsonl_path: Path,
    collection_name: str,
    skip_documents: int | None = None,
    batch_size_override: int | None = None,
) -> dict[str, Any]:
    client = TypesenseREST(config["typesense"])
    if not client.health().get("ok"):
        raise RuntimeError("Typesense is not healthy yet; wait for recovery before resuming")
    path = f"/collections/{urllib.parse.quote(collection_name)}"
    metadata = client.request("GET", path)
    existing = int(metadata.get("num_documents", 0))
    start = existing if skip_documents is None else int(skip_documents)
    if start != existing:
        print(
            f"WARNING: --skip-documents={start:,}, current collection count={existing:,}",
            file=sys.stderr,
        )
    return import_jsonl_into_collection(
        config,
        jsonl_path,
        collection_name,
        client,
        skip_documents=start,
        batch_size_override=batch_size_override,
    )


class BrandDetector:
    def __init__(self, path: Path, generic_brands: Iterable[str]):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        canonical = set(payload.get("brands", []))
        self.aliases = {**payload.get("aliases", {})}
        self.generic = {normalize_text(value) for value in generic_brands}
        self.by_first_token: dict[str, list[str]] = defaultdict(list)
        for brand in canonical | set(self.aliases):
            normalized = normalize_text(brand)
            if normalized and normalized not in self.generic:
                self.by_first_token[normalized.split()[0]].append(normalized)
        for candidates in self.by_first_token.values():
            candidates.sort(key=len, reverse=True)

    def detect(self, normalized_query: str) -> set[str]:
        padded = f" {normalized_query} "
        found: set[str] = set()
        for token in set(normalized_query.split()):
            for brand in self.by_first_token.get(token, []):
                if f" {brand} " in padded:
                    found.add(self.aliases.get(brand, brand))
        return found


@dataclass
class RankedHit:
    document: dict[str, Any]
    lexical_score: float
    vector_score: float
    final_score: float
    explanation: list[str]
    initial_rank: int
    retrieval_channels: list[str]
    channel_ranks: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "materialId": str(self.document.get("materialId", "")),
            "brandName": self.document.get("brandName", ""),
            "productName": self.document.get("productName", ""),
            "productSpecification": self.document.get("productSpecification", ""),
            "companyERPCode": self.document.get("companyERPCode", ""),
            "source": self.document.get("source", ""),
            "lexical_score": round(self.lexical_score, 6),
            "vector_score": round(self.vector_score, 6),
            "final_hybrid_score": round(self.final_score, 6),
            "retrieval_channels": ";".join(self.retrieval_channels),
            "channel_ranks": json.dumps(self.channel_ranks, separators=(",", ":")),
            "match_explanation": "; ".join(self.explanation),
        }


class HybridSearcher:
    def __init__(
        self,
        config: dict[str, Any],
        brand_lexicon_path: Path,
        lexical_only: bool = False,
        cross_encoder: bool | None = None,
    ):
        self.config = config
        self.search_cfg = config["search"]
        self.client = TypesenseREST(config["typesense"])
        self.abbreviations = load_json_map(ROOT / "abbreviations.json")
        self.aliases = load_json_map(ROOT / "brand_aliases.json")
        self.brands = BrandDetector(brand_lexicon_path, config["normalization"]["generic_brands"])
        self.lexical_only = lexical_only
        self.model = None if lexical_only else load_sentence_transformer(
            config["embedding"]["model"], config["embedding"].get("device", "cpu")
        )
        enabled = self.search_cfg.get("adaptive_cross_encoder", False) if cross_encoder is None else cross_encoder
        self.cross_encoder = load_cross_encoder(self.search_cfg["cross_encoder_model"]) if enabled else None

    @staticmethod
    def _csv(values: Iterable[Any]) -> str:
        return ",".join(str(value).lower() if isinstance(value, bool) else str(value) for value in values)

    def _include_fields(self) -> str:
        return ",".join((
            "materialId", "categoryName", "brandName", "productName", "productSpecification", "companyERPCode", "source",
            "categoryNameNormalized", "brandNameNormalized", "productNameNormalized", "productSpecificationNormalized",
            "companyERPCodeNormalized", "modelTokens", "numericTokens", "physicalFingerprint",
        ))

    def _main_search(self, normalized_query: str, vector: list[float] | None, source: str) -> dict[str, Any]:
        cfg = self.search_cfg
        candidate_k = int(cfg.get(f"{source}_candidate_k", cfg["candidate_k"]))
        query = {
            "collection": self.config["typesense"]["collection_alias"],
            "q": normalized_query,
            "query_by": self._csv(cfg["query_by"]),
            "query_by_weights": self._csv(cfg["query_by_weights"]),
            "num_typos": self._csv(cfg["num_typos"]),
            "prefix": self._csv(cfg["prefix"]),
            "infix": self._csv(cfg["infix"]),
            "per_page": candidate_k,
            "include_fields": self._include_fields(),
            "exclude_fields": "embedding",
            "filter_by": f"source:={source}",
            "prioritize_exact_match": True,
            "prioritize_num_matching_fields": True,
            "enable_typos_for_alpha_numerical_tokens": False,
            "split_join_tokens": "always",
            "drop_tokens_threshold": int(cfg.get("drop_tokens_threshold", 100)),
            "typo_tokens_threshold": 5,
            "text_match_type": "sum_score",
            "sort_by": "_text_match:desc",
        }
        if vector is not None:
            vector_text = ",".join(f"{value:.8g}" for value in vector)
            query["vector_query"] = (
                f"embedding:([{vector_text}], k:{candidate_k}, "
                f"alpha:{float(cfg['vector_alpha'])}, ef:{int(cfg['hnsw_ef'])})"
            )
            query["rerank_hybrid_matches"] = bool(cfg.get("rerank_hybrid_matches", False))
        return query

    def _erp_search(self, code: str, source: str) -> dict[str, Any]:
        return {
            "collection": self.config["typesense"]["collection_alias"],
            "q": code,
            "query_by": "companyERPCodeNormalized",
            "query_by_weights": "127",
            "num_typos": "0",
            "prefix": "false",
            "infix": "off",
            "enable_typos_for_alpha_numerical_tokens": False,
            "prioritize_exact_match": True,
            "filter_by": f"source:={source}",
            "per_page": 10,
            "include_fields": self._include_fields(),
        }

    @staticmethod
    def _filter_literal(value: str) -> str:
        return value.replace("\\", "\\\\").replace("`", "\\`")

    def _source_filter(self, source: str, brand: str | None = None) -> str:
        clauses = [f"source:={source}"]
        if brand:
            clauses.append(
                f"brandNameNormalized:=[`{self._filter_literal(brand)}`]"
            )
        return " && ".join(clauses)

    def _model_search(self, model: str, source: str) -> dict[str, Any]:
        return {
            "collection": self.config["typesense"]["collection_alias"],
            "q": model,
            "query_by": "modelTokens",
            "query_by_weights": "127",
            "num_typos": "0",
            "prefix": "false",
            "infix": "off",
            "drop_tokens_threshold": 0,
            "typo_tokens_threshold": 0,
            "enable_typos_for_alpha_numerical_tokens": False,
            "prioritize_exact_match": True,
            "filter_by": self._source_filter(source),
            "per_page": int(self.search_cfg.get("structured_candidate_k", 40)),
            "include_fields": self._include_fields(),
        }

    def _numeric_search(self, numeric_query: str, source: str) -> dict[str, Any]:
        return {
            "collection": self.config["typesense"]["collection_alias"],
            "q": numeric_query,
            "query_by": "numericTokens",
            "query_by_weights": "127",
            "num_typos": "0",
            "prefix": "false",
            "infix": "off",
            "drop_tokens_threshold": 0,
            "typo_tokens_threshold": 0,
            "enable_typos_for_alpha_numerical_tokens": False,
            "prioritize_exact_match": True,
            "prioritize_num_matching_fields": True,
            "filter_by": self._source_filter(source),
            "per_page": int(self.search_cfg.get("structured_candidate_k", 40)),
            "include_fields": self._include_fields(),
        }

    def _brand_search(self, normalized_query: str, brand: str, source: str) -> dict[str, Any]:
        cfg = self.search_cfg
        return {
            "collection": self.config["typesense"]["collection_alias"],
            "q": normalized_query,
            "query_by": self._csv(cfg["query_by"]),
            "query_by_weights": self._csv(cfg["query_by_weights"]),
            "num_typos": self._csv(cfg["num_typos"]),
            "prefix": self._csv(cfg["prefix"]),
            "infix": self._csv(cfg["infix"]),
            "per_page": int(cfg.get("brand_candidate_k", 40)),
            "include_fields": self._include_fields(),
            "exclude_fields": "embedding",
            "filter_by": self._source_filter(source, brand),
            "prioritize_exact_match": True,
            "prioritize_num_matching_fields": True,
            "enable_typos_for_alpha_numerical_tokens": False,
            "split_join_tokens": "always",
            "drop_tokens_threshold": int(cfg.get("drop_tokens_threshold", 100)),
            "typo_tokens_threshold": 5,
            "text_match_type": "sum_score",
            "sort_by": "_text_match:desc",
        }

    @staticmethod
    def _score_key(item: RankedHit) -> tuple[float, float, int]:
        return (item.final_score, item.lexical_score, -item.initial_rank)

    def _primary_is_convincing(self, ranked: list[RankedHit]) -> tuple[bool, float, float]:
        primary = sorted(
            (item for item in ranked if item.document.get("source") == "master"),
            key=self._score_key,
            reverse=True,
        )
        if not primary:
            return False, 0.0, 0.0
        best = primary[0]
        second_score = primary[1].final_score if len(primary) > 1 else 0.0
        margin = best.final_score - second_score
        explanations = set(best.explanation)
        has_conflict = any("conflict" in value for value in explanations)
        trusted_erp = "trusted exact ERP code" in explanations
        strong_structured_match = (
            "model match" in explanations
            and ("brand match" in explanations or any("numeric specification match" in value for value in explanations))
        )
        threshold = float(self.search_cfg.get("primary_confidence_threshold", 0.9))
        margin_threshold = float(self.search_cfg.get("primary_margin_threshold", 0.08))
        convincing = trusted_erp or (
            not has_conflict
            and best.final_score >= threshold
            and (margin >= margin_threshold or strong_structured_match)
        )
        return convincing, best.final_score, margin

    def _deduplicate(self, ranked: list[RankedHit], primary_convincing: bool) -> list[RankedHit]:
        if not self.search_cfg.get("deduplicate_physical_fingerprints", True):
            return ranked
        winners: dict[str, RankedHit] = {}
        order: list[str] = []
        for item in ranked:
            material_id = str(item.document.get("materialId", ""))
            fingerprint = str(item.document.get("physicalFingerprint", ""))
            key = fingerprint or f"material:{material_id}"
            if key not in winners:
                winners[key] = item
                order.append(key)
                continue
            current = winners[key]
            prefer_primary_duplicate = (
                primary_convincing
                and item.document.get("source") == "master"
                and current.document.get("source") != "master"
            )
            if prefer_primary_duplicate or self._score_key(item) > self._score_key(current):
                winners[key] = item
        output = [winners[key] for key in order]
        output.sort(key=self._score_key, reverse=True)
        return output

    def _select_source_balanced_top(self, ranked: list[RankedHit], primary_convincing: bool) -> list[RankedHit]:
        top_k = int(self.search_cfg["top_k"])
        max_temp = int(self.search_cfg.get("max_temp_results_when_primary_strong", top_k))
        selected: list[RankedHit] = []
        for item in ranked:
            temp_count = sum(candidate.document.get("source") == "temp" for candidate in selected)
            if primary_convincing and item.document.get("source") == "temp" and temp_count >= max_temp:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

        minimums = {
            "master": int(self.search_cfg.get("min_master_results", 1)),
            "temp": int(self.search_cfg.get("min_temp_results", 1)),
        }
        selected_ids = {str(item.document.get("materialId", "")) for item in selected}
        for source, minimum in minimums.items():
            while sum(item.document.get("source") == source for item in selected) < minimum:
                replacement = next(
                    (
                        item for item in ranked
                        if item.document.get("source") == source
                        and str(item.document.get("materialId", "")) not in selected_ids
                    ),
                    None,
                )
                if replacement is None:
                    break
                removable_index = next(
                    (
                        index for index in range(len(selected) - 1, -1, -1)
                        if selected[index].document.get("source") != source
                        and sum(
                            item.document.get("source") == selected[index].document.get("source")
                            for item in selected
                        ) > minimums.get(str(selected[index].document.get("source")), 0)
                    ),
                    None,
                )
                if removable_index is None:
                    break
                removed = selected.pop(removable_index)
                selected_ids.discard(str(removed.document.get("materialId", "")))
                selected.append(replacement)
                selected_ids.add(str(replacement.document.get("materialId", "")))
                selected.sort(key=self._score_key, reverse=True)
        return selected[:top_k]

    def search(self, query: str, precomputed_vector: list[float] | None = None) -> dict[str, Any]:
        features = query_features(query, self.abbreviations, self.aliases)
        if not features["normalized"]:
            return {"query_text": query, "normalized_query": "", "no_confident_match": True, "results": []}
        vector = precomputed_vector
        if vector is None and self.model is not None:
            encoded = self.model.encode(
                [features["normalized"]],
                normalize_embeddings=bool(self.config["embedding"].get("normalize_embeddings", True)),
                show_progress_bar=False,
            )[0]
            vector = np.asarray(encoded, dtype=np.float32).tolist()

        trusted_erp_codes = set(features.get("trusted_erp_candidates", []))
        query_brands = self.brands.detect(str(features["normalized"]))
        query_models = set(features["model_tokens"])
        query_numbers = set(features["numeric_tokens"])
        searches: list[dict[str, Any]] = []
        search_metadata: list[tuple[str, str]] = []
        for code in trusted_erp_codes:
            for source in ("master", "temp"):
                searches.append(self._erp_search(code, source))
                search_metadata.append(("erp", source))
        for source in ("master", "temp"):
            searches.append(self._main_search(str(features["normalized"]), vector, source))
            search_metadata.append(("main", source))

        # Independent structured channels prevent exact engineering evidence
        # from being lost before the broad lexical/vector reranker sees it.
        max_model_queries = int(self.search_cfg.get("max_model_queries", 3))
        for model in sorted(query_models, key=lambda value: (-len(value), value))[:max_model_queries]:
            for source in ("master", "temp"):
                searches.append(self._model_search(model, source))
                search_metadata.append(("model", source))
        if query_numbers:
            numeric_query = " ".join(sorted(query_numbers))
            for source in ("master", "temp"):
                searches.append(self._numeric_search(numeric_query, source))
                search_metadata.append(("numeric", source))
        for brand in sorted(query_brands)[:1]:
            for source in ("master", "temp"):
                searches.append(self._brand_search(str(features["normalized"]), brand, source))
                search_metadata.append(("brand", source))

        response = self.client.multi_search(searches)
        result_sets = response.get("results", [])
        candidates: dict[str, dict[str, Any]] = {}
        main_rank: dict[str, int] = {}
        for result, (search_type, source) in zip(result_sets, search_metadata, strict=False):
            for rank, hit in enumerate(result.get("hits", []), start=1):
                document = hit.get("document", {})
                material_id = str(document.get("materialId", ""))
                if not material_id:
                    continue
                channel = f"{source}:{search_type}"
                entry = candidates.setdefault(material_id, {
                    "hit": hit,
                    "trusted_erp": False,
                    "retrieval_channels": set(),
                    "channel_ranks": {},
                })
                entry["retrieval_channels"].add(channel)
                previous_rank = entry["channel_ranks"].get(channel)
                if previous_rank is None or rank < previous_rank:
                    entry["channel_ranks"][channel] = rank
                if search_type == "erp":
                    entry["trusted_erp"] = True
                elif search_type == "main":
                    entry["hit"] = hit
                    main_rank[material_id] = rank
                if not document.get("source"):
                    document["source"] = source

        query_words = set(str(features["normalized"]).split())
        rules = self.search_cfg["rules"]
        ranked: list[RankedHit] = []

        for initial_rank, (material_id, candidate) in enumerate(candidates.items(), start=1):
            hit = candidate["hit"]
            document = hit.get("document", {})
            document_words = set(" ".join((
                str(document.get("categoryNameNormalized", "")),
                str(document.get("brandNameNormalized", "")),
                str(document.get("productNameNormalized", "")),
                str(document.get("productSpecificationNormalized", "")),
                " ".join(document.get("modelTokens") or []),
            )).split())
            overlap_score = len(query_words & document_words) / max(len(query_words), 1)
            match_info = hit.get("text_match_info") or {}
            matched_tokens = int(match_info.get("tokens_matched", 0) or 0)
            lexical_score = max(overlap_score, min(1.0, matched_tokens / max(len(query_words), 1)))
            distance = hit.get("vector_distance")
            vector_score = max(0.0, min(1.0, 1.0 - float(distance))) if distance is not None else 0.0
            score = (1.0 - float(self.search_cfg["vector_alpha"])) * lexical_score
            if vector is not None:
                score += float(self.search_cfg["vector_alpha"]) * vector_score
            # Typesense's fused order is a small tie-breaker, never the confidence
            # signal itself; rank-normalizing alone would make every top hit score 1.
            rank = main_rank.get(material_id)
            if rank is not None:
                source = str(document.get("source", "master"))
                source_candidate_k = int(self.search_cfg.get(f"{source}_candidate_k", self.search_cfg["candidate_k"]))
                score += 0.05 * (1.0 - ((rank - 1) / max(source_candidate_k, 1)))
            explanation: list[str] = []

            erp_codes = set(document.get("companyERPCodeNormalized") or [])
            doc_models = set(document.get("modelTokens") or [])
            exact_erp = bool(candidate["trusted_erp"] and trusted_erp_codes & erp_codes)
            if exact_erp:
                lexical_score = max(lexical_score, 1.0)
                score = max(score, (1.0 - float(self.search_cfg["vector_alpha"])) * lexical_score)
                score += float(rules["exact_erp_bonus"])
                explanation.append("trusted exact ERP code")
            model_overlap = query_models & doc_models
            if model_overlap and not exact_erp:
                score += float(rules["exact_model_bonus"])
                explanation.append("model match")
                model_ranks = [
                    rank for channel, rank in candidate["channel_ranks"].items()
                    if channel.endswith(":model")
                ]
                if model_ranks:
                    quality = 1.0 - ((min(model_ranks) - 1) / max(int(self.search_cfg.get("structured_candidate_k", 40)), 1))
                    score += float(rules.get("exact_model_retrieval_bonus", 0.0)) * max(0.0, quality)
                    explanation.append("exact model retrieval")

            doc_brand = str(document.get("brandNameNormalized", ""))
            if query_brands and doc_brand in query_brands:
                score += float(rules["brand_match_bonus"])
                explanation.append("brand match")
                brand_ranks = [
                    rank for channel, rank in candidate["channel_ranks"].items()
                    if channel.endswith(":brand")
                ]
                if brand_ranks:
                    quality = 1.0 - ((min(brand_ranks) - 1) / max(int(self.search_cfg.get("brand_candidate_k", 40)), 1))
                    score += float(rules.get("brand_filtered_retrieval_bonus", 0.0)) * max(0.0, quality)
                    explanation.append("brand-filtered retrieval")
            elif query_brands and doc_brand and doc_brand not in self.brands.generic:
                score -= float(rules["brand_conflict_penalty"])
                explanation.append("brand conflict penalty")

            matches, conflicts = numeric_conflicts(query_numbers, document.get("numericTokens") or [])
            if matches:
                score += float(rules["numeric_coverage_bonus"]) * min(1.0, matches / max(len(query_numbers), 1))
                explanation.append(f"{matches} numeric specification match(es)")
                numeric_ranks = [
                    rank for channel, rank in candidate["channel_ranks"].items()
                    if channel.endswith(":numeric")
                ]
                if numeric_ranks and not conflicts:
                    quality = 1.0 - ((min(numeric_ranks) - 1) / max(int(self.search_cfg.get("structured_candidate_k", 40)), 1))
                    score += float(rules.get("exact_numeric_retrieval_bonus", 0.0)) * max(0.0, quality)
                    explanation.append("strict numeric retrieval")
            if conflicts:
                score -= float(rules["numeric_conflict_penalty"]) * min(1.0, conflicts / max(len(query_numbers), 1))
                explanation.append(f"{conflicts} numeric specification conflict(s)")
            if not explanation:
                explanation.append("lexical/semantic relevance")
            ranked.append(RankedHit(
                document=document,
                lexical_score=lexical_score,
                vector_score=vector_score,
                final_score=score,
                explanation=explanation,
                initial_rank=initial_rank,
                retrieval_channels=sorted(candidate["retrieval_channels"]),
                channel_ranks=dict(sorted(candidate["channel_ranks"].items())),
            ))

        ranked.sort(key=self._score_key, reverse=True)

        if self.cross_encoder is not None and ranked:
            pairs = [
                (query, " ".join((
                    str(item.document.get("brandName", "")),
                    str(item.document.get("categoryName", "")),
                    str(item.document.get("productName", "")),
                    str(item.document.get("productSpecification", "")),
                )))
                for item in ranked
            ]
            logits = np.asarray(self.cross_encoder.predict(pairs), dtype=float)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            weight = float(self.search_cfg["cross_encoder_weight"])
            for item, probability in zip(ranked, probabilities, strict=True):
                if "trusted exact ERP code" not in item.explanation:
                    item.final_score += weight * float(probability)
                    item.explanation.append("cross-encoder reranked")
            ranked.sort(key=self._score_key, reverse=True)

        primary_convincing, primary_score, primary_margin = self._primary_is_convincing(ranked)
        source_bonus = float(self.search_cfg.get(
            "primary_source_bonus_strong" if primary_convincing else "primary_source_bonus_weak",
            0.0,
        ))
        if source_bonus:
            for item in ranked:
                if item.document.get("source") == "master":
                    item.final_score += source_bonus
                    item.explanation.append("primary-master preference")
            ranked.sort(key=self._score_key, reverse=True)

        ranked = self._deduplicate(ranked, primary_convincing)
        top = self._select_source_balanced_top(ranked, primary_convincing)
        threshold = float(self.search_cfg["confidence_threshold"])
        no_match = not top or top[0].final_score < threshold
        return {
            "query_text": query,
            "normalized_query": features["normalized"],
            "detected_brands": sorted(query_brands),
            "query_model_tokens": sorted(query_models),
            "query_numeric_tokens": sorted(query_numbers),
            "trusted_erp_candidates": sorted(trusted_erp_codes),
            "source_strategy": "primary_confident" if primary_convincing else "temp_fallback",
            "primary_score": round(primary_score, 6),
            "primary_margin": round(primary_margin, 6),
            "no_confident_match": no_match,
            "results": [item.as_dict() for item in top],
        }


def load_corpus_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def aggregate_metrics(rows: list[dict[str, Any]], eligible_only: bool = False) -> dict[str, Any]:
    selected = [row for row in rows if row["corpus_reachable"]] if eligible_only else rows
    count = len(selected)
    if not count:
        return {"queries": 0, "top1_accuracy": None, "recall_at_5": None, "recall_at_10": None, "mrr_at_10": None}
    accepted = [row for row in selected if not row["no_confident_match"]]
    return {
        "queries": count,
        "top1_accuracy": sum(row["top1_correct"] for row in selected) / count,
        "raw_top1_accuracy_before_abstention": sum(row["raw_top1_correct"] for row in selected) / count,
        "recall_at_5": sum(row["recall_at_5"] for row in selected) / count,
        "recall_at_10": sum(row["recall_at_10"] for row in selected) / count,
        "mrr_at_10": sum(row["reciprocal_rank"] for row in selected) / count,
        "raw_recall_at_5_before_abstention": sum(row["raw_recall_at_5"] for row in selected) / count,
        "raw_recall_at_10_before_abstention": sum(row["raw_recall_at_10"] for row in selected) / count,
        "raw_mrr_at_10_before_abstention": sum(row["raw_reciprocal_rank"] for row in selected) / count,
        "no_confident_match_rate": sum(row["no_confident_match"] for row in selected) / count,
        "accepted_query_rate": len(accepted) / count,
        "accepted_top1_precision": (
            sum(row["top1_correct"] for row in accepted) / len(accepted) if accepted else None
        ),
    }


def read_evaluation_frame(evaluation_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(evaluation_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        print(
            f"WARNING: {evaluation_path.name} is not valid UTF-8; retrying with Windows-1252",
            file=sys.stderr,
        )
        frame = pd.read_csv(evaluation_path, dtype=str, keep_default_na=False, encoding="cp1252")
    required = {"query_text", "materialId"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation file must contain {sorted(required)}; found {frame.columns.tolist()}")
    frame = frame.copy()
    frame["query_text"] = frame["query_text"].map(clean_scalar)
    frame["materialId"] = frame["materialId"].map(clean_scalar)
    return frame[(frame["query_text"] != "") & (frame["materialId"] != "")].reset_index(drop=True)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def expand_evaluation_rows(
    evaluation_frame: pd.DataFrame,
    query_result_rows: list[dict[str, Any]],
    corpus_ids: set[str] | None,
) -> list[dict[str, Any]]:
    """Expand one searched query into one row per original expected material ID."""
    query_results: dict[str, dict[str, Any]] = {}
    for row in query_result_rows:
        query = clean_scalar(row.get("query_text", ""))
        if query in query_results:
            raise ValueError(
                "The query-results input contains duplicate query_text rows. "
                "Use the collated query-level output, not an already-expanded file."
            )
        query_results[query] = row

    expanded: list[dict[str, Any]] = []
    missing_queries: list[str] = []
    for source_row in evaluation_frame.to_dict("records"):
        query = clean_scalar(source_row.get("query_text", ""))
        expected_material_id = clean_scalar(source_row.get("materialId", ""))
        query_row = query_results.get(query)
        if query_row is None:
            missing_queries.append(query)
            continue

        row = dict(query_row)
        predicted = [
            clean_scalar(row.get(f"predicted_material_id_{rank}", ""))
            for rank in range(1, 11)
        ]
        predicted = [material_id for material_id in predicted if material_id]
        raw_rank = next(
            (rank for rank, material_id in enumerate(predicted, start=1) if material_id == expected_material_id),
            0,
        )
        no_match = _as_bool(row.get("no_confident_match", False))
        rank = 0 if no_match else raw_rank

        row.update({
            "query_text": query,
            # Keep the established column name for compatibility, but every
            # row now contains exactly one material ID and never a semicolon list.
            "expected_material_ids": expected_material_id,
            "corpus_reachable": True if corpus_ids is None else expected_material_id in corpus_ids,
            "no_confident_match": no_match,
            "rank_of_first_correct": rank or "",
            "raw_rank_of_first_correct": raw_rank or "",
            "top1_correct": int(bool(rank) and rank == 1),
            "raw_top1_correct": int(bool(raw_rank) and raw_rank == 1),
            "recall_at_5": int(bool(rank) and rank <= 5),
            "recall_at_10": int(bool(rank) and rank <= 10),
            "raw_recall_at_5": int(bool(raw_rank) and raw_rank <= 5),
            "raw_recall_at_10": int(bool(raw_rank) and raw_rank <= 10),
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "raw_reciprocal_rank": (1.0 / raw_rank) if raw_rank else 0.0,
        })
        expanded.append(row)

    if missing_queries:
        examples = ", ".join(repr(value) for value in missing_queries[:3])
        raise ValueError(
            f"Could not find query-level results for {len(missing_queries)} evaluation rows. "
            f"Examples: {examples}"
        )
    return expanded


def expand_existing_evaluation(
    evaluation_path: Path,
    query_results_path: Path,
    output_path: Path,
    corpus_ids_path: Path | None,
) -> dict[str, Any]:
    frame = read_evaluation_frame(evaluation_path)
    query_results = pd.read_csv(query_results_path, dtype=str, keep_default_na=False).to_dict("records")
    corpus_ids = load_corpus_ids(corpus_ids_path)
    expanded = expand_evaluation_rows(frame, query_results, corpus_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(expanded).to_csv(output_path, index=False)
    summary = {
        "evaluation_rows": len(frame),
        "unique_queries": frame["query_text"].nunique(),
        "output_rows": len(expanded),
        "expected_material_ids_are_individual": True,
        "overall": aggregate_metrics(expanded),
        "output": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


def evaluate(
    searcher: HybridSearcher,
    evaluation_path: Path,
    output_dir: Path,
    corpus_ids_path: Path | None,
    max_queries: int | None,
    workers: int = 2,
) -> dict[str, Any]:
    frame = read_evaluation_frame(evaluation_path)
    grouped = frame.groupby("query_text", sort=False)["materialId"].agg(lambda values: sorted(set(values)))
    if max_queries is not None:
        grouped = grouped.iloc[:max_queries]
    corpus_ids = load_corpus_ids(corpus_ids_path)

    grouped_items = list(grouped.items())
    precomputed_vectors: list[list[float] | None] = [None] * len(grouped_items)
    if searcher.model is not None and grouped_items:
        print(f"encoding_queries_in_batches={len(grouped_items):,}", file=sys.stderr)
        encoded = searcher.model.encode(
            [query for query, _ in grouped_items],
            batch_size=int(searcher.config["embedding"].get("evaluation_batch_size", 128)),
            normalize_embeddings=bool(searcher.config["embedding"].get("normalize_embeddings", True)),
            show_progress_bar=True,
        )
        precomputed_vectors = [
            np.asarray(vector, dtype=np.float32).tolist()
            for vector in encoded
        ]

    workers = max(1, min(int(workers), 4))
    if searcher.cross_encoder is not None and workers != 1:
        print("cross-encoder mode forces --workers 1", file=sys.stderr)
        workers = 1
    print(f"typesense_parallel_workers={workers}", file=sys.stderr)

    responses: list[dict[str, Any] | None] = [None] * len(grouped_items)

    def run_search(position: int) -> tuple[int, dict[str, Any]]:
        query = grouped_items[position][0]
        return position, searcher.search(query, precomputed_vector=precomputed_vectors[position])

    if workers == 1:
        for position in range(len(grouped_items)):
            _, response = run_search(position)
            responses[position] = response
            completed = position + 1
            if completed % 100 == 0 or completed == len(grouped_items):
                print(f"evaluated={completed:,}/{len(grouped_items):,}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="typesense-eval") as executor:
            futures = [executor.submit(run_search, position) for position in range(len(grouped_items))]
            completed = 0
            for future in as_completed(futures):
                position, response = future.result()
                responses[position] = response
                completed += 1
                if completed % 100 == 0 or completed == len(grouped_items):
                    print(f"evaluated={completed:,}/{len(grouped_items):,}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    query_result_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for position, (query, expected_list) in enumerate(grouped_items):
        response = responses[position]
        if response is None:
            raise RuntimeError(f"Missing Typesense response for evaluation query {position + 1}: {query}")
        candidates = response["results"]
        predicted = [str(item["materialId"]) for item in candidates]
        expected = set(expected_list)
        raw_first_rank = next((rank for rank, material_id in enumerate(predicted, start=1) if material_id in expected), 0)
        returned = [] if response["no_confident_match"] else predicted
        first_rank = next((rank for rank, material_id in enumerate(returned, start=1) if material_id in expected), 0)
        reachable = True if corpus_ids is None else bool(expected & corpus_ids)
        row: dict[str, Any] = {
            "query_text": query,
            "expected_material_ids": ";".join(expected_list),
            "corpus_reachable": reachable,
            "no_confident_match": bool(response["no_confident_match"]),
            "source_strategy": response.get("source_strategy", ""),
            "primary_score": response.get("primary_score", ""),
            "primary_margin": response.get("primary_margin", ""),
            "trusted_erp_candidates": ";".join(response.get("trusted_erp_candidates", [])),
            "top1_source": candidates[0].get("source", "") if candidates else "",
            "master_results_in_top10": sum(item.get("source") == "master" for item in candidates[:10]),
            "temp_results_in_top10": sum(item.get("source") == "temp" for item in candidates[:10]),
            "rank_of_first_correct": first_rank or "",
            "raw_rank_of_first_correct": raw_first_rank or "",
            "top1_correct": int(bool(returned) and returned[0] in expected),
            "raw_top1_correct": int(bool(predicted) and predicted[0] in expected),
            "recall_at_5": len(expected & set(returned[:5])) / len(expected),
            "recall_at_10": len(expected & set(returned[:10])) / len(expected),
            "raw_recall_at_5": len(expected & set(predicted[:5])) / len(expected),
            "raw_recall_at_10": len(expected & set(predicted[:10])) / len(expected),
            "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
            "raw_reciprocal_rank": (1.0 / raw_first_rank) if raw_first_rank else 0.0,
            "top1_lexical_score": candidates[0]["lexical_score"] if candidates else "",
            "top1_vector_score": candidates[0]["vector_score"] if candidates else "",
            "top1_final_hybrid_score": candidates[0]["final_hybrid_score"] if candidates else "",
            "top1_match_explanation": candidates[0]["match_explanation"] if candidates else "",
        }
        for rank in range(1, 11):
            row[f"predicted_material_id_{rank}"] = predicted[rank - 1] if len(predicted) >= rank else ""
        query_result_rows.append(row)
        for rank, candidate in enumerate(candidates, start=1):
            candidate_rows.append({
                "query_text": query,
                "rank": rank,
                "is_expected": candidate["materialId"] in expected,
                **candidate,
            })
    results_path = output_dir / "evaluation_results.csv"
    query_results_path = output_dir / "evaluation_query_summary.csv"
    candidates_path = output_dir / "ranked_candidates.csv"
    individual_result_rows = expand_evaluation_rows(frame, query_result_rows, corpus_ids)
    pd.DataFrame(individual_result_rows).to_csv(results_path, index=False)
    pd.DataFrame(query_result_rows).to_csv(query_results_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_path, index=False)
    summary = {
        "evaluation_rows": len(frame),
        "unique_queries": len(grouped),
        "corpus_coverage_known": corpus_ids is not None,
        "corpus_reachable_rows": sum(row["corpus_reachable"] for row in individual_result_rows),
        "corpus_reachable_queries": sum(row["corpus_reachable"] for row in query_result_rows),
        "metric_unit": "individual evaluation rows",
        "execution": {
            "batched_query_embeddings": searcher.model is not None,
            "typesense_parallel_workers": workers,
        },
        "overall": aggregate_metrics(individual_result_rows, eligible_only=False),
        "reachable_only": aggregate_metrics(individual_result_rows, eligible_only=True) if corpus_ids is not None else None,
        "unique_query_level": aggregate_metrics(query_result_rows, eligible_only=False),
        "unique_query_level_reachable_only": (
            aggregate_metrics(query_result_rows, eligible_only=True) if corpus_ids is not None else None
        ),
        "targets": {"top1_accuracy": 0.70, "recall_at_10": 0.90},
        "routing": {
            "primary_confident_queries": sum(row["source_strategy"] == "primary_confident" for row in query_result_rows),
            "temp_fallback_queries": sum(row["source_strategy"] == "temp_fallback" for row in query_result_rows),
            "average_master_results_in_top10": (
                sum(row["master_results_in_top10"] for row in query_result_rows) / len(query_result_rows)
                if query_result_rows else None
            ),
            "average_temp_results_in_top10": (
                sum(row["temp_results_in_top10"] for row in query_result_rows) / len(query_result_rows)
                if query_result_rows else None
            ),
        },
        "outputs": {
            "results_individual_rows": str(results_path.resolve()),
            "query_summary": str(query_results_path.resolve()),
            "candidates": str(candidates_path.resolve()),
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def profile_data(config: dict[str, Any], data_dir: Path) -> None:
    data_cfg = config["data"]
    for key in ("primary_master", "secondary_master", "evaluation_file"):
        path = resolve_data_path(data_cfg[key], data_dir)
        if not path.exists():
            print(json.dumps({"file": str(path), "available": False}))
            continue
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                candidates = [
                    info for info in archive.infolist()
                    if info.filename.lower().endswith(".csv") and not info.filename.startswith("__MACOSX/")
                ]
                if not candidates:
                    raise ValueError(f"No CSV file found inside {path}")
                member = max(candidates, key=lambda info: info.file_size)
                with archive.open(member) as raw:
                    header = pd.read_csv(raw, nrows=0).columns.tolist()
                with archive.open(member) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                    row_count = sum(1 for _ in csv.reader(text)) - 1
        else:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                row_count = sum(1 for _ in csv.reader(handle)) - 1
        print(json.dumps({
            "file": str(path),
            "available": True,
            "bytes": path.stat().st_size,
            "rows": row_count,
            "columns": header,
        }))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid material matching pipeline for Typesense")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Inspect file availability, row counts and columns")
    profile.add_argument("--data-dir", type=Path, default=Path.cwd())

    subparsers.add_parser("health", help="Verify direct Typesense API connectivity")

    build = subparsers.add_parser("build-index", help="Merge, normalize and optionally embed both masters")
    build.add_argument("--data-dir", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path("build/materials.jsonl"))
    build.add_argument("--corpus-ids-output", type=Path, default=Path("build/corpus_ids.txt"))
    build.add_argument("--brand-lexicon-output", type=Path, default=Path("build/brand_lexicon.json"))
    build.add_argument("--no-embeddings", action="store_true")
    build.add_argument("--require-secondary", action="store_true")
    build.add_argument("--max-documents", type=int, help="Bounded smoke-test build; omit for the full corpus")

    upgrade = subparsers.add_parser(
        "upgrade-jsonl",
        help="Rebuild normalized fields from an existing JSONL while preserving its embeddings",
    )
    upgrade.add_argument("--input", type=Path, default=Path("build/materials.jsonl"))
    upgrade.add_argument("--output", type=Path, default=Path("build/materials_primary_first.jsonl"))
    upgrade.add_argument("--max-documents", type=int, help="Bounded smoke test; omit for the full JSONL")

    index = subparsers.add_parser("index", help="Create a versioned Typesense collection and atomically switch alias")
    index.add_argument("--schema", type=Path, default=ROOT / "typesense_schema.json")
    index.add_argument("--jsonl", type=Path, default=Path("build/materials.jsonl"))
    index.add_argument("--collection-name")
    index.add_argument("--batch-size", type=int, help="Override Typesense import batch size")

    resume = subparsers.add_parser(
        "resume-index",
        help="Resume a partial collection from its current document count, then publish the alias",
    )
    resume.add_argument("--jsonl", type=Path, default=Path("build/materials_primary_first.jsonl"))
    resume.add_argument("--collection-name", required=True)
    resume.add_argument(
        "--skip-documents",
        type=int,
        help="Override automatic resume position; use only when the previous run had failed documents",
    )
    resume.add_argument("--batch-size", type=int, default=50)

    search = subparsers.add_parser("search", help="Search one query")
    search.add_argument("query")
    search.add_argument("--brand-lexicon", type=Path, default=Path("build/brand_lexicon.json"))
    search.add_argument("--lexical-only", action="store_true")
    search.add_argument("--cross-encoder", action="store_true")
    search.add_argument("--rerank-hybrid-matches", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate labelled queries against Typesense")
    evaluate_parser.add_argument("--evaluation-file", type=Path)
    evaluate_parser.add_argument("--brand-lexicon", type=Path, default=Path("build/brand_lexicon.json"))
    evaluate_parser.add_argument("--corpus-ids", type=Path, default=Path("build/corpus_ids.txt"))
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("evaluation_output"))
    evaluate_parser.add_argument("--max-queries", type=int)
    evaluate_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel Typesense requests (1-4). Use 2 for a local Mac server.",
    )
    evaluate_parser.add_argument("--lexical-only", action="store_true")
    evaluate_parser.add_argument("--cross-encoder", action="store_true")
    evaluate_parser.add_argument("--rerank-hybrid-matches", action="store_true")

    expand_parser = subparsers.add_parser(
        "expand-results",
        help="Expand an existing query-level evaluation into one row per original materialId",
    )
    expand_parser.add_argument("--evaluation-file", type=Path, required=True)
    expand_parser.add_argument("--query-results-file", type=Path, required=True)
    expand_parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("evaluation_results_individual.csv"),
    )
    expand_parser.add_argument("--corpus-ids", type=Path, default=Path("build/corpus_ids.txt"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "profile":
        profile_data(config, args.data_dir)
    elif args.command == "health":
        client = TypesenseREST(config["typesense"])
        print(json.dumps({"base_url": client.base_url, "health": client.health()}, indent=2))
    elif args.command == "build-index":
        build_index_jsonl(
            config,
            args.data_dir,
            args.output,
            args.corpus_ids_output,
            args.brand_lexicon_output,
            include_embeddings=not args.no_embeddings,
            require_secondary=args.require_secondary,
            max_documents=args.max_documents,
        )
    elif args.command == "upgrade-jsonl":
        upgrade_index_jsonl(args.input, args.output, config, max_documents=args.max_documents)
    elif args.command == "index":
        print(json.dumps(
            create_and_import(config, args.schema, args.jsonl, args.collection_name, args.batch_size),
            indent=2,
        ))
    elif args.command == "resume-index":
        print(json.dumps(
            resume_import(
                config,
                args.jsonl,
                args.collection_name,
                skip_documents=args.skip_documents,
                batch_size_override=args.batch_size,
            ),
            indent=2,
        ))
    elif args.command == "expand-results":
        corpus_path = args.corpus_ids if args.corpus_ids.exists() else None
        expand_existing_evaluation(
            args.evaluation_file,
            args.query_results_file,
            args.output_file,
            corpus_path,
        )
    elif args.command in {"search", "evaluate"}:
        if args.rerank_hybrid_matches:
            config["search"]["rerank_hybrid_matches"] = True
        searcher = HybridSearcher(
            config,
            args.brand_lexicon,
            lexical_only=args.lexical_only,
            cross_encoder=args.cross_encoder,
        )
        if args.command == "search":
            print(json.dumps(searcher.search(args.query), ensure_ascii=False, indent=2))
        else:
            evaluation_path = args.evaluation_file or resolve_data_path(config["data"]["evaluation_file"], Path.cwd())
            corpus_path = args.corpus_ids if args.corpus_ids.exists() else None
            evaluate(
                searcher,
                evaluation_path,
                args.output_dir,
                corpus_path,
                args.max_queries,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
