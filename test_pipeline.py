import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import HybridSearcher, TypesenseREST, import_jsonl_into_collection, upgrade_index_jsonl
from text_utils import load_json_map


ROOT = Path(__file__).resolve().parent


class FakeClient:
    def __init__(self, response):
        self.response = response

    def multi_search(self, searches):
        self.searches = searches
        return self.response


class FakeBrands:
    generic = {"", "generic", "unbranded"}

    def detect(self, normalized_query):
        return {"skf"} if "skf" in normalized_query.split() else set()


def hit(material_id, brand, model, size, erp=None, distance=0.25, matched=4, source="master", fingerprint=None):
    return {
        "document": {
            "materialId": material_id,
            "categoryName": "Bearings",
            "brandName": brand.upper(),
            "productName": "BALL BEARING",
            "productSpecification": f"{model} {size}",
            "companyERPCode": erp or "",
            "source": source,
            "categoryNameNormalized": "bearing",
            "brandNameNormalized": brand,
            "productNameNormalized": "ball bearing",
            "productSpecificationNormalized": f"{model} {size}",
            "companyERPCodeNormalized": [erp] if erp else [],
            "modelTokens": [model],
            "numericTokens": [size],
            "physicalFingerprint": fingerprint or f"fp-{material_id}",
        },
        "text_match_info": {"tokens_matched": matched},
        "vector_distance": distance,
    }


class HybridRankingTests(unittest.TestCase):
    def make_searcher(self, response):
        searcher = HybridSearcher.__new__(HybridSearcher)
        searcher.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        searcher.search_cfg = searcher.config["search"]
        searcher.client = FakeClient(response)
        searcher.abbreviations = load_json_map(ROOT / "abbreviations.json")
        searcher.aliases = load_json_map(ROOT / "brand_aliases.json")
        searcher.brands = FakeBrands()
        searcher.lexical_only = True
        searcher.model = None
        searcher.cross_encoder = None
        return searcher

    def test_unit_and_model_tokens_do_not_trigger_erp_route(self):
        primary = hit("A", "skf", "6205", "10mm", matched=5)
        misleading = hit("B", "generic", "other", "10mm", erp="10mm", matched=1, source="temp")
        response = {"results": [{"hits": [primary]}, {"hits": [misleading]}]}
        searcher = self.make_searcher(response)
        result = searcher.search("SKF bearing 6205 10mm")
        self.assertFalse(any(
            request["query_by"] == "companyERPCodeNormalized"
            for request in searcher.client.searches
        ))
        self.assertEqual(result["results"][0]["materialId"], "A")
        self.assertNotIn("ERP", result["results"][0]["match_explanation"])

    def test_exact_short_model_channel_recovers_candidate_missing_from_main_search(self):
        exact = hit("A", "skf", "831", "130mm", matched=1)
        response = {
            "results": [
                {"hits": []},
                {"hits": []},
                {"hits": [exact]},
                {"hits": []},
            ]
        }
        searcher = self.make_searcher(response)
        result = searcher.search("screwdriver model no 831")
        self.assertEqual(result["results"][0]["materialId"], "A")
        self.assertIn("exact model retrieval", result["results"][0]["match_explanation"])
        self.assertTrue(any(
            request["query_by"] == "modelTokens" and request["q"] == "831"
            for request in searcher.client.searches
        ))

    def test_explicit_erp_uses_gated_exact_route(self):
        exact = hit("A", "skf", "6205", "10mm", erp="m3900153", matched=1)
        temp = hit("B", "skf", "6205", "10mm", matched=3, source="temp")
        response = {
            "results": [
                {"hits": [exact]},
                {"hits": []},
                {"hits": [exact]},
                {"hits": [temp]},
            ]
        }
        result = self.make_searcher(response).search("ERP code: M3900153")
        self.assertEqual(result["results"][0]["materialId"], "A")
        self.assertIn("trusted exact ERP code", result["results"][0]["match_explanation"])

    def test_conflicting_numeric_dimension_is_penalized(self):
        correct = hit("A", "skf", "6205", "10mm", distance=0.25, matched=4)
        wrong = hit("B", "skf", "6205", "12mm", distance=0.10, matched=4)
        temp = hit("T", "generic", "other", "9mm", matched=1, source="temp")
        response = {"results": [{"hits": [correct, wrong]}, {"hits": [temp]}]}
        result = self.make_searcher(response).search("SKF bearing 6205 10mm")
        self.assertEqual(result["results"][0]["materialId"], "A")
        wrong_result = next(item for item in result["results"] if item["materialId"] == "B")
        self.assertIn("numeric specification conflict", wrong_result["match_explanation"])

    def test_weak_primary_allows_temp_to_rank_first_but_keeps_both_sources(self):
        weak_primary = hit("A", "generic", "other", "12mm", matched=1)
        strong_temp = hit("B", "skf", "6205", "10mm", matched=5, source="temp")
        response = {"results": [{"hits": [weak_primary]}, {"hits": [strong_temp]}]}
        result = self.make_searcher(response).search("SKF bearing 6205 10mm")
        self.assertEqual(result["source_strategy"], "temp_fallback")
        self.assertEqual(result["results"][0]["materialId"], "B")
        self.assertEqual({item["source"] for item in result["results"]}, {"master", "temp"})

    def test_strong_primary_still_keeps_a_temp_result(self):
        primary = hit("A", "skf", "6205", "10mm", matched=5)
        temp = hit("B", "skf", "6205", "10mm", matched=4, source="temp")
        response = {"results": [{"hits": [primary]}, {"hits": [temp]}]}
        result = self.make_searcher(response).search("SKF bearing 6205 10mm")
        self.assertEqual(result["source_strategy"], "primary_confident")
        self.assertEqual(result["results"][0]["source"], "master")
        self.assertEqual({item["source"] for item in result["results"]}, {"master", "temp"})


class DirectApiConfigurationTests(unittest.TestCase):
    def test_typesense_url_environment_override(self):
        config = {
            "base_url_env": "TYPESENSE_URL",
            "api_key_env": "TYPESENSE_API_KEY",
            "protocol": "http",
            "host": "localhost",
            "port": 8108,
        }
        with patch.dict(os.environ, {
            "TYPESENSE_URL": "https://search.example.typesense.net/",
            "TYPESENSE_API_KEY": "test-key",
        }, clear=False):
            client = TypesenseREST(config)
        self.assertEqual(client.base_url, "https://search.example.typesense.net")

    def test_read_timeout_is_retried(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"ok": true}'

        config = {
            "api_key_env": "TYPESENSE_API_KEY",
            "protocol": "http",
            "host": "localhost",
            "port": 8108,
            "connection_timeout_seconds": 1,
        }
        with patch.dict(os.environ, {"TYPESENSE_API_KEY": "test-key"}, clear=False):
            client = TypesenseREST(config)
        with patch("pipeline.urllib.request.urlopen", side_effect=[TimeoutError("slow"), FakeResponse()]):
            with patch("pipeline.time.sleep"):
                result = client.request("GET", "/health", retries=1)
        self.assertEqual(result, {"ok": True})


class JsonlUpgradeTests(unittest.TestCase):
    def test_upgrade_preserves_embedding_and_separates_erp_from_models(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old.jsonl"
            output = Path(directory) / "new.jsonl"
            source.write_text(json.dumps({
                "materialId": "1",
                "categoryName": "Bearings",
                "brandName": "SKF",
                "productName": "BALL BEARING",
                "productSpecification": "6205-ZZ 10mm",
                "companyERPCode": "ERP-001",
                "source": "master",
                "modelTokens": ["6205zz", "erp001"],
                "embedding": [0.1, 0.2],
            }) + "\n", encoding="utf-8")
            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            summary = upgrade_index_jsonl(source, output, config)
            upgraded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["embeddings_preserved"], 1)
        self.assertEqual(upgraded["embedding"], [0.1, 0.2])
        self.assertIn("6205zz", upgraded["modelTokens"])
        self.assertNotIn("erp001", upgraded["modelTokens"])

    def test_resume_import_skips_existing_prefix_and_publishes_alias(self):
        class ImportClient:
            def __init__(self):
                self.imported_ids = []
                self.alias_published = False

            def request(self, method, path, payload=None, content_type="application/json", **kwargs):
                if "/documents/import" in path:
                    documents = [json.loads(line) for line in payload.splitlines()]
                    self.imported_ids.extend(document["id"] for document in documents)
                    return "\n".join(json.dumps({"success": True}) for _ in documents)
                if method == "GET" and path.startswith("/collections/"):
                    return {"num_documents": 3}
                if method == "PUT" and path.startswith("/aliases/"):
                    self.alias_published = True
                    return {"collection_name": "partial"}
                raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "materials.jsonl"
            source.write_text("\n".join(json.dumps({"id": str(i)}) for i in range(1, 4)) + "\n")
            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            client = ImportClient()
            result = import_jsonl_into_collection(
                config,
                source,
                "partial",
                client,
                skip_documents=2,
                batch_size_override=1,
            )
        self.assertEqual(client.imported_ids, ["3"])
        self.assertTrue(client.alias_published)
        self.assertEqual(result["final_documents"], 3)


if __name__ == "__main__":
    unittest.main()
