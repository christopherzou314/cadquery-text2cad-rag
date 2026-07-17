import unittest

from src.text2cad.rag import (
    FULL_LIBRARY_PATH,
    LIGHTWEIGHT_LIBRARY_PATH,
    format_reference_context,
    retrieve_references,
)


class RagTests(unittest.TestCase):
    def test_airplane_retrieves_complex_shape_references(self):
        matches = retrieve_references("Create a simplified airplane with wings", top_k=3)
        ids = {match.id for match in matches}

        self.assertIn("loft_and_tapered_forms", ids)
        self.assertIn("assemblies", ids)

    def test_chinese_hole_prompt_retrieves_hole_reference(self):
        matches = retrieve_references("生成一个带有四个安装孔的板", top_k=2)

        self.assertEqual(matches[0].id, "holes_and_patterns")

    def test_formatted_context_contains_examples(self):
        matches = retrieve_references("open-top box with wall thickness", top_k=1)
        context = format_reference_context(matches)

        self.assertIn("Guidance:", context)
        self.assertIn("```python", context)

    def test_short_words_do_not_match_substrings(self):
        matches = retrieve_references(
            "Create an open-top box with 3 mm wall thickness",
            top_k=3,
        )
        ids = {match.id for match in matches}

        self.assertNotIn("transformations_and_symmetry", ids)

    def test_lightweight_mode_only_uses_original_library(self):
        matches = retrieve_references(
            "Use Workplane loft to create lofted sections",
            top_k=5,
            library_path=LIGHTWEIGHT_LIBRARY_PATH,
        )

        self.assertTrue(matches)
        self.assertTrue(all(not match.id.startswith("api.") for match in matches))

    def test_full_mode_can_retrieve_api_reference(self):
        matches = retrieve_references(
            "Use Workplane loft to create lofted sections",
            top_k=5,
            library_path=FULL_LIBRARY_PATH,
        )

        self.assertIn("api.Workplane.loft", {match.id for match in matches})


if __name__ == "__main__":
    unittest.main()
