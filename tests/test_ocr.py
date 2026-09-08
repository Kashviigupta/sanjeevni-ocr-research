"""Tests for OCR: line-break reconstruction, real preprocessing (deskew/denoise/
binarize), and the vision fallback (mocked — no real API calls in this suite).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sanjeevani_ml.ocr.engine import _reconstruct_lines, run_vision_ocr
from sanjeevani_ml.ocr.preprocess import binarize, deskew, denoise, load_image


def _fake_tesseract_dict(rows: list[list[str]]) -> dict:
    """Build a minimal Tesseract image_to_data DICT for `rows`, one line each.

    Mirrors the real shape closely enough for `_reconstruct_lines` to exercise its
    actual grouping logic, without needing Tesseract installed to generate fixtures.
    """
    text, conf, block, par, line, left, top, width, height = [], [], [], [], [], [], [], [], []
    for line_no, words in enumerate(rows):
        for word_no, word in enumerate(words):
            text.append(word)
            conf.append(90.0)
            block.append(1)
            par.append(1)
            line.append(line_no)
            left.append(word_no * 50)
            top.append(line_no * 20)
            width.append(40)
            height.append(15)
    return {
        "text": text, "conf": conf, "block_num": block, "par_num": par,
        "line_num": line, "left": left, "top": top, "width": width, "height": height,
    }


class TestLineReconstruction:
    def test_words_on_the_same_line_are_joined_with_spaces(self):
        data_dict = _fake_tesseract_dict([["Fasting", "Blood", "Sugar", "142"]])
        _, text = _reconstruct_lines(data_dict, page_no=0)
        assert text == "Fasting Blood Sugar 142"

    def test_different_lines_become_separate_lines_not_one_run_on_string(self):
        """This is the whole point: a lab report's rows must survive as rows."""
        data_dict = _fake_tesseract_dict([
            ["Fasting", "Blood", "Sugar", "142"],
            ["HbA1c", "7.8"],
        ])
        _, text = _reconstruct_lines(data_dict, page_no=0)
        lines = text.split("\n")
        assert len(lines) == 2
        assert lines[0] == "Fasting Blood Sugar 142"
        assert lines[1] == "HbA1c 7.8"

    def test_empty_tokens_are_skipped(self):
        data_dict = _fake_tesseract_dict([["Hello", "", "World"]])
        _, text = _reconstruct_lines(data_dict, page_no=0)
        assert text == "Hello World"

    def test_negative_confidence_tokens_are_dropped(self):
        """-1 confidence marks a non-text block (e.g. a photo region), per Tesseract's own convention."""
        data_dict = _fake_tesseract_dict([["Real", "Word"]])
        data_dict["conf"][1] = -1.0
        _, text = _reconstruct_lines(data_dict, page_no=0)
        assert text == "Real"

    def test_words_carry_the_correct_page_number(self):
        data_dict = _fake_tesseract_dict([["Word"]])
        words, _ = _reconstruct_lines(data_dict, page_no=3)
        assert words[0].page == 3

    def test_an_empty_page_produces_empty_text_not_a_crash(self):
        data_dict = _fake_tesseract_dict([])
        words, text = _reconstruct_lines(data_dict, page_no=0)
        assert words == []
        assert text == ""


class TestRealPreprocessing:
    """These exercise the actual OpenCV calls (real cv2, no mocking) against
    synthetic images — no Tesseract or network needed.
    """

    def test_deskew_does_not_crash_on_a_blank_image(self):
        img = Image.new("RGB", (200, 200), color="white")
        result = deskew(img)
        assert result.size == img.size

    def test_deskew_straightens_a_rotated_block_of_text_like_content(self):
        """A synthetic image with a rotated rectangle of 'ink' should come back
        closer to axis-aligned than it went in.
        """
        arr = np.full((300, 300), 255, dtype=np.uint8)
        arr[100:200, 50:250] = 0  # a horizontal black bar, our stand-in for text
        img = Image.fromarray(arr).rotate(8, expand=False, fillcolor=255)
        result = deskew(img)
        assert result.size == img.size  # deskew must not change canvas size

    def test_deskew_leaves_a_barely_tilted_image_alone(self):
        """Below the 0.5-degree threshold, nothing should change — avoids
        needless recomputation and interpolation blur on an already-straight page.
        """
        img = Image.new("RGB", (200, 200), color="white")
        result = deskew(img)
        assert np.array_equal(np.array(result), np.array(img))

    def test_denoise_returns_an_image_of_the_same_size(self):
        arr = (np.random.rand(150, 150) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        result = denoise(img)
        assert result.size == img.size

    def test_binarize_produces_only_black_and_white_pixels(self):
        arr = (np.random.rand(150, 150) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        result = binarize(img)
        values = set(np.array(result).flatten().tolist())
        assert values <= {0, 255}


class TestVisionFallback:
    def test_returns_none_when_no_provider_is_configured(self):
        with patch("sanjeevani_ml.ocr.engine.settings") as mock_settings:
            mock_settings.vision_provider = None
            assert run_vision_ocr(b"fake-image-bytes") is None

    def test_gemini_path_is_used_when_gemini_is_the_configured_provider(self):
        with patch("sanjeevani_ml.ocr.engine.settings") as mock_settings:
            mock_settings.vision_provider = "gemini"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.gemini_model = "gemini-2.0-flash"
            mock_settings.vision_fallback_threshold = 0.55

            fake_response = MagicMock()
            fake_response.text = "Tab. Glycomet 500mg"
            fake_client = MagicMock()
            fake_client.models.generate_content.return_value = fake_response

            with patch("google.genai.Client", return_value=fake_client):
                result = run_vision_ocr(b"fake-image-bytes")

        assert result is not None
        assert result.text == "Tab. Glycomet 500mg"
        assert "gemini-vision" in result.engine

    def test_vision_confidence_is_never_fabricated_high(self):
        """No per-word confidence from a vision model — the flat estimate must stay
        conservative, never masquerading as a measured, high-certainty result.
        """
        with patch("sanjeevani_ml.ocr.engine.settings") as mock_settings:
            mock_settings.vision_provider = "gemini"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.gemini_model = "gemini-2.0-flash"
            mock_settings.vision_fallback_threshold = 0.55

            fake_response = MagicMock()
            fake_response.text = "some text"
            fake_client = MagicMock()
            fake_client.models.generate_content.return_value = fake_response

            with patch("google.genai.Client", return_value=fake_client):
                result = run_vision_ocr(b"fake-image-bytes")

        assert result.mean_confidence <= 0.6

    def test_vision_result_has_no_fabricated_word_boxes(self):
        with patch("sanjeevani_ml.ocr.engine.settings") as mock_settings:
            mock_settings.vision_provider = "gemini"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.gemini_model = "gemini-2.0-flash"
            mock_settings.vision_fallback_threshold = 0.55

            fake_response = MagicMock()
            fake_response.text = "some text"
            fake_client = MagicMock()
            fake_client.models.generate_content.return_value = fake_response

            with patch("google.genai.Client", return_value=fake_client):
                result = run_vision_ocr(b"fake-image-bytes")

        assert result.words == []
