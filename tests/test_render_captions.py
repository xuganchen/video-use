"""Caption chunking and joining across scripts.

Both defaults were tuned for English and produce unreadable Chinese: ASR emits
one token per CJK character, so a 2-token caption is two glyphs, and joining
with spaces gives "的 使 用 了" instead of "的使用了".
"""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "render.py"
SPEC = importlib.util.spec_from_file_location("video_use_render_caps", MODULE_PATH)
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


def toks(*items):
    return [{"text": t} for t in items]


class AutoChunkTests(unittest.TestCase):
    def test_english_keeps_two_words(self):
        words = toks(*"ninety percent of what a web agent does is wasted".split())
        self.assertEqual(render.auto_chunk_words(words), 2)

    def test_chinese_characters_chunk_far_higher(self):
        words = toks(*"这就是刚才大家看到的二十年不太变的生产率")
        self.assertGreaterEqual(render.auto_chunk_words(words), 8)

    def test_empty_input_falls_back_to_two(self):
        self.assertEqual(render.auto_chunk_words([]), 2)
        self.assertEqual(render.auto_chunk_words(toks("", "  ")), 2)

    def test_mostly_latin_with_some_cjk_stays_english(self):
        words = toks("the", "team", "at", "字节", "shipped", "it", "last", "week")
        self.assertEqual(render.auto_chunk_words(words), 2)


class JoinTokensTests(unittest.TestCase):
    def test_cjk_runs_join_without_spaces(self):
        self.assertEqual(render.join_tokens(["的", "使", "用", "了"]), "的使用了")

    def test_english_keeps_spaces(self):
        self.assertEqual(render.join_tokens(["ninety", "percent"]), "ninety percent")

    def test_latin_embedded_in_cjk_keeps_its_spaces(self):
        # "AI" must not fuse into the surrounding characters, but ASR splits
        # "95%" into two tokens and the percent sign must not be stranded.
        self.assertEqual(
            render.join_tokens(["95", "%", "的", "使", "用", "了", "AI", "的"]),
            "95% 的使用了 AI 的",
        )

    def test_no_space_before_trailing_punctuation(self):
        self.assertEqual(render.join_tokens(["hello", ",", "world", "!"]), "hello, world!")

    def test_fullwidth_comma_does_not_open_a_gap(self):
        # "，" sits in the fullwidth block, not the Han block. Treating it as
        # non-CJK put a space after every Chinese comma.
        self.assertEqual(render.join_tokens(["公", "司", "，", "大家"]), "公司，大家")

    def test_empty_tokens_are_dropped(self):
        self.assertEqual(render.join_tokens(["公", "", "司"]), "公司")

    def test_empty_list(self):
        self.assertEqual(render.join_tokens([]), "")


class CjkDetectionTests(unittest.TestCase):
    def test_detects_han_kana_hangul(self):
        for s in ("公司", "ひらがな", "한글"):
            self.assertTrue(render._is_cjk(s), s)

    def test_latin_and_digits_are_not_cjk(self):
        for s in ("AI", "95", "%", "hello"):
            self.assertFalse(render._is_cjk(s), s)


if __name__ == "__main__":
    unittest.main()
