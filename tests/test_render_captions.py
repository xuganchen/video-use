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


def chars(text, start=0.0, dur=0.2, gap=0.0):
    """One word dict per character, the shape Chinese ASR actually returns."""
    out, at = [], start
    for c in text:
        out.append({"text": c, "start": round(at, 3), "end": round(at + dur, 3),
                    "type": "word"})
        at += dur + gap
    return out


class GlyphWidthTests(unittest.TestCase):
    def test_cjk_is_a_full_cell_latin_is_half(self):
        self.assertEqual(render.glyph_width("公司"), 2.0)
        self.assertEqual(render.glyph_width("AI"), 1.0)
        self.assertEqual(render.glyph_width("95%的"), 2.5)


class FullwidthPunctTests(unittest.TestCase):
    def test_ascii_marks_after_cjk_are_widened(self):
        self.assertEqual(render.fullwidth_punct("公司, 大家"), "公司，大家")
        self.assertEqual(render.fullwidth_punct("是的."), "是的。")

    def test_latin_punctuation_is_left_alone(self):
        self.assertEqual(render.fullwidth_punct("hello, world"), "hello, world")

    def test_trailing_soft_punctuation_is_dropped(self):
        self.assertEqual(render.fullwidth_punct("那个时候，"), "那个时候")


class LatinRunTests(unittest.TestCase):
    def test_split_latin_fuses_into_an_acronym(self):
        # Chinese ASR emits "M" + "IT"; unfused this reads "是 M IT 做的".
        self.assertEqual(
            render.join_tokens(["是", "M", "IT", "做", "的"], fuse_latin=True),
            "是 MIT 做的",
        )

    def test_fusing_is_off_by_default_so_english_keeps_its_spaces(self):
        self.assertEqual(render.join_tokens(["ninety", "percent"]), "ninety percent")

    def test_fusing_never_welds_cjk_to_latin(self):
        self.assertEqual(
            render.join_tokens(["95", "%", "的", "公", "司"], fuse_latin=True),
            "95% 的公司",
        )


class MajorityCjkTests(unittest.TestCase):
    def test_routes_chinese_to_the_cjk_path(self):
        self.assertTrue(render.majority_cjk(chars("这就是刚才大家看到的")))

    def test_leaves_english_on_the_chunker(self):
        self.assertFalse(render.majority_cjk(toks(*"the team shipped it".split())))

    def test_no_words_is_not_cjk(self):
        self.assertFalse(render.majority_cjk([]))


class CjkCuesTests(unittest.TestCase):
    """Punctuation and pauses decide the cue, never a token count."""

    def test_breaks_on_hard_punctuation(self):
        w = chars("是MIT做的。他们问了一千家公司。")
        cues = render.cjk_cues(w, 0.0, 100.0, 0.0)
        self.assertEqual([c[2] for c in cues], ["是 MIT 做的。", "他们问了一千家公司。"])

    def test_breaks_on_a_speech_gap(self):
        a = chars("平均的团队规模", start=0.0)
        b = chars("在那个时候", start=a[-1]["end"] + 1.0)   # 1s pause
        cues = render.cjk_cues(a + b, 0.0, 100.0, 0.0)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0][2], "平均的团队规模")

    def test_never_exceeds_the_glyph_budget(self):
        # 60 characters, no punctuation and no pauses: must still split.
        cues = render.cjk_cues(chars("这" * 60), 0.0, 100.0, 0.0)
        self.assertGreater(len(cues), 1)
        for _, _, text in cues:
            self.assertLessEqual(render.glyph_width(text), render.CUE_MAX_WIDTH + 2)

    def test_never_splits_inside_a_word(self):
        if render.word_boundaries("测试") is None:
            self.skipTest("jieba not installed")
        run = "传统的护城河品牌生产力悖论而是一个非常重要的问题所以我们要看清楚这件事情的本质"
        cues = render.cjk_cues(chars(run), 0.0, 100.0, 0.0)
        self.assertGreater(len(cues), 1)
        legal = render.word_boundaries(run)
        pos = 0
        for _, _, text in cues[:-1]:
            pos += len(text)
            self.assertIn(pos, legal, f"cut inside a word at {pos}: {text!r}")

    def test_times_land_on_the_output_timeline(self):
        # A cue's time is word.start - segment_start + segment_offset.
        cues = render.cjk_cues(chars("公司。", start=10.0), 10.0, 20.0, 5.0)
        self.assertAlmostEqual(cues[0][0], 5.0, places=3)

    def test_no_space_before_a_chinese_comma(self):
        cues = render.cjk_cues(chars("那个时候,大概是这样的一个情况。"), 0.0, 100.0, 0.0)
        for _, _, text in cues:
            self.assertNotIn(" ，", text)
            self.assertNotIn(" 。", text)


class SplitAtWidestGapTests(unittest.TestCase):
    def test_prefers_the_longest_pause_among_legal_points(self):
        w = chars("这" * 40)
        # widen one gap well past the halfway mark
        w[30]["end"] = w[31]["start"] - 2.0
        k = render.split_at_widest_gap(w)
        self.assertEqual(k, 30)


if __name__ == "__main__":
    unittest.main()
