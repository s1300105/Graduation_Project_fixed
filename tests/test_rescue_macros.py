"""Signature-macro stripping used by the --cpg rescue pass (stages A / B / C)."""
import hashlib
from pathlib import Path

import pytest

from utils.data.datamanager import (strip_signature_macros, rescue_variants, strip_lowercase_signature_macros,
                                    RESCUE_STAGES, LOWERCASE_SIGNATURE_MACROS)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_INPUTS = ROOT / "tests" / "fixtures" / "joern_inputs"


def test_conservative_strips_attribute_dunder_and_trailing_caps():
    code = "static int __init foo(struct x *p __user, int n UNUSED) __attribute__((unused))\n{\n  return n;\n}"
    out = strip_signature_macros(code)
    assert "__init" not in out and "__user" not in out and "UNUSED" not in out and "__attribute__" not in out
    assert out.endswith("{\n  return n;\n}")                 # body untouched
    assert "static int foo(struct x *p , int n )" in out.replace("  ", " ")


def test_conservative_keeps_caps_types():
    code = "static VALUE rb_foo(VALUE self, FILE *fp)\n{\n  return self;\n}"
    assert strip_signature_macros(code) == code


def test_aggressive_rewrites_macro_shaped_signature():
    code = "PHP_FUNCTION(foo)\n{\n  RETURN_TRUE;\n}"
    out = strip_signature_macros(code, aggressive=True)
    assert out.startswith("void PHP_FUNCTION_foo(void) {") and "RETURN_TRUE;" in out


def test_aggressive_drops_caps_tokens_in_signature_only():
    code = "static VALUE rb_foo(VALUE self)\n{\n  VALUE x = Qnil;\n  return x;\n}"
    out = strip_signature_macros(code, aggressive=True)
    assert out.split("{", 1)[0].strip() == "static rb_foo( self)"
    assert "VALUE x = Qnil;" in out


def test_rescue_variants_are_distinct_and_ordered():
    code = "int f(int a UNUSED)\n{\n  return a;\n}"
    v = rescue_variants(code)
    assert [s for s, _ in v] == ["A"]                         # B would be identical to A here, C to B
    code2 = "PHP_FUNCTION(foo)\n{\n  return;\n}"
    assert [s for s, _ in rescue_variants(code2)] == ["B"]     # A changes nothing
    assert RESCUE_STAGES == ("A", "B", "C")


# --- stage C: lowercase signature macros (FFmpeg / QEMU / APR) --------------------------

@pytest.mark.parametrize("code,head", [
    ("static av_cold void x8_vlc_init(void)\n{\n    int i;\n}", "static void x8_vlc_init(void)"),
    ("static int coroutine_fn nfs_co_flush(BlockDriverState *bs)\n\n{\n    return 0;\n}", "static int nfs_co_flush(BlockDriverState *bs)"),
    ("static av_always_inline int decode_mb_row(AVCodecContext *avctx, void *tdata)\n{\n}", "static int decode_mb_row(AVCodecContext *avctx, void *tdata)"),
    ("static void av_noinline filter_mb_edgev(uint8_t *pix, int stride) {\n}", "static void filter_mb_edgev(uint8_t *pix, int stride)"),
    ("static inline av_flatten int get_symbol_inline(RangeCoder *c)\n{\n}", "static inline int get_symbol_inline(RangeCoder *c)"),
    ("static always_inline void fload_invalid_op_excp (int op)\n\n{\n}", "static void fload_invalid_op_excp (int op)"),
    ("int attribute_align_arg avcodec_encode_video(AVCodecContext *avctx)\n{\n}", "int avcodec_encode_video(AVCodecContext *avctx)"),
    ("static int av_printf_format(3, 4) log_fmt(void *a, int l, const char *fmt, ...)\n{\n}", "static int log_fmt(void *a, int l, const char *fmt, ...)"),
    ("void g_log_x(const char *fmt, ...) G_GNUC_PRINTF(1, 2)\n{\n}", "void g_log_x(const char *fmt, ...)"),
    ("APR_DECLARE(int) foo(int x)\n{\n    return x;\n}", "int foo(int x)"),
    ("static APR_DECLARE_NONSTD(void *) bar(void)\n{\n}", "static void * bar(void)"),
    ("static APR_INLINE unsigned char *c2x(unsigned what, unsigned char *where)\n{\n}", "static unsigned char *c2x(unsigned what, unsigned char *where)"),
])
def test_stage_c_prepass_cleans_the_head_only(code, head):
    out = strip_lowercase_signature_macros(code)
    out_head, out_body = out.split("{", 1)
    assert " ".join(out_head.split()) == " ".join(head.split())
    assert "{" + out_body == code[code.index("{"):]           # body untouched


def test_stage_c_prepass_does_not_touch_identifiers_or_bodies():
    code = "int av_coldstart(int coroutine_fnord)\n{\n    av_cold = 1; APR_DECLARE(int) y;\n    return coroutine_fn;\n}"
    assert strip_lowercase_signature_macros(code) == code     # word boundaries; body untouched
    assert strip_lowercase_signature_macros("PHP_FUNCTION(foo)\n{\n}") == "PHP_FUNCTION(foo)\n{\n}"  # left to stage B


def test_lowercase_macro_list_is_reviewable():
    assert isinstance(LOWERCASE_SIGNATURE_MACROS, tuple)
    for name in ("av_cold", "av_always_inline", "av_noinline", "av_unused", "av_used", "av_const", "av_pure",
                 "av_flatten", "av_alias", "av_warn_unused_result", "av_noreturn", "coroutine_fn",
                 "coroutine_mixed_fn", "always_inline", "APR_INLINE", "APR_THREAD_FUNC"):
        assert name in LOWERCASE_SIGNATURE_MACROS, name


def test_stage_c_only_when_it_differs_from_stage_b():
    # av_cold: A and B change nothing, so only C is offered
    code = "static av_cold void x8_vlc_init(void)\n{\n    int i;\n}"
    v = rescue_variants(code)
    assert [s for s, _ in v] == ["C"]
    assert dict(v)["C"].startswith("static void x8_vlc_init(void)")
    # coroutine_fn plus a dunder in the parameters: A strips the dunder, C the macro too
    code = "static int coroutine_fn f(void *p __user)\n{\n    return 0;\n}"
    v = dict(rescue_variants(code))
    assert sorted(v) == ["A", "C"]
    assert "coroutine_fn" in v["A"] and "__user" not in v["A"]
    assert v["C"].split("{", 1)[0].split() == ["static", "int", "f(void", "*p", ")"]
    # APR_INLINE is ALL-CAPS: stage B already removes it, so C == B and is not yielded
    code = "static APR_INLINE unsigned char *c2x(unsigned what)\n{\n    return 0;\n}"
    assert [s for s, _ in rescue_variants(code)] == ["B"]
    # APR_DECLARE(int): B rewrites the whole head into a macro-shaped stub, C unwraps the type
    code = "APR_DECLARE(int) foo(int x)\n{\n    return x;\n}"
    v = dict(rescue_variants(code))
    assert sorted(v) == ["B", "C"]
    assert v["B"].startswith("void APR_DECLARE_int_foo_int_x(void) ")
    assert v["C"].startswith("int foo(int x)")
    # stage C still applies stage B's transformation (ALL-CAPS tokens in the signature)
    code = "static av_cold int init(AVCodecContext *avctx, int n UNUSED)\n{\n}"
    v = dict(rescue_variants(code))
    assert "UNUSED" not in v["C"] and "av_cold" not in v["C"]


def test_stage_c_variants_of_probe_signatures_parse_shape():
    """The 20 lowercase-macro failures found by the dataset probe (CodeXGLUE defect / D2A)."""
    heads = [
        "static int coroutine_fn blkreplay_co_pwritev(BlockDriverState *bs,\n\n    uint64_t offset, uint64_t bytes, QEMUIOVector *qiov, int flags)\n\n",
        "static av_cold void x8_vlc_init(void)",
        "static av_always_inline int decode_mb_row_no_filter(AVCodecContext *avctx, void *tdata,\n\n int jobnr, int threadnr, int is)\n",
        "static always_inline void fload_invalid_op_excp (int op)\n\n",
        "int attribute_align_arg avcodec_encode_video(AVCodecContext *avctx, uint8_t *buf, int buf_size,\n const AVFrame *pict)\n",
        "static void av_noinline filter_mb_edgev( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, H264Context *h) ",
        "static inline av_flatten int get_symbol_inline(RangeCoder *c, uint8_t *state,\n int is_signed)\n",
        "static av_always_inline void\nyuv2plane1_16_c_template(const int32_t *src, uint16_t *dest, int dstW,\n int big_endian, int output_bits)\n",
        "static av_cold int ass_decode_init(AVCodecContext *avctx)\n",
    ]
    for head in heads:
        v = dict(rescue_variants(head + "{\n    return 0;\n}"))
        assert "C" in v, head
        c_head = v["C"].split("{", 1)[0]
        assert not any(tok in c_head.split() for tok in LOWERCASE_SIGNATURE_MACROS), c_head
        assert c_head.count("(") == c_head.count(")") >= 1


# --- stages A / B must stay what they were -----------------------------------------------

# sha256 over repr((file name, rescue_variants(code))) of the HEAD implementation before stage C
# was added, for every file of tests/fixtures/joern_inputs (A / B entries only).
_AB_FIXTURE_DIGEST = "e2b58fd9f2e6efa3cae5623455618db1af1ca59ce47b53c23bb8fc1282d2373a"


def test_stages_a_and_b_are_unchanged_on_the_existing_fixtures():
    h = hashlib.sha256()
    stages_seen = []
    for p in sorted(FIXTURE_INPUTS.iterdir()):
        code = p.read_text(encoding="utf-8")
        all_v = rescue_variants(code)
        ab = [(s, v) for s, v in all_v if s in ("A", "B")]
        stages_seen.append([s for s, _ in all_v])
        # A / B are exactly strip_signature_macros(aggressive=False / True), in that order, de-duplicated
        expect, seen = [], {code}
        for stage, aggressive in (("A", False), ("B", True)):
            v = strip_signature_macros(code, aggressive=aggressive)
            if v not in seen:
                seen.add(v)
                expect.append((stage, v))
        assert ab == expect, p.name
        h.update(repr((p.name, ab)).encode())
    assert h.hexdigest() == _AB_FIXTURE_DIGEST
    # none of the BigVul fixture functions carries a lowercase signature macro: stage C never fires on them
    assert not any("C" in s for s in stages_seen)
