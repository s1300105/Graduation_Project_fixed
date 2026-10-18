"""Signature-macro stripping used by the --cpg rescue pass."""
from utils.data.datamanager import strip_signature_macros, rescue_variants


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
    assert [s for s, _ in v] == ["A"]                         # B would be identical to A here
    code2 = "PHP_FUNCTION(foo)\n{\n  return;\n}"
    assert [s for s, _ in rescue_variants(code2)] == ["B"]     # A changes nothing
