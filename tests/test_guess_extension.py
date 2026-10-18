"""C / C++ detection used to name Joern input files when the jsonl has no lang column."""
import pytest
from utils.data.datamanager import guess_extension, looks_like_cpp

C_PROBES = [
    'void f(void) { printf("a::b %s", x); }',                       # :: inside a string literal
    "int g(int n) { int operators = n + 1; return operators; }",     # identifier containing operator
    "int h(struct s *p) { return p->virtual; }",                     # C++ keyword used as a C field
    "int k(int x) { delete(x); return namespace_count; }",           # C function named delete, identifier namespace_count
    'void m(int flag) { printf("%s", flag\n ? get_name(flag)\n : default_name(flag)); }',  # multi-line ternary in call args
    "int n(int c) { switch (c) { CASE(X): foo(c); break; } return 0; }",          # macro case label followed by a call
    "/* a::b in a comment */ int o(void) { return 1; }",
    "int p(char *s) { return s[0] == ':' && s[1] == ':'; }",
    "int q(int a) { return a < b ? 1 : 2; }",
]
CPP_PROBES = [
    "void CClient::ClearServerDependentCaps() { m_caps.clear(); }",
    "KillRing() : size(0), index(0), lastAction(actionOther) {\n    theRing.reserve(capacity);\n}",
    "template <typename T> T add(T a, T b) { return a + b; }",
    "void A::f() { this->x = 1; }",
    "Foo *make() { return new Foo(1); }",
    "void kill(Foo *p) { delete p; }",
    "void kill2(Foo *p) { delete[] p; }",
    "class Widget : public Base { public: int x; };",
    "namespace util { int x; }",
    "bool operator==(const A &a, const A &b) { return a.x == b.x; }",
    "void r() { auto p = static_cast<Foo *>(q); }",
    "void s() { try { t(); } catch (const std::exception &e) { } }",
    "int *u() { return nullptr; }",
]


@pytest.mark.parametrize("code", C_PROBES)
def test_c_code_stays_c(code):
    assert not looks_like_cpp(code), code
    assert guess_extension(code) == ".c"


@pytest.mark.parametrize("code", CPP_PROBES)
def test_cpp_code_is_detected(code):
    assert looks_like_cpp(code), code
    assert guess_extension(code) == ".cpp"


def test_lang_column_is_only_a_positive_hint():
    assert guess_extension("int f(void) { return 1; }", "CPP") == ".cpp"
    # BigVul marks most C++ functions as "C": the syntax check must still win
    assert guess_extension("void A::f() {}", "C") == ".cpp"
    assert guess_extension("int f(void) { return 1; }", "C") == ".c"
