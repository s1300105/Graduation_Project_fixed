"""C / C++ detection used to name Joern input files when the jsonl has no lang column,
and the explicit `ext` column that whole-program datasets use instead."""
import numpy as np
import pandas as pd
import pytest
from utils.data.datamanager import guess_extension, looks_like_cpp, to_files

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


# --- to_files: explicit `ext` column vs the guessed extension ----------------------------

def _names(d):
    return sorted(p.name for p in d.iterdir())


def test_to_files_without_ext_column_guesses_from_lang_and_syntax(tmp_path):
    df = pd.DataFrame({"func": ["int f(void) { return 1; }", "void A::f() {}", "int g(void) { return 2; }"],
                       "lang": ["C", "C", "CPP"]}, index=[5, 6, 7])
    to_files(df, str(tmp_path))
    assert _names(tmp_path) == ["5.c", "6.cpp", "7.cpp"]
    assert (tmp_path / "6.cpp").read_text(encoding="utf-8") == "void A::f() {}"
    d2 = tmp_path / "nolang"
    to_files(df.drop(columns=["lang"]), str(d2))
    assert _names(d2) == ["5.c", "6.cpp", "7.c"]                 # no lang: syntax hints only


def test_to_files_uses_the_ext_column_when_it_is_a_string(tmp_path):
    # a K&R C program that looks_like_cpp would never flag, forced to .c; modern C++ without any
    # of the C++ hint constructs (cin/cout only) forced to .cpp; a C++-looking function forced to .c
    df = pd.DataFrame({"func": ["main()\n{\n}", "#include <iostream>\nusing namespace std;\nint main() { cin >> x; }",
                                "void A::f() {}"],
                       "ext": [".c", ".cpp", ".c"], "lang": ["CPP", "C", "C"]}, index=[1, 2, 3])
    to_files(df, str(tmp_path))
    assert _names(tmp_path) == ["1.c", "2.cpp", "3.c"]           # ext wins over lang and over the syntax check


def test_to_files_falls_back_per_row_when_ext_is_null(tmp_path):
    df = pd.DataFrame({"func": ["void A::f() {}", "int f(void) { return 1; }", "int g(void) { return 1; }"],
                       "ext": [None, np.nan, ".cpp"]}, index=[1, 2, 3])
    to_files(df, str(tmp_path))
    assert _names(tmp_path) == ["1.cpp", "2.c", "3.cpp"]
    df2 = pd.DataFrame({"func": ["int f(void) { return 1; }"], "ext": pd.Series([pd.NA], dtype="string")}, index=[9])
    to_files(df2, str(tmp_path / "na"))
    assert _names(tmp_path / "na") == ["9.c"]


def test_to_files_rejects_unknown_extensions(tmp_path):
    df = pd.DataFrame({"func": ["int f(void) { return 1; }"], "ext": [".cc"]}, index=[1])
    with pytest.raises(ValueError):
        to_files(df, str(tmp_path))
    df = pd.DataFrame({"func": ["int f(void) { return 1; }"], "ext": ["c"]}, index=[1])
    with pytest.raises(ValueError):
        to_files(df, str(tmp_path))
