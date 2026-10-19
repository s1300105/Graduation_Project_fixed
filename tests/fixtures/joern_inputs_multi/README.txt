Whole-program Joern fixture (tests/test_joern_export.py, whole_file graph unit)
===============================================================================

Five small programs written through utils.data.datamanager.to_files with an explicit
`ext` column (numeric stems are the row index), then exported with the SAME two
commands the pipeline runs (utils.process.cpg_generator.joern_parse / joern_create):

  Joern 4.0.400 (/home/user/joern_dl/joern-cli, `joern --version` -> "Version: 4.0.400")

  joern-parse <this dir> --output <work>/multi.bin                      (cwd = <work>)
  joern --script joern/graph-for-funcs.sc --param cpgFile=<work>/multi.bin \
        --param outFile=<work>/multi.json                                (cwd = <work>)
  cp <work>/multi.json tests/fixtures/joern_v4_export_multi.json

Files (what each one pins):
  201.c    C program: helper() defined BEFORE main(), two global arrays/ints with
           initialisers. Joern lists `helper` first, so graph_unit="first_method" keeps
           the helper and silently drops main (the defect whole_file fixes). Its
           `201.c:<global>` holds helper + main + the global LOCALs (`int table[4]`,
           `int counter`) and their initialisers.
  202.c    K&R `main()` with no return type, no #include (POJ-104 style). Written as .c
           the C frontend yields a real method `main<duplicate>0`; written as .cpp it
           would lose main entirely (4-node <global> with one UNKNOWN node).
  203.cpp  Modern C++: #include <iostream>, using namespace std, template helper
           `add:T(T,T)` and `main:int()`.
  204.c    C file with a function-like `#define SQ(x) ((x)*(x))` used in main. The
           macro is exported as an extra method `204.c:SQ:ANY(1)` that shares no node
           with `204.c:<global>`.
  205.cpp  Two-function C++ file with a struct (`dist2:int(Point,Point)`,
           `main<duplicate>0:int()`).

Export facts (Joern 4.0.400): 34 methods = 5 x <global> + 9 real/macro methods +
`<includes>:<global>` + 19 call-target stubs with file `<empty>`. Every real method's
node set is a strict subset of its file's <global> node set.

Regenerating the recording (the sources in this directory are the ground truth, the
JSON is a recording of what Joern 4.0.400 exports for them):

  from utils.process import cpg_generator as cp
  cp.joern_parse(JOERN_CLI, "tests/fixtures/joern_inputs_multi", WORK + "/", "multi", timeout_sec=1500)
  res = cp.joern_create(JOERN_CLI, WORK, WORK, ["multi.bin"], timeout_sec=1500)
  shutil.copy(res["multi.bin"], "tests/fixtures/joern_v4_export_multi.json")

The sources themselves were written with
  dm.to_files(pd.DataFrame({"func": [...5 programs...], "ext": [".c", ".c", ".cpp", ".c", ".cpp"]},
                           index=[201, 202, 203, 204, 205]), "tests/fixtures/joern_inputs_multi")
i.e. with the explicit `ext` column (the whole-program datasets choose the extension
this way: K&R main -> .c, else C++ hints -> .cpp) rather than guess_extension.

joern_v4_export_crafted.json (same Joern 4.0.400 + joern/graph-for-funcs.sc invocation) was produced
during the adversarial review from seven crafted files written through datamanager.to_files:
  12.c    struct Only { int a; int b; };            (declarations only; same numeric stem as 12.cpp)
  12.cpp  int main() { return 0; }
  13.c    #define SQ(x) ((x)*(x))                     (function-like macro only)
  14.cpp  namespace ns { int f(int); }  struct Foo { int c(); int cpp(int); }  static int helper(int);
          int main() { auto cmp = [](int a, int b) { ... }; ... }   (lambda inside main)
  15.c    #define ADD(a,b) ((a)+(b))  int sq(int);  int main()
  16.cpp  main() { ... }                              (K&R main written as .cpp: Joern emits no METHOD)
  17.c    int table[4] = {1, 2, 3, 4};                 (global array only)
It pins: lambda bodies are separate methods that whole_file mode folds back in; a struct member named
`c` / `cpp` is not a file prefix; files defining no function are dropped in whole_file mode.
