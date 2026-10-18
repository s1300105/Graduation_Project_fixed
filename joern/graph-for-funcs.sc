// joern/graph-for-funcs.sc
//
// Export one JSON object per method with its AST nodes and the typed edges
// (AST, CFG, REACHING_DEF, CDG) among them:
//
//   {"functions":[{"function": <fullName>, "file": <filename>, "id": <id>,
//                  "nodes":[{"id","label","type","properties":{"CODE","LINE_NUMBER","COLUMN_NUMBER"}}],
//                  "edges":[{"out": <src id>, "in": <dst id>, "type": "AST"|"CFG"|"REACHING_DEF"|"CDG"}]}]}
//
// This is a reconstruction: the script used for the thesis experiments was
// never committed (audit M39). The Python parser (utils/functions/cpg.py)
// reads the relation from each edge's own "type" field and accepts both this
// flat layout and the thesis-era per-kind layout. `<global>` pseudo-methods
// are exported too and dropped on the Python side.
//
// Usage: joern --script graph-for-funcs.sc --param cpgFile=/abs/x.bin --param outFile=/abs/x.json
// Written against Joern 4.x (Scala 3). Reverse-direction accessors (_astOut,
// _cfgOut, _reachingDefOut, _cdgOut) exist in Joern 2.x and 4.x.

import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes

@main def main(cpgFile: String, outFile: String): Unit = {

  def esc(s0: String): String = {
    val s = if (s0 == null) "" else s0
    val sb = new StringBuilder(s.length + 16)
    s.foreach {
      case '"'  => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\b' => sb.append("\\b")
      case '\f' => sb.append("\\f")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case c if c < ' ' => sb.append(f"\\u${c.toInt}%04x")
      case c => sb.append(c)
    }
    sb.toString
  }
  def js(s: String): String = "\"" + esc(s) + "\""

  val cpg = importCpg(cpgFile).getOrElse(throw new RuntimeException(s"Failed to load CPG: $cpgFile"))

  def lineOf(n: nodes.StoredNode): String = n match {
    case a: nodes.AstNode => a.lineNumber.map(_.toString).getOrElse("null")
    case _ => "null"
  }
  def colOf(n: nodes.StoredNode): String = n match {
    case a: nodes.AstNode => a.columnNumber.map(_.toString).getOrElse("null")
    case _ => "null"
  }
  def codeOf(n: nodes.StoredNode): String = n match {
    case a: nodes.AstNode => a.code
    case _ => ""
  }
  def nodeJson(n: nodes.StoredNode): String =
    s"""{"id":${js(n.id().toString)},"label":${js(n.label)},"type":${js(n.getClass.getSimpleName.replace("$", ""))},""" +
      s""""properties":{"CODE":${js(codeOf(n))},"LINE_NUMBER":${lineOf(n)},"COLUMN_NUMBER":${colOf(n)}}}"""

  def edgeJson(src: nodes.StoredNode, dst: nodes.StoredNode, t: String): String =
    s"""{"out":${js(src.id().toString)},"in":${js(dst.id().toString)},"type":${js(t)}}"""

  val sb = new StringBuilder
  sb.append("{\"functions\":[")
  var firstFn = true
  cpg.method.l.foreach { method =>
    val astNodes: List[nodes.StoredNode] = method.ast.l.map(_.asInstanceOf[nodes.StoredNode])
    val ids = astNodes.map(_.id()).toSet
    val edges = scala.collection.mutable.LinkedHashSet.empty[String]
    astNodes.foreach { n =>
      n._astOut.foreach { d => if (ids.contains(d.id())) edges += edgeJson(n, d, "AST") }
      n._cfgOut.foreach { d => if (ids.contains(d.id())) edges += edgeJson(n, d, "CFG") }
      n._reachingDefOut.foreach { d => if (ids.contains(d.id())) edges += edgeJson(n, d, "REACHING_DEF") }
      n._cdgOut.foreach { d => if (ids.contains(d.id())) edges += edgeJson(n, d, "CDG") }
    }
    val file = method.filename
    if (!firstFn) sb.append(",")
    firstFn = false
    sb.append("{\"function\":").append(js(method.fullName))
      .append(",\"file\":").append(js(file))
      .append(",\"id\":").append(js(method.id().toString))
      .append(",\"nodes\":[").append(astNodes.map(nodeJson).mkString(",")).append("]")
      .append(",\"edges\":[").append(edges.mkString(",")).append("]}")
  }
  sb.append("]}")
  Files.write(Paths.get(outFile), sb.toString.getBytes(StandardCharsets.UTF_8))
  println(s"wrote $outFile (${cpg.method.size} methods)")
}
