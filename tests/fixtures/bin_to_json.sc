// simple_graph_for_funcs.sc
// Joern v4: 関数ごとに AST ノード＋ASTエッジをシンプルな JSON で出力

import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes
import io.shiftleft.semanticcpg.language.locationCreator
import scala.jdk.CollectionConverters._

@main def simple_graph_for_funcs(cpgFile: String, outFile: String): Unit = {

  // ------------------------------ utils ------------------------------------
  def esc(s0: String): String = {
    val s: String = if (s0 == null) "" else s0
    val sb = new StringBuilder(s.length + 16)
    s.foreach {
      case '"'  => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\b' => sb.append("\\b")
      case '\f' => sb.append("\\f")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case c if c < ' ' =>
        sb.append(f"\\u${c.toInt}%04x")
      case c =>
        sb.append(c)
    }
    sb.toString
  }

  def js(s: String): String =
    "\"" + esc(if (s == null) "" else s) + "\""

  def arr(xs: List[String]): String =
    xs.mkString("[", ",", "]")

  def writeText(path: String, text: String): Unit =
    Files.write(Paths.get(path), text.getBytes(StandardCharsets.UTF_8))

  // ------------------------------ load CPG ----------------------------------
  val cpg = importCpg(cpgFile).getOrElse {
    throw new RuntimeException(s"Failed to load CPG: $cpgFile")
  }

  // ------------------------- node JSON（プロパティ全部） --------------------
    def nodeJson(n: nodes.StoredNode): String = {
        val id    = n.id().toString
        val label = n.label
        val tpe   = n.getClass.getSimpleName.replace("$", "")

        // 修正ポイント：asScala いらない
        val propMap = n.properties  // Map[String, Any]

        val propsStr = propMap.map { case (key, value) =>
            val vStr = Option(value).map(_.toString).getOrElse("")
            s"${js(key)}:${js(vStr)}"
        }.mkString("{", ",", "}")

        s"""{
        "id": ${js(id)},
        "label": ${js(label)},
        "type": ${js(tpe)},
        "properties": $propsStr
    }"""
    }


  // ------------------------- AST edges -------------------------------------
  def astEdgesFor(n: nodes.StoredNode): List[String] = n match {
    case a: nodes.AstNode =>
      a.astChildren.l.map { ch =>
        val src = n.id().toString
        val dst = ch.id().toString
        s"""{"out":${js(src)},"in":${js(dst)},"type":"AST"}"""
      }
    case _ => Nil
  }

  // ------------------------- per function JSON ------------------------------
  val funcsJson: String =
    cpg.method.l.map { method =>
      val methodName = method.fullName
      val methodId   = method.id().toString
      val methodFile =
        Option(method.location).flatMap(l => Option(l.filename)).getOrElse("")

      // この関数の AST ノード（ルート除く）
      val astNodes = method.astMinusRoot.l

      // ノード JSON：StoredNode 全部
      val nodesJson = astNodes.collect {
        case n: nodes.StoredNode => nodeJson(n)
      }

      // エッジ JSON：AST エッジだけ
      val edgesJson = astNodes.flatMap(astEdgesFor).distinct

      s"""{
  "function": ${js(methodName)},
  "file": ${js(methodFile)},
  "id": ${js(methodId)},
  "nodes": ${arr(nodesJson)},
  "edges": ${arr(edgesJson)}
}"""
    }.mkString("[", ",", "]")

  val root = s"""{"functions":$funcsJson}"""
  writeText(outFile, root)
}
