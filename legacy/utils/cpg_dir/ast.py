from .node import Node


import re

_NODE_RE = re.compile(
    r"""^
        nodes\.(?P<type>[A-Za-z0-9_]+)
        \[
          label=(?P<label>[^;\]]+);
          \s*seq=(?P<seq>\d+);
          \s*id=(?P<id>\d+)
        \]
        $
    """,
    re.X,
)

class AST:
    def __init__(self, nodes, indentation):
        self.indentation = indentation

        # --- 1) もし nodes が「文字列のリスト」なら dict リストに変換 ---
        if isinstance(nodes, (list, tuple)) and nodes and isinstance(nodes[0], str):
            parsed = []
            for line in nodes:
                m = _NODE_RE.match(line.strip())
                if not m:
                    # 解析できない行はスキップ or 例外にしたい場合は raise に変更
                    continue
                d = m.groupdict()
                d["seq"] = int(d["seq"])
                # 統一的に使えるキーに整形
                parsed.append({
                    "id": d["id"],
                    "label": d["label"],    # 例: 'CALL', 'IDENTIFIER', ...
                    "type": d["type"],      # 例: 'Call', 'Identifier', ...
                    "seq": d["seq"],
                })
            nodes = parsed

        # --- 2) ここから下はこれまで通り（dict list を前提） ---
        if not isinstance(nodes, (list, tuple)):
            raise TypeError(f"AST expects list[dict], got {type(nodes)}")

        # あなたの Node クラスがこの辞書キーを参照する前提：
        # Node(node_dict, self.indentation) で OK
        self.nodes = {
            str(node["id"]).split(".")[-1]: Node(node, self.indentation)
            for node in nodes
            if isinstance(node, dict) and "id" in node
        }
        
    def __str__(self):
        indentation = self.indentation * "\t"
        nodes_str = ""

        for node in self.nodes:
            nodes_str += f"{indentation}{self.nodes[node]}"

        return f"\n{indentation}Size: {self.size}\n{indentation}Nodes:{nodes_str}"

    def get_nodes_type(self):
        return {n_id: node.type for n_id, node in self.nodes.items()}