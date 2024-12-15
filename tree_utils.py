def get_best_path_node_ids(node):
    """
    終端ノードからrootまで辿り、そのパス上のノードIDをセットで返す。

    Args:
        node (MCTSNode): 終端ノード

    Returns:
        Set[int]: パス上のノードのidを格納したセット
    """
    best_path = []
    n = node
    while n is not None:
        best_path.append(id(n))
        n = n.parent
    return set(best_path)

def print_tree(node, tokenizer, prefix="", is_last=True, highlight_path=None):
    """
    MCTSツリーを表示する。

    Args:
        node (MCTSNode): ツリーのルートまたは現在のノード
        tokenizer (PreTrainedTokenizer): トークン列を文字列に変換するトークナイザ
        prefix (str, optional): 枝線描画用の前置文字列。内部使用。
        is_last (bool, optional): 兄弟中最後の子ノードかどうか。内部使用。
        highlight_path (Set[int], optional): 強調表示するノードIDの集合
    """
    node_id = id(node)
    marker = "* " if (highlight_path and node_id in highlight_path) else ""
    connector = "└── " if is_last else "├── "

    value_avg = (node.value_sum/node.visit_count) if node.visit_count > 0 else 0.0

    if node.action_tokens is None:
        # ルートノード
        action_text = "[ROOT]"
    else:
        # action_tokensをデコードし改行をエスケープ
        raw_text = tokenizer.decode(node.action_tokens, skip_special_tokens=True)
        action_text = raw_text.replace("\n", "\\n")

    print(f"{prefix}{connector}{marker}action={action_text}, visits={node.visit_count}, value_sum={node.value_sum:.2f}, avg={value_avg:.2f}")

    children = node.children
    for i, child in enumerate(children):
        is_last_child = (i == len(children)-1)
        print_tree(child, tokenizer, prefix + ("    " if is_last else "│   "), is_last_child, highlight_path)

def print_tree_with_best_path(final_node, tokenizer):
    """
    終端ノードからルートまでの最良パスをハイライトしてツリーを表示する。

    Args:
        final_node (MCTSNode): MCTS探索で最後に選ばれたノード
        tokenizer (PreTrainedTokenizer): トークン列を文字列に戻すためのトークナイザ
    """
    best_path_node_ids = get_best_path_node_ids(final_node)
    # ルートノードまで戻る
    root_node = final_node
    while root_node.parent is not None:
        root_node = root_node.parent

    # ツリー表示
    print("ツリー構造を表示します（'*'が付いているノードは最終選択パス上）：")
    print_tree(root_node, tokenizer, highlight_path=best_path_node_ids)
