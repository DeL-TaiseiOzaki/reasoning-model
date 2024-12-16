import math

def ucb_score(parent, child, c_param=1.41):
    """
    UCB1スコアを計算する関数。
    MCTSで子ノード選択時に用いる。

    Args:
        parent (MCTSNode): 親ノード
        child (MCTSNode): 子ノード
        c_param (float, optional): 探索パラメータ(Upper Confidence Boundの定数)。デフォルト1.41。

    Returns:
        float: UCB1スコア
    """
    if child.visit_count == 0:
        return float('inf')
    return (child.value_sum / child.visit_count) + c_param * math.sqrt(math.log(parent.visit_count) / child.visit_count)

def select_child(node):
    """
    UCBスコアに基づいて子ノードを1つ選択する。

    Args:
        node (MCTSNode): 現在のノード

    Returns:
        MCTSNode: 選択された子ノード
    """
    best_score = -float('inf')
    best_child = None
    for child in node.children:
        score = ucb_score(node, child)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

def backpropagate(node, value):
    """
    子ノードの評価値を親へと遡って更新する。

    Args:
        node (MCTSNode): 評価を開始するノード
        value (float): 更新する価値(報酬)
    """
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        node = node.parent

def mcts_search(root, llm, iterations=5, mini_step_size=32, expand_threshold=0, step_separator_ids=None):
    """
    MCTS探索をrootノードから指定回数繰り返し、最良と判断される子ノードを返す。

    Args:
        root (MCTSNode): ルートノード
        llm (ReasoningCausalLM): モデル
        iterations (int, optional): MCTSの繰り返し回数。デフォルト5。
        mini_step_size (int, optional): 1ステップでの最大生成トークン数。デフォルト32。
        expand_threshold (int, optional): ノードを拡張するために必要なvisit_countの閾値。デフォルト0。
        step_separator_ids (List[int], optional): Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト

    Returns:
        MCTSNode: 最良の子ノード
    """
    for _ in range(iterations):
        # Selection
        node = root
        while not node.is_leaf():
            node = select_child(node)

        # Expansion: visit_countがexpand_thresholdを超えていたら拡張する
        if node.visit_count > expand_threshold:
            node.expand(llm, beam_size=2, mini_step_size=mini_step_size, step_separator_ids=step_separator_ids)
        else:
            # 閾値未満なら拡張せず、そのままバックプロパゲーション
            backpropagate(node, node.reward_score)
            continue

        # Backpropagation
        if len(node.children) == 0:
            # 拡張できなかった場合
            backpropagate(node, 0.0)
        else:
            # reward_scoreを用いてbest_childを選ぶ
            best_child = max(node.children, key=lambda c: c.reward_score)
            backpropagate(best_child, best_child.reward_score)

    best_child = max(root.children, key=lambda c: c.value_sum/c.visit_count if c.visit_count>0 else -float('inf'))
    return best_child

def mcts_search_until_eos(root_node, llm, iterations_per_step=5, max_iterations=20, mini_step_size=32, expand_threshold=0, step_separator_ids=None):
    """
    EOSが生成されるまでMCTSによる探索を繰り返す。

    Args:
        root_node (MCTSNode): ルートノード
        llm (ReasoningCausalLM): モデル
        iterations_per_step (int, optional): 各ステップでのMCTS探索反復回数。デフォルト5。
        max_iterations (int, optional): 最大ステップ数。デフォルト20。
        mini_step_size (int, optional): 1ステップでの最大生成トークン数。デフォルト32。
        expand_threshold (int, optional): ノードを拡張するために必要なvisit_countの閾値。デフォルト0。
        step_separator_ids (List[int], optional): Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト。Noneとするとmodel configの値が利用される。[]を渡すとStep as Actionが不採用（必ずmini-stepで区切り）となる。

    Returns:
        Tuple[List[List[int]], MCTSNode]:
            完成したトークン列の配列(各ステップごとの差分トークン)と、最終ノード。
    """
    current_node = root_node
    complete_path_tokens = []

    for _ in range(max_iterations):
        best_node = mcts_search(current_node, llm, iterations=iterations_per_step, mini_step_size=mini_step_size, expand_threshold=expand_threshold, step_separator_ids=step_separator_ids)
        if best_node.action_tokens is not None:
            complete_path_tokens.append(best_node.action_tokens)
        current_node = best_node
        if best_node.is_terminal(llm):
            break

    return complete_path_tokens, current_node
