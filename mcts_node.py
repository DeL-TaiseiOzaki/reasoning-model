class MCTSNode:
    """
    MCTSで使用するノードクラス。
    各ノードは、これまで生成したトークン列、親ノード、子ノード、および訪問回数や評価値を保持します。
    """

    def __init__(self, input_ids, parent=None, action_tokens=None):
        """
        コンストラクタ。

        Args:
            input_ids (List[int]): このノードに対応する状態(これまでの入力)のトークン列。
            parent (MCTSNode, optional): 親ノード。ルートの場合はNone。
            action_tokens (List[int], optional): 親ノードからこのノードに至る際に生成された差分トークン列。ルートの場合None。
        """
        self.input_ids = input_ids
        self.parent = parent
        self.children = []
        self.action_tokens = action_tokens

        self.visit_count = 0
        self.value_sum = 0.0

    def is_leaf(self):
        """
        このノードが葉ノードか判定する。

        Returns:
            bool: 子ノードが無い場合True
        """
        return len(self.children) == 0

    def is_terminal(self, llm):
        """
        このノードが終端状態かを判定する。
        action_tokens中にEOSトークンIDが含まれる場合、終端とみなす。

        Args:
            llm (ReasoningCausalLM): ロジックモデル

        Returns:
            bool: 終端状態ならTrue
        """
        if self.action_tokens is None:
            return False
        return llm.contains_eos_id(self.action_tokens)

    def expand(self, llm, beam_size=2, mini_step_size=32):
        """
        ノードを展開し、子ノードを生成する。

        Args:
            llm (ReasoningCausalLM): モデルインスタンス。
            beam_size (int, optional): 拡張時に生成する子ノード数。デフォルト2。
            mini_step_size (int, optional): 1ステップでの最大生成トークン数。デフォルト32。
        """
        for _ in range(beam_size):
            new_ids, value = llm.generate_single_step(self.input_ids, beam_k=5, max_new_tokens=mini_step_size)
            diff_tokens = new_ids[len(self.input_ids):]
            child_node = MCTSNode(new_ids, parent=self, action_tokens=diff_tokens)
            child_node.visit_count = 1
            child_node.value_sum = value
            self.children.append(child_node)
