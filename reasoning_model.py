import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria

class StepSeparatorStoppingCriteria(StoppingCriteria):
    """
    ステップ区切りトークンIDに達したら生成を停止するStoppingCriteria。
    """
    def __init__(self, step_separator_id):
        """
        コンストラクタ。

        Args:
            step_separator_id (int): ステップ区切りトークンのID。
        """
        self.step_separator_id = step_separator_id

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        停止判定を行う。

        Returns:
            bool: 最終トークンがステップ区切りトークンIDならTrue
        """
        return input_ids[0, -1].item() == self.step_separator_id


class ReasoningModelForCausalLM():
    """
    ReasoningModelForCausalLMはAutoModelForCausalLMでロードしたモデルを内部に持ち、MCTSによる推論を行うためのラッパークラスです。
    MCTSを用いて、逐次的なステップごとのサンプル生成を試み、その中からUCB1などを用いて最良と思われるノードを選択します。

    Transformersライクな使用感を目指し、以下を提供します。
    - from_pretrained でのモデルロード
    - configプロパティでモデル設定へのアクセス
    - toメソッドでデバイスやdtype変更を可能にする
    """

    def __init__(self, model):
        """
        コンストラクタ。直接呼ぶのではなく、from_pretrainedを利用してください。

        Args:
            model (PreTrainedModel): AutoModelForCausalLM.from_pretrainedでロードしたモデル。
        """
        self.model = model
        self.model.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, torch_dtype="auto", device_map="auto", **kwargs):
        """
        TransformersのAutoモデルと同様のインターフェースでモデルをロードするクラスメソッド。

        Args:
            pretrained_model_name_or_path (str | os.PathLike): モデル名（HuggingFace HubのIDなど）もしくはパス
            torch_dtype (str or torch.dtype, optional): モデルロード時のdtype指定。デフォルト"auto"。
            device_map (str or Dict, optional): デバイス配置指定。デフォルト"auto"。
            **kwargs: その他AutoModelForCausalLM.from_pretrainedに渡したい追加の引数。

        Returns:
            ReasoningCausalLM: 初期化済みのReasoningCausalLMインスタンス。
        """
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs
        )
        return cls(model)
    
    @property
    def config(self):
        """
        モデルのconfigを返すプロパティ。
        Transformersモデルと同様にmodel.configで参照可能。

        Returns:
            PretrainedConfig: モデルの設定オブジェクト
        """
        return self.model.config

    @property
    def device(self):
        """
        モデルが配置されているデバイスを返すプロパティ。
        """
        return self.model.device

    def contains_eos_id(self, token_list):
        """
        与えられたトークン列にEOSトークンIDが含まれるか判定する。

        Args:
            token_list (List[int]): トークンID列。

        Returns:
            bool: EOSトークンIDを含む場合True
        """
        return self.model.config.eos_token_id in token_list

    def generate_single_step(self, input_ids, beam_k=5, max_new_tokens=32, step_separator_ids=None):
        """
        1ステップ分のテキスト生成を行う。

        Args:
            input_ids (List[int]): 現在までの入力トークン列。
            beam_k (int, optional): サンプリング時のtop_k値。デフォルト5。
            max_new_tokens (int, optional): 1ステップで生成する最大トークン数。デフォルト32。
            step_separator_ids (List[int], optional): Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト

        Returns:
            Tuple[List[int], float]:
                更新後のトークン列と、そのステップにおける信頼度(平均値)。
        """
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)

        # stopping_criteriaの設定
        stopping_criteria = None
        if step_separator_ids is not None:
            stopping_criteria = StoppingCriteriaList([StepSeparatorStoppingCriteria(step_id) for step_id in step_separator_ids])
        elif hasattr(self.model.config, 'step_separator_ids') and self.model.config.step_separator_ids:
            stopping_criteria = StoppingCriteriaList([StepSeparatorStoppingCriteria(step_id) for step_id in self.model.config.step_separator_ids])

        outputs = self.model.generate(
            input_tensor,
            do_sample=True,
            top_k=beam_k,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_ids = outputs.sequences
        scores = outputs.scores

        start_idx = input_tensor.size(1)
        gen_tokens = generated_ids[0, start_idx:].tolist()

        confidence_scores = []
        for step_idx, token_id in enumerate(gen_tokens):
            logits = scores[step_idx][0]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 5)
            chosen_pos = (topk_indices == token_id).nonzero(as_tuple=True)
            if len(chosen_pos[0]) > 0:
                pos = chosen_pos[0].item()
                chosen_prob = topk_probs[pos]
                top5_sum = topk_probs.sum()
                c_i = chosen_prob / top5_sum
                confidence_scores.append(c_i.item())
            else:
                confidence_scores.append(0.0)

        v = np.mean(confidence_scores) if confidence_scores else 0.0
        new_input_ids = input_ids + gen_tokens
        return new_input_ids, v

    def generate(self, input_ids, iterations_per_step=5, max_iterations=15, mini_step_size=32, expand_threshold=0, step_separator_ids=None, **kwargs):
        """
        MCTSを用いてトークン列を生成する。

        Args:
            input_ids (List[int]): 初期入力トークン列。
            iterations_per_step (int, optional): 各ステップでのMCTS反復回数。デフォルト5。
            max_iterations (int, optional): MCTSステップの最大反復数。デフォルト15。
            mini_step_size (int, optional): 1ステップあたりの最大新規トークン生成数。デフォルト32。
            expand_threshold (int, optional): ノードを拡張するために必要なvisit_countの閾値。デフォルト0。
            step_separator_ids (List[int], optional): Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト。Noneとするとmodel configの値が利用される。[]を渡すとStep as Actionが不採用（必ずmini-stepで区切り）となる。
            **kwargs: attention_maskなど他のmodel_inputsを受けるが未使用。

        Returns:
            Tuple[List[int], MCTSNode]:
                最終的に決定したトークン列と、その終端ノード。
        """
        from mcts_node import MCTSNode
        from mcts import mcts_search_until_eos

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        # バッチサイズチェック
        if input_ids.size(0) != 1:
            raise ValueError("Batch size must be 1.")

        # input_idsはbatchサイズ1を仮定
        input_ids = input_ids[0].tolist()

        root = MCTSNode(input_ids)
        complete_path_tokens, final_node = mcts_search_until_eos(
            root_node=root,
            llm=self,
            iterations_per_step=iterations_per_step,
            max_iterations=max_iterations,
            mini_step_size=mini_step_size, 
            expand_threshold=expand_threshold, 
            step_separator_ids=step_separator_ids
        )

        final_tokens = []
        for tokens in complete_path_tokens:
            final_tokens.extend(tokens)
        return final_tokens, final_node
    
    def to(self, **kwargs):
        """
        モデルを指定したデバイスやdtypeへ移動または変換する。
        torch.nn.Module.to(**kwargs)と同様に呼び出し側でdevice, dtypeを指定できる。

        例:
          model.to(device="cuda")
          model.to(dtype=torch.bfloat16)
          model.to(device="cuda", dtype=torch.bfloat16)

        Args:
            **kwargs: device, dtypeなどtorch.nn.Module.toで受け取れる引数

        Returns:
            self
        """
        self.model.to(**kwargs)
        return self