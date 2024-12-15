# MCTS-Integrated Causal LM Generation

本プロジェクトでは、Hugging Face Transformersの `AutoModelForCausalLM` を拡張し、Monte Carlo Tree Search (MCTS) を用いたステップごとの探索に基づくテキスト生成を実行します。  
**重要:**
 - `ReasoningCausalLM`はトークン列を直接受け取り、MCTSによる探索を行い、トークン列で出力します。  
 - EOSトークンやステップ区切りトークンIDも事前にユーザーが特定してモデルに渡してください。  
 - 最終的な生成出力の文字列化やツリー表示時のトークナイズは外部で行ってください。

## インストール

[Poetry](https://python-poetry.org/) を利用して依存関係を管理しています。  
以下のコマンドでインストールできます。

```bash
poetry install
```

## 使い方
### 準備
```
# 1. GitHubリポジトリからクローン
git clone https://github.com/Hajime-Y/reasoning-model.git
cd reasoning-model

# 2. 必要なライブラリをpipインストール
poetry install
```

### 生成
```
from transformers import AutoTokenizer
from reasoning_model import ReasoningCausalLM
from tree_utils import print_tree, get_best_path_node_ids

model_name = "HachiML/QwQ-CoT-0.5B-JA-v0.4"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 必要なトークンIDを取得
eos_token_id = tokenizer.eos_token_id
step_separator = "\n\n"
step_separator_id = tokenizer.encode(step_separator, add_special_tokens=False)
if len(step_separator_id) != 1:
    raise ValueError("ステップ区切りは1トークンで指定してください")

# モデルを初期化
model = ReasoningCausalLM(
    model_name=model_name,
    step_separator_ids=step_separator_id,
    eos_token_id=eos_token_id
)

system_prompt = "あなたは有能な数学の教師です。"
prompt = "231 + 65*6 = ?"

# ここで外部でトークナイズ
system_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

# 入力を結合 (モデルの仕様に合わせて自由に)
# 例としてsystem + promptをそのまま繋げる
input_ids = system_ids + prompt_ids

final_tokens, final_node = model.generate(
    input_ids=input_ids,
    iterations_per_step=5, 
    max_iterations=15, 
    mini_step_size=32
)

# 出力を文字列へ変換
final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
print("最終生成テキスト:")
print(final_text)

# ツリー構造を表示
best_path_node_ids = get_best_path_node_ids(final_node)
root_node = final_node
while root_node.parent is not None:
    root_node = root_node.parent

# decode_funcを定義してツリー表示
def decode_func(token_list):
    return tokenizer.decode(token_list, skip_special_tokens=True).replace("\n","\\n")

print("探索ツリー:")
print_tree(root_node, decode_func, highlight_path=best_path_node_ids)
```

## 注意事項

 - Qwenモデル使用時は `tokenizer.apply_chat_template` が使用可能ですが、他モデルの場合は適宜ご自身でプロンプト整形してください。
 - パラメータ（`iterations_per_step`, `max_iterations`, `mini_step_size` など）はニーズに合わせて調整してください。