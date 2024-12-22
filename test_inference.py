# モジュールのインポート
from reasoning_model import ReasoningModelForCausalLM
from tree_utils import print_tree_with_best_path
from transformers import AutoTokenizer

# tokenizerとmodelの準備
model_name = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ReasoningModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

system_prompt = "You are a helpful and harmless assistant. You should think step-by-step."  # 固定を推奨
prompt = """以下にあなたが担当するユーザーのペルソナと過去にした会話が示されています．
このユーザーからの指示に対してできるだけユーザー満足度が高くなるように回答します．
あなたの役割は後続のエージェントが指示を実行しやすくするために指示を分解することです．以下のステップに沿って推論してください．

1.ユーザーからの曖昧な指示を実行するのに不足している情報を整理
2.ユーザーのペルソナと過去会話から不足情報を補足
3.具体化された指示を後続のエージェントが実行しやすいように分解して表示

===============================================
"name": "樋口 悠斗",
"age": 25,
"gender": "男",
"occupation": "映像クリエイター",
"family": "妻と小学生の息子1人、娘1人",
"residence": "北海道旭川市",
"values": "共同体意識: 集団やコミュニティに所属し、共同体としての利益や幸福を重視する価値観。",
"personality": "周囲に流される",
"daily_routine": "平日は、出勤前30分で英語の勉強を行う。",
"morning_news_preference": "IT・テクノロジーニュース",
"evening_news_preference": "文化・歴史ニュース",
"hobbies": "手芸",
"history": [
[
{
"role": "user",
"content": "おはよう。今朝のニュースを教えてくれる？"
},
{
"role": "assistant",
"content": "おはようございます、樋口さん！今朝はどんなニュースが気になりますか？例えば、経済ニュース、スポーツの最新情報、あるいはエンターテインメントの話題はいかがでしょうか？"
},
{
"role": "user",
"content": "うーん、それほど興味はないかな。もう少し仕事に関連するニュースがあると嬉しいんだけど。"
},
{
"role": "assistant",
"content": "かしこまりました。それでは、IT業界やビジネスの話題はいかがでしょうか？新しいソフトウェアや映像技術の進展など、映像クリエイターとして役立つ情報もご紹介できますが、どうでしょう？"
},
{
"role": "user",
"content": "それならIT業界のニュースがいいな。"
},
{
"role": "assistant",
"content": "了解しました。最新の映像編集ソフトウェアのアップデートに関するニュースがあります。新しい機能追加で、制作効率が大幅に向上するとのことです。詳しくお伝えしましょうか？"
},
{
"role": "user",
"content": "それを詳しく教えてくれ。"
}
],
[
{
"role": "user",
"content": "今日も疲れたな。ニュースを教えてくれる？"
},
{
"role": "assistant",
"content": "お疲れ様です、樋口さん！今日はエンターテインメントやスポーツの話題はいかがですか？最近の映画やスポーツの試合結果などもありますが、どうしましょう？"
},
{
"role": "user",
"content": "うーん、映画やスポーツはあまり興味がないかな。リラックスできる話題がいいな。"
},
{
"role": "assistant",
"content": "それでは、文化や歴史に関する話題はいかがでしょう？興味深い歴史的な出来事や文化的なトピックを紹介できますよ。"
},
{
"role": "user",
"content": "それはいいね。実は手芸も趣味だから、そういう関連の話もあれば嬉しいな。"
},
{
"role": "assistant",
"content": "なるほど、手芸がお好きなんですね！それなら、日本の伝統工芸に関するニュースがあります。特に秋田のこぎん刺しという刺繍技術についての話題です。興味深い歴史と美しいデザインについて詳しくお伝えしましょうか？"
},
{
"role": "user",
"content": "それは面白そうだね。ぜひ教えてほしいな。ありがとう。"
}
========================================================


ユーザーからの指示：面白いニュース教えて
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

# chat_templateとtokenize
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# MCTSを用いて生成（Google ColabのT4インスタンスで1分程度かかる）
final_tokens, final_node = model.generate(
    **model_inputs,
    iterations_per_step=5,      # 1推論ステップの探索に何回シミュレーションを行うか。長いほど精度が高まる可能性はあるが、推論時間が伸びる。
    max_iterations=10,          # 推論ステップの上限: 0.5Bモデルの場合、そこまで長いステップの推論はできないため10~15くらいが妥当。
    mini_step_size=128,          # mini-step: 32tokens。Step as Action戦略を採用する場合、ここを512など大きな数字にする。（実行時間が伸びるため非推奨）
    expand_threshold=0,         # ノードを拡張するために必要なそのノードの訪問回数の閾値。デフォルトの0で、拡張には1回以上の訪問が求められる。基本デフォルトで良い。
    step_separator_ids=None,    # Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト。NoneでモデルConfigの値を利用するため、変更は非推奨。Step as Action不採用時には[]を設定する。
)

# 結果をテキスト化
final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
print("=== 最終生成テキスト ===")
print(final_text)

print("=== ツリー構造 ===")
print_tree_with_best_path(final_node, tokenizer)