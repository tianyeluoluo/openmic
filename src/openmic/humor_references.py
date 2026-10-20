"""Source-traceable comedy mechanisms, without storing protected routine text.

These are human-written abstractions of publicly described performances and
creators' own writing lessons. Their reception is evidence that the examples
are worth studying, not a numeric 5/5 label for any generated script.
"""

from __future__ import annotations

from typing import Dict, List


REFERENCE_MECHANISMS: List[Dict[str, str]] = [
    {
        "id": "zhou_noise_reframe",
        "source": "腾讯视频官方发布：周奇墨装修噪音片段",
        "url": "https://www.youtube.com/watch?v=35DUOSac4-k",
        "trigger_type": "observed_detail",
        "mechanism": "日常烦恼保持真实，一个重复的细节被讲述者赋予私人含义；"
                     "人物据此行动后，细节的重复反过来暴露其心理冲突。",
        "avoid": "不能复制演员的拟声、台词或房产相关原情节。",
    },
    {
        "id": "zhou_fast_food_expectation",
        "source": "腾讯视频官方发布：周奇墨快餐店片段",
        "url": "https://www.youtube.com/watch?v=1msD5f9z9JQ",
        "trigger_type": "observed_detail",
        "mechanism": "选一个人人看得懂的流程承诺，再让人物的具体动作一步步背离承诺；"
                     "冲突来自赶时间的顾客与慢下来的互动，而非凭空新规定。",
        "avoid": "不能照搬点餐经过、人物模仿或原句。",
    },
    {
        "id": "pang_social_self_consciousness",
        "source": "腾讯视频官方发布：庞博替家人购物片段",
        "url": "https://www.youtube.com/watch?v=0MSMGcXZWKY",
        "trigger_type": "observed_detail",
        "mechanism": "普通任务本身没问题，讲述者过度担心别人怎么看自己；"
                     "为显得自然采取的动作反而使自己更显眼。",
        "avoid": "不能照搬购买的商品、保安情节或原句；旁人反应须符合日常。",
    },
    {
        "id": "apatow_one_public_failure",
        "source": "Judd Apatow 的写作课对首次投球单口故事的拆解；MLB 原始事件记录",
        "url": "https://www.masterclass.com/articles/9-secrets-to-writing-stand-up-comedy-from-judd-apatow",
        "event_url": "https://www.mlb.com/video/apatow-s-first-pitch-c1002247083",
        "trigger_type": "minor_chance",
        "mechanism": "只用一次确实发生的公开失误作为中心结果；事前紧张、动作细节和他人的"
                     "即时反应逐步延迟揭底，后续不再增加离奇巧合。",
        "avoid": "不能照搬投球、比赛、演员本人的经历或原句。",
    },
    {
        "id": "martin_delayed_reinterpretation",
        "source": "Steve Martin 早期笑话，见 MasterClass 写作课的结构分析",
        "url": "https://www.masterclass.com/articles/how-to-write-a-joke-in-7-easy-steps",
        "trigger_type": "observed_detail",
        "mechanism": "前半段让观众按最普通的方式理解一件事；最后给一个具体身体动作，"
                     "使前面的细节突然获得第二种解释。",
        "avoid": "不能照搬猫洗澡的事件、原台词或末句。",
    },
]


def reference_guidance() -> str:
    """Feed only transferable mechanics, not source jokes, into generation."""
    return "\n".join(
        f"{item['id']}（{item['trigger_type']}）：{item['mechanism']}"
        f" 禁止移植：{item['avoid']}"
        for item in REFERENCE_MECHANISMS
    )
