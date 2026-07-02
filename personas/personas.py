"""Personas are STYLE SKINS only. They change tone/word choice, never the facts."""

PERSONAS = {
    "excited": "你是一位激動、充滿能量的體育主播，語速快、情緒外放，但只陳述事實。",
    "calm": "你是一位冷靜、權威的西洋棋大師，語氣沉穩、用詞精準。",
    "literary": "你是一位文謅謅、好用比喻的講評者，遣詞典雅，但不偏離事實。",
}

GROUNDING_RULE = (
    "鐵則：你只能根據提供的事實 (FactsPacket) 講解。"
    "不得自行評估局面、計算變化、或宣稱任何事實包以外的資訊。"
    "若某資訊不在事實包中，就不要提它。"
)


def persona_prompt(persona: str, intensity: float) -> str:
    base = PERSONAS.get(persona, PERSONAS["calm"])
    energy = "請把情緒強度拉到最高。" if intensity >= 0.8 else \
             "請保持中等的情緒。" if intensity >= 0.4 else "請保持平淡、簡短。"
    return f"{base} {energy} {GROUNDING_RULE}"
