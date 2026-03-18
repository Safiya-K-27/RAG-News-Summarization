"""Stage 10: Personalization prompt controls for summary style."""

from utils.schema import UserPreferences


class PersonalizationAgent:
    """Builds style and control instructions from user preferences."""

    def build_prompt_controls(self, preferences: UserPreferences) -> str:
        length_instruction = {
            "short": "Write 3-4 concise sentences.",
            "medium": "Write 1-2 short paragraphs with key facts.",
            "long": "Write 3 concise paragraphs with structured detail.",
        }.get(preferences.length, "Write 1 short paragraph.")

        tone_instruction = {
            "formal": "Use precise, professional language.",
            "casual": "Use plain, conversational language while remaining factual.",
        }.get(preferences.tone, "Use clear language.")

        bias_instruction = {
            "neutral": "Avoid opinions and keep strictly source-grounded claims.",
            "balanced": "Present multiple viewpoints when available.",
        }.get(preferences.bias_control, "Stay factual and balanced.")

        reading_level_instruction = {
            "simple": "Use short sentences and common vocabulary suitable for broad audiences.",
            "medium": "Use clear language with moderate detail and minimal jargon.",
            "advance": "Use richer detail, nuanced phrasing, and domain-appropriate terminology.",
            "advanced": "Use richer detail, nuanced phrasing, and domain-appropriate terminology.",
        }.get(preferences.reading_level, "Use clear language with moderate detail.")

        return "\n".join([length_instruction, tone_instruction, bias_instruction, reading_level_instruction])
