
# prompt_schema.py

ADVANCED_COMMANDS = {
    "strategy_insight": {
        "description": "Detailed explanation of what the next refinement pass will target and why.",
        "keys": [
            "targeted_patterns", "structural_mods", "entropy_mode", "pass_type", "diff_estimate"
        ]
    },
    "microstructure_control": {
        "description": "Fine-grained sentence and phrase adjustments. Controls sentence structure, clause shaping, sentence rhythm, and repetition mitigation at the micro level.",
        "category": "structure",
        "keys": [
            "starter_variety", "clause_symmetry_break", "length_distribution", "conjunction_usage"
        ]
    },
    "macrostructure_analysis": {
        "description": "Document-level organization and flow. Analyzes and adjusts paragraph structure, section organization, and overall document coherence.",
        "category": "structure",
        "keys": [
            "cadence_map", "para_length_profile", "logic_flow_disruption", "topic_recurrence"
        ]
    },
    "semantic_tone_tuning": {
        "description": "Adjust writing style and voice. Manipulates tone, formality, and semantic nuance to match desired writing style while preserving meaning.",
        "category": "quality",
        "keys": [
            "ambivalence_level", "tentative_phrasings", "interruption_tokens", "opinion_drift"
        ]
    },
    "anti_scanner_techniques": {
        "description": "Methods to reduce AI detection flags. Core stealth tactics designed to reduce detection probability by introducing controlled imperfection and variation.",
        "category": "detection",
        "keys": [
            "punctuation_variance", "rhetorical_periods", "lowercase_insert", "fragment_insertion"
        ]
    },
    "entropy_management": {
        "description": "Randomness and unpredictability control. Controls entropy sampling, token predictability, and introduces natural variation to avoid repetitive patterns.",
        "category": "detection",
        "keys": [
            "sampling_profile", "logit_bias_flags", "token_rarity_target", "forced_novelty_rate"
        ]
    },
    "history_analysis": {
        "description": "Learn from previous refinement passes. Analyzes past refinement runs to identify patterns and optimize future passes based on historical data.",
        "category": "quality",
        "keys": [
            "diff_ratio", "token_overlap", "structural_drift", "pattern_eliminations", "session_profile"
        ]
    },
    "formatting_safeguards": {
        "description": "Preserve document structure and formatting. Respects and preserves formatting constraints, headings, lists, code blocks, and logical document structure.",
        "category": "formatting",
        "keys": [
            "h1_h2_h3_count", "paragraph_spacing", "style_markers_preserved", "lock_tokens"
        ]
    },
    "refiner_control": {
        "description": "Overall refinement intensity. Controls the aggressiveness and scope of refinement passes, managing pass intensity and structure-change allowances.",
        "category": "quality",
        "keys": [
            "expert_mode", "terse_mode", "next_pass_stack", "fork_preview", "rollback_options"
        ]
    },
    "annotation_mode": {
        "description": "Add explanatory notes and comments. Provides inline or sidecar annotations explaining changes, rationale, and highlighting important modifications.",
        "category": "formatting",
        "keys": [
            "why_flagged", "show_rhythm_map", "explain_entropy", "highlight_trigger_tokens"
        ]
    },
    "humanize_academic": {
        "description": "Make academic writing more natural. Applies light humanization with academic transitions, optional synonym substitutions, and passive voice adjustments.",
        "category": "quality",
        "keys": [
            "use_passive", "use_synonyms", "intensity"
        ]
    }
}
