from typing import List, Dict
from app.core.prompt_schema import ADVANCED_COMMANDS
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

class ConversationalRefiner:
    def __init__(self, api_key, conversation_history: List[Dict[str, str]] = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        # Initialize messages with conversation history if provided
        self.messages = conversation_history.copy() if conversation_history else []
        self.score = None
        self.conversation_context = {
            "current_file": None,
            "current_pass": None,
            "recent_changes": [],
            "user_preferences": {},
            "session_goals": []
        }
        # Reuse a small thread pool to enforce timeouts around blocking API calls
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _score_hint(self):
        if self.score is None:
            return ""
        return f"(Current detection score: {self.score:.1f}%)"

    def is_schema_request(self, message: str) -> bool:
        m = message.strip().lower()
        
        # First, check for simple greetings or random text - these should NOT trigger schema
        simple_greetings = {"hi", "hello", "hey", "hi there", "hello there", "asasas", "test", "test test"}
        if m in simple_greetings or len(m) < 3:
            return False
        
        # Exact matches for schema commands
        if m in {"/schema", "schema", "/help", "help", "show schema", "list schema", "show schemas"}:
            return True
        
        # Only match explicit schema-related queries
        schema_terms = [
            "schema details", "advanced commands", "capabilities",
            "toggles", "flags", "what can you do", "show commands", "list commands",
            "what schemas", "which schemas", "available schemas", "schema options",
            "who are you", "what are you"
        ]
        
        # Check if message contains schema terms
        return any(term in m for term in schema_terms)

    def matches_strategy_request(self, message: str) -> bool:
        triggers = [
            "strategy", "strat", "gameplan", "plan", "playbook",
            "approach", "next move", "your move", "what are you doing",
            "how will you proceed", "what's your move", "what's the strat",
            "what comes next", "refinement direction", "what tactic"
        ]
        lowered = message.lower()
        return any(t in lowered for t in triggers)

    def extract_refiner_flags(self, message: str) -> dict:
        flags = {}
        msg = message.lower()
        for key in ADVANCED_COMMANDS:
            if key in msg or f"#{key}" in msg:
                flags[key] = True
        return flags

    def get_advanced_strategy_insight(self) -> str:
        lines = ["🎯 Turbo Alan Strategy Modes:\n"]
        for k, v in ADVANCED_COMMANDS.items():
            desc = v.get("description") if isinstance(v, dict) else str(v)
            friendly_name = k.replace('_', ' ').title()
            lines.append(f"• {friendly_name} — {desc}")
        lines.append("\nType /schema to view these again, or ask about any control.")
        return "\n".join(lines)

    # --- Schema descriptions (rule-based, fast path) ---
    def _level_label(self, level: int | None) -> str:
        if level is None:
            return "(level: n/a)"
        return {
            0: "(off)",
            1: "(light)",
            2: "(moderate)",
            3: "(aggressive)",
        }.get(int(level), "(custom)")

    def _normalize_schema_name(self, name: str) -> str | None:
        """Convert human-readable schema names to schema IDs."""
        name_lower = name.strip().lower().replace(' ', '_').replace('-', '_')
        
        # Direct match
        if name_lower in ADVANCED_COMMANDS:
            return name_lower
        
        # Mapping of common variations
        schema_map = {
            'microstructure': 'microstructure_control',
            'microstructure control': 'microstructure_control',
            'micro structure': 'microstructure_control',
            'macrostructure': 'macrostructure_analysis',
            'macrostructure analysis': 'macrostructure_analysis',
            'macro structure': 'macrostructure_analysis',
            'anti scanner': 'anti_scanner_techniques',
            'anti-scanner': 'anti_scanner_techniques',
            'scanner techniques': 'anti_scanner_techniques',
            'detection reduction': 'anti_scanner_techniques',
            'entropy': 'entropy_management',
            'entropy management': 'entropy_management',
            'randomness': 'entropy_management',
            'tone tuning': 'semantic_tone_tuning',
            'semantic tone': 'semantic_tone_tuning',
            'writing style': 'semantic_tone_tuning',
            'voice': 'semantic_tone_tuning',
            'formatting': 'formatting_safeguards',
            'formatting safeguards': 'formatting_safeguards',
            'preserve formatting': 'formatting_safeguards',
            'refiner': 'refiner_control',
            'refiner control': 'refiner_control',
            'refinement intensity': 'refiner_control',
            'history': 'history_analysis',
            'history analysis': 'history_analysis',
            'learn from past': 'history_analysis',
            'annotation': 'annotation_mode',
            'annotation mode': 'annotation_mode',
            'explanatory notes': 'annotation_mode',
            'humanize': 'humanize_academic',
            'humanize academic': 'humanize_academic',
            'academic humanization': 'humanize_academic',
        }
        
        # Check variations
        for key, schema_id in schema_map.items():
            if key in name_lower or name_lower in key:
                return schema_id
        
        # Partial match on schema IDs
        for schema_id in ADVANCED_COMMANDS.keys():
            if name_lower in schema_id or schema_id.replace('_', ' ') in name_lower:
                return schema_id
        
        return None

    def describe_schema(self, schema_id: str, level: int | None = None) -> str:
        # Try to normalize the schema name first
        normalized_id = self._normalize_schema_name(schema_id)
        if normalized_id:
            schema_id = normalized_id
        
        sid = str(schema_id or '').strip().lower()
        entry = ADVANCED_COMMANDS.get(sid)
        if not entry:
            return f"Unknown schema: {schema_id}. Available schemas: {', '.join(ADVANCED_COMMANDS.keys())}"
        
        desc = entry.get('description') if isinstance(entry, dict) else str(entry)
        category = entry.get('category', 'general') if isinstance(entry, dict) else 'general'
        level_note = self._level_label(level)
        
        # User-friendly schema name
        friendly_name = sid.replace('_', ' ').title()
        
        # Detailed explanations for each schema
        detailed_info = {
            'microstructure_control': {
                'what': 'Fine-grained sentence and phrase adjustments',
                'how': 'Adjusts sentence structure, clause patterns, sentence length variation, and removes repetitive phrasing at the sentence level.',
                'when': 'Use for improving sentence-level flow, reducing repetition, and creating more natural sentence variety.',
                'levels': {
                    0: 'Off - No microstructure adjustments',
                    1: 'Low - Minimal sentence-level changes',
                    2: 'Medium - Moderate sentence restructuring and variety',
                    3: 'High - Aggressive sentence-level optimization'
                }
            },
            'macrostructure_analysis': {
                'what': 'Document-level organization and flow',
                'how': 'Analyzes paragraph structure, section organization, introduction/conclusion quality, and overall document coherence.',
                'when': 'Use for improving document structure, ensuring logical flow, and organizing content at the paragraph/section level.',
                'levels': {
                    0: 'Off - No macro-level analysis',
                    1: 'Low - Basic paragraph and section checks',
                    2: 'Medium - Moderate structural improvements',
                    3: 'High - Comprehensive document restructuring'
                }
            },
            'anti_scanner_techniques': {
                'what': 'Methods to reduce AI detection flags',
                'how': 'Introduces controlled imperfection, punctuation variation, rare word substitutions, and removes AI-like patterns.',
                'when': 'Use when you need to reduce the likelihood of AI detection while maintaining content quality.',
                'levels': {
                    0: 'Off - No anti-detection measures',
                    1: 'Low - Minimal variation and imperfection',
                    2: 'Medium - Moderate anti-detection techniques',
                    3: 'High - Aggressive anti-detection measures'
                }
            },
            'entropy_management': {
                'what': 'Randomness and unpredictability control',
                'how': 'Controls token predictability, introduces natural variation, and avoids repetitive n-grams and patterns.',
                'when': 'Use to make text less predictable and more natural-sounding by managing randomness and variation.',
                'levels': {
                    0: 'Off - No entropy management',
                    1: 'Low - Minimal randomness control',
                    2: 'Medium - Moderate unpredictability',
                    3: 'High - Maximum variation and randomness'
                }
            },
            'semantic_tone_tuning': {
                'what': 'Adjust writing style and voice',
                'how': 'Modifies tone, formality level, and semantic nuance while preserving the core meaning and domain-specific terms.',
                'when': 'Use to adjust the tone (formal/casual), voice, or style of your writing to match your target audience.',
                'levels': {
                    0: 'Off - No tone adjustments',
                    1: 'Low - Subtle tone modifications',
                    2: 'Medium - Moderate style adjustments',
                    3: 'High - Significant tone and voice changes'
                }
            },
            'formatting_safeguards': {
                'what': 'Preserve document structure and formatting',
                'how': 'Protects headings, lists, code blocks, paragraph spacing, and other formatting elements during refinement.',
                'when': 'Essential when working with formatted documents (Word, Markdown) where structure must be preserved.',
                'levels': {
                    0: 'Off - No formatting protection',
                    1: 'Low - Basic formatting preservation',
                    2: 'Medium - Moderate formatting safeguards',
                    3: 'High - Strict formatting preservation'
                }
            },
            'refiner_control': {
                'what': 'Overall refinement intensity',
                'how': 'Controls the aggressiveness and scope of refinement passes, managing how much the text can be changed.',
                'when': 'Use to set the overall intensity of refinement - lower for conservative changes, higher for more significant improvements.',
                'levels': {
                    0: 'Off - Minimal refinement',
                    1: 'Low - Conservative refinement',
                    2: 'Medium - Balanced refinement',
                    3: 'High - Aggressive refinement'
                }
            },
            'history_analysis': {
                'what': 'Learn from previous refinement passes',
                'how': 'Analyzes past refinement runs to identify patterns, optimize future passes, and learn from historical data.',
                'when': 'Use when doing multiple refinement passes to improve consistency and learn from previous iterations.',
                'levels': {
                    0: 'Off - No history analysis',
                    1: 'Low - Basic pattern recognition',
                    2: 'Medium - Moderate learning from history',
                    3: 'High - Comprehensive historical analysis'
                }
            },
            'annotation_mode': {
                'what': 'Add explanatory notes and comments',
                'how': 'Provides inline or sidecar annotations explaining changes, rationale, and highlighting important modifications.',
                'when': 'Use when you want to understand what changes were made and why, especially for educational or review purposes.',
                'levels': {
                    0: 'Off - No annotations',
                    1: 'Low - Minimal explanatory notes',
                    2: 'Medium - Moderate annotation detail',
                    3: 'High - Comprehensive change explanations'
                }
            },
            'humanize_academic': {
                'what': 'Make academic writing more natural',
                'how': 'Applies light humanization with academic transitions, optional synonym substitutions, and passive voice adjustments.',
                'when': 'Use for academic papers and research documents to make them sound more natural while maintaining academic rigor.',
                'levels': {
                    0: 'Off - No academic humanization',
                    1: 'Low - Subtle naturalization',
                    2: 'Medium - Moderate humanization',
                    3: 'High - Significant naturalization'
                }
            }
        }
        
        info = detailed_info.get(sid, {})
        level_info = info.get('levels', {}).get(level or 0, '')
        
        response_parts = [
            f"{friendly_name} {level_note if level is not None else ''}",
            f"\n{desc}",
        ]
        
        if info:
            response_parts.append(f"\n\nWhat it does: {info.get('what', '')}")
            response_parts.append(f"\nHow it works: {info.get('how', '')}")
            response_parts.append(f"\nWhen to use: {info.get('when', '')}")
            if level_info:
                response_parts.append(f"\nCurrent level ({level or 0}): {level_info}")
        
        response_parts.append(f"\n\nCategory: {category.title()}")
        
        return ''.join(response_parts)

    def describe_all_schemas(self, schema_levels: dict | None) -> str:
        levels = {str(k): int(v) for k, v in (schema_levels or {}).items() if isinstance(v, (int, float))}
        lines = ["📊 Current Schema Overview:\n"]
        for sid in ADVANCED_COMMANDS.keys():
            lines.append(self.describe_schema(sid, levels.get(sid)))
        return "\n".join(lines)

    def summarize_active_strategy(self, flags: dict) -> str:
        if not flags:
            return "No schema flags are currently active. Enable some toggles or pass flags to activate strategy modes."

        lines = ["🧠 Turbo Alan Active Strategy:\n"]
        for key in flags:
            if key in ADVANCED_COMMANDS:
                desc = ADVANCED_COMMANDS[key]["description"]
                friendly_name = key.replace('_', ' ').title()
                lines.append(f"• {friendly_name} — {desc}")
        return "\n".join(lines)

    def _safe_chat_completion(self, messages, model: str = "gpt-4", temperature: float = 0.7, timeout_seconds: int = 30) -> str:
        """Run OpenAI chat completion with a hard timeout and safe fallback."""
        def _call():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

        try:
            response = self._executor.submit(_call).result(timeout=timeout_seconds)
            return response.choices[0].message.content
        except FuturesTimeoutError:
            return "The chat request timed out while contacting the model. Please try again."
        except Exception:
            return "The chat request failed while contacting the model. Please try again later."

    def _is_direct_schema_query(self, message: str) -> str | None:
        """Check if the message is a direct schema name query (e.g., just 'microstructure control')."""
        mlow = message.strip().lower()
        
        # Check if message is exactly or closely matches a schema name
        for sid in ADVANCED_COMMANDS.keys():
            # Exact match with schema ID
            if mlow == sid or mlow == sid.replace('_', ' '):
                return sid
            
            # Exact match with friendly name
            friendly_name = sid.replace('_', ' ').title().lower()
            if mlow == friendly_name:
                return sid
            
            # Check if message is just the schema name (allowing for minor variations)
            # Remove common words and check if it's mostly the schema name
            cleaned = mlow.replace('what is', '').replace('tell me about', '').replace('explain', '').strip()
            if cleaned == sid or cleaned == sid.replace('_', ' ') or cleaned == friendly_name:
                return sid
        
        # Try normalized matching
        normalized = self._normalize_schema_name(mlow)
        if normalized:
            # Check if the normalized result is close to the original message
            normalized_friendly = normalized.replace('_', ' ').title().lower()
            if mlow == normalized_friendly or mlow == normalized.replace('_', ' '):
                return normalized
        
        return None

    def chat(self, message, flags=None):
        # Extract context from the message
        self.extract_context_from_message(message)
        
        # Schema fast-paths (no model call)
        mlow = (message or '').strip().lower()
        if self.is_schema_request(mlow):
            return self.get_advanced_strategy_insight()
        
        # Check if message is a direct schema name query (e.g., just "microstructure control")
        direct_schema = self._is_direct_schema_query(message)
        if direct_schema:
            return self.describe_schema(direct_schema, getattr(self, 'schema_levels', {}).get(direct_schema))
        
        # Explain single schema: "explain <schema>", "what is <schema>", "tell me about <schema>", etc.
        schema_query_patterns = ['explain', 'what is', 'tell me about', 'describe', 'how does', 'what does', 'information about']
        if any(pattern in mlow for pattern in schema_query_patterns):
            # Try to match any schema name in the message
            for sid in ADVANCED_COMMANDS.keys():
                # Check for exact schema ID
                if sid in mlow or sid.replace('_', ' ') in mlow:
                    return self.describe_schema(sid, getattr(self, 'schema_levels', {}).get(sid))
                # Check for friendly name variations
                friendly_name = sid.replace('_', ' ').title().lower()
                if friendly_name in mlow:
                    return self.describe_schema(sid, getattr(self, 'schema_levels', {}).get(sid))
            
            # Try normalized schema name matching
            normalized = self._normalize_schema_name(mlow)
            if normalized:
                return self.describe_schema(normalized, getattr(self, 'schema_levels', {}).get(normalized))
            
            # If asking about "all" schemas
            if 'all' in mlow or 'current schema' in mlow or 'schemas' in mlow or 'every schema' in mlow:
                return self.describe_all_schemas(getattr(self, 'schema_levels', {}))

        if flags is None:
            flags = self.extract_refiner_flags(message)
        if hasattr(self, 'last_flags'):
            flags.update(self.last_flags)
        if self.matches_strategy_request(message):
            return self.summarize_active_strategy(flags)

        # Build context-aware prompt
        context_summary = self.get_context_summary()
        context_prompt = f"Context: {context_summary}\n\n" if context_summary != "No specific context available." else ""
        
        prompt = f"{context_prompt}{message}"
        if self.score:
            prompt += "\n" + self._score_hint()

        # Add system message if this is the start of a conversation
        if not self.messages or not any(m.get("role") == "system" for m in self.messages):
            system_message = (
                "You are Turbo Alan Refiner, an AI assistant specialized in text refinement and AI detection reduction. "
                "You help users refine their text to make it more human-like while maintaining meaning and quality. "
                "You can answer questions about refinement strategies, schema controls, and provide guidance on improving text. "
                "IMPORTANT: Do not rewrite anything unless the user explicitly asks you to rewrite, revise, or propose edits. "
                "If this is feedback or a question, only respond with insight or advice. "
                "Consider the conversation context when providing responses. "
                "If the user is referring to a specific file or pass, acknowledge it in your response. "
                "Maintain a helpful, conversational tone and remember previous messages in the conversation."
            )
            self.messages.insert(0, {"role": "system", "content": system_message})
        
        # Add user message with context
        self.messages.append({"role": "user", "content": prompt})

        reply = self._safe_chat_completion(self.messages, model="gpt-4", temperature=0.7, timeout_seconds=30)

        # Optional post-processing: humanize academic tone if requested
        if flags.get("humanize_academic"):
            try:
                # Lazy import to avoid heavy deps when not used
                from app.services.academic_humanizer import AcademicTextHumanizer, download_nltk_resources
                download_nltk_resources()

                # Allow simple tuning via flags dict
                use_passive = bool(flags.get("use_passive", False))
                use_synonyms = bool(flags.get("use_synonyms", False))
                humanizer = AcademicTextHumanizer()
                reply = humanizer.humanize_text(reply, use_passive=use_passive, use_synonyms=use_synonyms)
            except Exception:
                # Fail open: if humanization fails, return the original reply
                pass
        # Add assistant reply to conversation history
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get current conversation messages"""
        return self.messages.copy()

    def set_flags(self, flags: dict):
        self.last_flags = flags

    def set_score(self, score):
        self.score = score

    def update_context(self, **kwargs):
        """Update conversation context with new information"""
        for key, value in kwargs.items():
            if key in self.conversation_context:
                self.conversation_context[key] = value

    def get_context_summary(self):
        """Generate a context summary for the AI"""
        context_parts = []
        
        if self.conversation_context["current_file"]:
            context_parts.append(f"Currently working on: {self.conversation_context['current_file']}")
        
        if self.conversation_context["current_pass"]:
            context_parts.append(f"Current pass: {self.conversation_context['current_pass']}")
        
        if self.conversation_context["recent_changes"]:
            changes = self.conversation_context["recent_changes"][-3:]  # Last 3 changes
            context_parts.append(f"Recent changes: {', '.join(changes)}")
        
        if self.conversation_context["user_preferences"]:
            prefs = self.conversation_context["user_preferences"]
            context_parts.append(f"User preferences: {', '.join(f'{k}={v}' for k, v in prefs.items())}")
        
        if self.conversation_context["session_goals"]:
            goals = self.conversation_context["session_goals"]
            context_parts.append(f"Session goals: {', '.join(goals)}")
        
        return "\n".join(context_parts) if context_parts else "No specific context available."

    def extract_context_from_message(self, message):
        """Extract context information from user message"""
        message_lower = message.lower()
        
        # Extract file references
        if "file" in message_lower and ("current" in message_lower or "working" in message_lower):
            # Try to extract filename from message
            import re
            file_match = re.search(r'file[:\s]+([^\s,]+)', message)
            if file_match:
                self.conversation_context["current_file"] = file_match.group(1)
        
        # Extract pass references
        if "pass" in message_lower:
            import re
            pass_match = re.search(r'pass[:\s]+(\d+)', message)
            if pass_match:
                self.conversation_context["current_pass"] = int(pass_match.group(1))
        
        # Extract preferences
        if "prefer" in message_lower or "like" in message_lower:
            if "formal" in message_lower:
                self.conversation_context["user_preferences"]["tone"] = "formal"
            elif "casual" in message_lower:
                self.conversation_context["user_preferences"]["tone"] = "casual"
            elif "academic" in message_lower:
                self.conversation_context["user_preferences"]["tone"] = "academic"
        
        # Extract goals
        if "goal" in message_lower or "want" in message_lower or "need" in message_lower:
            if "reduce" in message_lower and "ai" in message_lower:
                self.conversation_context["session_goals"].append("reduce AI detection")
            if "improve" in message_lower and "readability" in message_lower:
                self.conversation_context["session_goals"].append("improve readability")
            if "maintain" in message_lower and "meaning" in message_lower:
                self.conversation_context["session_goals"].append("maintain meaning")






