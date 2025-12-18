from __future__ import annotations

from typing import Protocol, List, Dict, Any
import time
import math
import os
import json
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
try:  # optional dependency for token budgeting
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

# Module-level logger
logger = logging.getLogger(__name__)

# OpenAI Pricing (as of 2024)
OPENAI_PRICING = {
    'gpt-4': {
        'input': 0.03,  # $0.03 per 1K tokens
        'output': 0.06  # $0.06 per 1K tokens
    },
    'gpt-4-turbo': {
        'input': 0.01,  # $0.01 per 1K tokens
        'output': 0.03  # $0.03 per 1K tokens
    },
    'gpt-4o': {
        'input': 0.0025,  # $0.0025 per 1K tokens (updated pricing)
        'output': 0.01    # $0.01 per 1K tokens (updated pricing)
    },
    'gpt-4o-mini': {
        'input': 0.00015,  # $0.00015 per 1K tokens (very cheap!)
        'output': 0.0006   # $0.0006 per 1K tokens
    },
    'gpt-3.5-turbo': {
        'input': 0.0005,  # $0.0005 per 1K tokens (updated pricing)
        'output': 0.0015  # $0.0015 per 1K tokens
    }
}

# Cost limits (hard-coded defaults so no env configuration is required)
# You can change these values directly in code if needed.
MAX_COST_PER_JOB: float = 5.0            # Max $ per job
MAX_COST_PER_USER_DAILY: float = 50.0    # Max $ per user per day
MAX_TOKENS_PER_JOB: int = 500_000        # Max tokens per job

class CostLimitExceeded(Exception):
    """Raised when a job or user exceeds their cost limit."""
    def __init__(self, message: str, current_cost: float, limit: float, limit_type: str = "job"):
        self.message = message
        self.current_cost = current_cost
        self.limit = limit
        self.limit_type = limit_type
        super().__init__(message)

def calculate_cost(tokens_in: int, tokens_out: int, model: str = "gpt-4") -> dict:
    """Calculate cost for given token usage and model"""
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING['gpt-4'])
    
    input_cost = (tokens_in / 1000) * pricing['input']
    output_cost = (tokens_out / 1000) * pricing['output']
    total_cost = input_cost + output_cost
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'tokens_in': tokens_in,
        'tokens_out': tokens_out,
        'model': model
    }

# Simple in-memory analytics store with disk persistence (exported for API)
class _Analytics:
    def __init__(self) -> None:
        self.total_requests = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost = 0.0
        self.current_model = "gpt-4"  # Track current model
        # ring buffer of (ts_sec, in_tokens, out_tokens, model, cost)
        self.events: deque[tuple[int,int,int,str,float]] = deque(maxlen=10_000)
        # Job-level cost tracking
        self.job_costs: Dict[str, float] = {}  # job_id -> total_cost
        self.pass_costs: Dict[str, List[float]] = {}  # job_id -> [pass1_cost, pass2_cost, ...]
        self.job_tokens: Dict[str, int] = {}  # job_id -> total tokens used
        # User-level daily cost tracking
        self.user_daily_costs: Dict[str, Dict[str, float]] = {}  # user_id -> {date: cost}
        # Schema usage tracking
        self.schema_usage: Dict[str, int] = {}  # schema_id -> usage_count
        self.schema_last_used: Dict[str, str] = {}  # schema_id -> last_used_timestamp
        
        # Persistence file path
        from app.core.paths import get_backend_root
        from pathlib import Path
        backend_root = Path(get_backend_root())
        self._persist_file = str(backend_root / "data" / "analytics_store.json")
        
        # Load persisted data on startup
        self._load_from_disk()
    
    def _load_from_disk(self) -> None:
        """Load analytics data from disk if it exists"""
        logger = logging.getLogger(__name__)
        try:
            if os.path.exists(self._persist_file):
                logger.debug(f"Loading analytics from disk: {self._persist_file}")
                with open(self._persist_file, 'r') as f:
                    data = json.load(f)
                    self.total_requests = data.get('total_requests', 0)
                    self.total_tokens_in = data.get('total_tokens_in', 0)
                    self.total_tokens_out = data.get('total_tokens_out', 0)
                    self.total_cost = data.get('total_cost', 0.0)
                    self.current_model = data.get('current_model', 'gpt-4')
                    self.job_costs = data.get('job_costs', {})
                    self.pass_costs = data.get('pass_costs', {})
                    self.schema_usage = data.get('schema_usage', {})
                    self.schema_last_used = data.get('schema_last_used', {})
                    # Restore events (limited to last 10k)
                    events_data = data.get('events', [])
                    self.events = deque(events_data[-10_000:], maxlen=10_000)
                    logger.info(f"Loaded analytics from disk: {self.total_requests} requests, ${self.total_cost:.6f} total cost")
            else:
                logger.debug(f"Analytics file not found at {self._persist_file}, starting with empty analytics")
        except Exception as e:
            logger.warning(f"Failed to load analytics from disk: {e}", exc_info=True)
            # Continue with defaults
    
    def _save_to_disk(self) -> None:
        """Save analytics data to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._persist_file), exist_ok=True)
            
            # Convert deque to list for JSON serialization
            events_list = list(self.events)
            
            data = {
                'total_requests': self.total_requests,
                'total_tokens_in': self.total_tokens_in,
                'total_tokens_out': self.total_tokens_out,
                'total_cost': self.total_cost,
                'current_model': self.current_model,
                'job_costs': self.job_costs,
                'pass_costs': self.pass_costs,
                'schema_usage': self.schema_usage,
                'schema_last_used': self.schema_last_used,
                'events': events_list,
            }
            
            # Atomic write: write to temp file, then rename
            temp_file = self._persist_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            os.replace(temp_file, self._persist_file)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to save analytics to disk: {e}", exc_info=True)
            # Don't fail the request if persistence fails

    def add(self, in_tokens: int, out_tokens: int, model: str = "gpt-4", job_id: str = None, user_id: str = None) -> dict:
        now = int(time.time())
        self.total_requests += 1
        self.total_tokens_in += max(0, in_tokens)
        self.total_tokens_out += max(0, out_tokens)
        self.current_model = model
        
        # Calculate cost for this request
        cost_info = calculate_cost(in_tokens, out_tokens, model)
        self.total_cost += cost_info['total_cost']
        
        # Track job-level costs and tokens
        if job_id:
            if job_id not in self.job_costs:
                self.job_costs[job_id] = 0.0
                self.pass_costs[job_id] = []
                self.job_tokens[job_id] = 0
            self.job_costs[job_id] += cost_info['total_cost']
            self.pass_costs[job_id].append(cost_info['total_cost'])
            self.job_tokens[job_id] += max(0, in_tokens) + max(0, out_tokens)
        
        # Track user daily costs
        if user_id:
            self.update_user_daily_cost(user_id, cost_info['total_cost'])
        
        self.events.append((now, in_tokens, out_tokens, model, cost_info['total_cost']))
        
        # Store to MongoDB (non-blocking, in background thread)
        if user_id:
            try:
                import threading
                from app.core.mongodb_db import db
                # Run in background thread to avoid blocking
                def _store():
                    try:
                        if db.is_connected():
                            success = db.store_usage_stats(
                                user_id=user_id,
                                request_count=1,
                                tokens_in=max(0, in_tokens),
                                tokens_out=max(0, out_tokens),
                                cost=cost_info['total_cost'],
                                model=model,
                                job_id=job_id
                            )
                            if not success:
                                logger.debug(f"MongoDB storage returned False for user_id={user_id}")
                    except Exception as e:
                        logger.warning(f"Exception in MongoDB storage thread: {e}")
                thread = threading.Thread(target=_store, daemon=True)
                thread.start()
            except Exception as e:
                # Don't fail the request if MongoDB storage fails
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to start MongoDB storage thread: {e}")
        
        # Persist to disk after each update (async, non-blocking)
        try:
            self._save_to_disk()
        except Exception as e:
            # Don't fail the request if persistence fails
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to persist analytics: {e}", exc_info=True)
        
        return cost_info

    def check_job_cost_limit(self, job_id: str, estimated_cost: float = 0.0) -> None:
        """Check if adding estimated_cost would exceed job cost limit.
        Raises CostLimitExceeded if limit would be exceeded.
        """
        if not job_id:
            return
        
        current_cost = self.job_costs.get(job_id, 0.0)
        projected_cost = current_cost + estimated_cost
        
        if projected_cost > MAX_COST_PER_JOB:
            raise CostLimitExceeded(
                f"Job cost limit exceeded: ${projected_cost:.4f} > ${MAX_COST_PER_JOB:.2f}",
                current_cost=current_cost,
                limit=MAX_COST_PER_JOB,
                limit_type="job"
            )
        
        # Also check token limit
        current_tokens = self.job_tokens.get(job_id, 0)
        if current_tokens > MAX_TOKENS_PER_JOB:
            raise CostLimitExceeded(
                f"Job token limit exceeded: {current_tokens:,} > {MAX_TOKENS_PER_JOB:,}",
                current_cost=current_cost,
                limit=MAX_TOKENS_PER_JOB,
                limit_type="tokens"
            )

    def check_user_daily_limit(self, user_id: str, estimated_cost: float = 0.0) -> None:
        """Check if user's daily spending would exceed limit.
        Raises CostLimitExceeded if limit would be exceeded.
        """
        if not user_id:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        user_costs = self.user_daily_costs.get(user_id, {})
        current_daily_cost = user_costs.get(today, 0.0)
        projected_cost = current_daily_cost + estimated_cost
        
        if projected_cost > MAX_COST_PER_USER_DAILY:
            raise CostLimitExceeded(
                f"Daily user cost limit exceeded: ${projected_cost:.4f} > ${MAX_COST_PER_USER_DAILY:.2f}",
                current_cost=current_daily_cost,
                limit=MAX_COST_PER_USER_DAILY,
                limit_type="user_daily"
            )

    def update_user_daily_cost(self, user_id: str, cost: float) -> None:
        """Update user's daily cost tracking."""
        if not user_id:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        if user_id not in self.user_daily_costs:
            self.user_daily_costs[user_id] = {}
        
        current = self.user_daily_costs[user_id].get(today, 0.0)
        self.user_daily_costs[user_id][today] = current + cost
        
        # Cleanup old dates (keep only last 7 days)
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        self.user_daily_costs[user_id] = {
            k: v for k, v in self.user_daily_costs[user_id].items()
            if k >= cutoff
        }

    def update_job_tokens(self, job_id: str, tokens: int) -> None:
        """Update job token count."""
        if not job_id:
            return
        current = self.job_tokens.get(job_id, 0)
        self.job_tokens[job_id] = current + tokens

    def get_job_stats(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive stats for a job."""
        return {
            "job_id": job_id,
            "total_cost": self.job_costs.get(job_id, 0.0),
            "total_tokens": self.job_tokens.get(job_id, 0),
            "pass_costs": self.pass_costs.get(job_id, []),
            "pass_count": len(self.pass_costs.get(job_id, [])),
            "cost_limit": MAX_COST_PER_JOB,
            "token_limit": MAX_TOKENS_PER_JOB,
            "cost_remaining": MAX_COST_PER_JOB - self.job_costs.get(job_id, 0.0),
            "tokens_remaining": MAX_TOKENS_PER_JOB - self.job_tokens.get(job_id, 0)
        }

    def summary_last_24h(self) -> Dict[str, Any]:
        now = int(time.time())
        cutoff = now - 24*3600
        per_hour = defaultdict(lambda: {"requests":0, "tokens_in":0, "tokens_out":0, "cost":0.0})
        reqs = 0; ti = 0; to = 0; cost_24h = 0.0
        for ts, tin, tout, model, cost in list(self.events):
            if ts < cutoff: continue
            hour_bucket = ts - (ts % 3600)
            b = per_hour[hour_bucket]
            b["requests"] += 1
            b["tokens_in"] += max(0,tin)
            b["tokens_out"] += max(0,tout)
            b["cost"] += cost
            reqs += 1; ti += max(0,tin); to += max(0,tout); cost_24h += cost
        series = [
            {"hour": k, **per_hour[k]}
            for k in sorted(per_hour.keys())
        ]
        return {"requests": reqs, "tokens_in": ti, "tokens_out": to, "cost": cost_24h, "series": series}
    
    def get_job_cost(self, job_id: str) -> dict:
        """Get cost information for a specific job"""
        total_cost = self.job_costs.get(job_id, 0.0)
        pass_costs = self.pass_costs.get(job_id, [])
        return {
            "job_id": job_id,
            "total_cost": total_cost,
            "pass_costs": pass_costs,
            "pass_count": len(pass_costs),
            "avg_cost_per_pass": total_cost / max(len(pass_costs), 1)
        }
    
    def track_schema_usage(self, schema_id: str, schema_level: int, user_id: str = None) -> None:
        """Track schema usage for analytics"""
        if schema_level > 0:  # Only track active schemas
            self.schema_usage[schema_id] = self.schema_usage.get(schema_id, 0) + 1
            self.schema_last_used[schema_id] = datetime.now().isoformat()
            
            # Store to MongoDB (non-blocking, in background thread)
            if user_id:
                try:
                    import threading
                    from app.core.mongodb_db import db
                    def _store():
                        try:
                            if db.is_connected():
                                success = db.store_schema_usage(user_id, schema_id)
                                if not success:
                                    logger.debug(f"MongoDB schema storage returned False for user_id={user_id}, schema={schema_id}")
                        except Exception as e:
                            logger.warning(f"Exception in MongoDB schema storage thread: {e}")
                    thread = threading.Thread(target=_store, daemon=True)
                    thread.start()
                except Exception as e:
                    logger.debug(f"Failed to start MongoDB schema storage thread: {e}")
            
            # Persist to disk
            try:
                self._save_to_disk()
            except Exception:
                pass  # Don't fail if persistence fails
    
    def get_schema_usage_stats(self) -> dict:
        """Get schema usage statistics"""
        total_usages = sum(self.schema_usage.values())
        most_used = max(self.schema_usage.items(), key=lambda x: x[1]) if self.schema_usage else None
        least_used = min(self.schema_usage.items(), key=lambda x: x[1]) if self.schema_usage else None
        
        return {
            "total_usages": total_usages,
            "most_used_schema": most_used[0] if most_used else None,
            "most_used_count": most_used[1] if most_used else 0,
            "least_used_schema": least_used[0] if least_used else None,
            "least_used_count": least_used[1] if least_used else 0,
            "average_usage": total_usages / len(self.schema_usage) if self.schema_usage else 0,
            "schema_usage": dict(self.schema_usage),
            "schema_last_used": dict(self.schema_last_used),
        }

analytics_store = _Analytics()


class LanguageModel(Protocol):
    def generate(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000) -> str: ...


class OpenAIModel:
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Track request timeout (configurable via env)
        self._request_timeout = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))  # 2 min default

    def generate(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000, job_id: str = None, user_id: str = None, fast_mode: bool = False, pass_num: int = 1, total_passes: int = 1) -> tuple[str, dict]:
        """Generate text using OpenAI API with cost controls and model tiering.
        
        Args:
            system: System prompt
            user: User prompt/content
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            job_id: Job ID for tracking
            user_id: User ID for tracking
            fast_mode: Use cheaper model (gpt-4o-mini)
            pass_num: Current pass number (1-indexed)
            total_passes: Total number of passes requested
            
        Returns:
            Tuple of (generated_text, cost_info)
        """
        t0 = time.perf_counter()
        
        # Check cost limits before proceeding
        try:
            # Estimate cost for this request (rough estimate: input tokens * 2 for output)
            estimated_tokens = (len(system) + len(user)) // 4  # Rough char-to-token ratio
            estimated_cost = (estimated_tokens / 1000) * 0.005  # Conservative estimate
            
            analytics_store.check_job_cost_limit(job_id, estimated_cost)
            analytics_store.check_user_daily_limit(user_id, estimated_cost)
        except CostLimitExceeded as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Cost limit exceeded: {e.message}")
            raise  # Re-raise to be handled by caller
        
        def _estimate_max_chars_for_model(model: str, max_input_tokens: int = 7000) -> int:
            # Heuristic: ~4 chars/token. Leave room for system + output tokens.
            return max(2000, max_input_tokens * 4)

        def _truncate_text(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text
            # Prefer keeping the beginning (context/structure) for refinement
            return text[:max_chars]

        # Best-effort: attempt the call, and on context errors, progressively truncate
        attempt_user = user
        max_chars = _estimate_max_chars_for_model(self.model)
        if len(attempt_user) > max_chars:
            attempt_user = _truncate_text(attempt_user, max_chars)

        # Optional token budget pre-check using tiktoken if enabled
        max_in = int(os.environ.get("REFINER_MAX_INPUT_TOKENS", "0") or 0)
        if max_in > 0 and tiktoken is not None:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            pre_tokens = len(enc.encode(system)) + len(enc.encode(attempt_user))
            if pre_tokens > max_in:
                raise ValueError(f"TOKEN_BEDGET_EXCEEDED: {pre_tokens}>{max_in}")

        # MODEL TIERING: Use cheaper model for early passes, premium for final.
        # These are hard-coded defaults so you don't need environment variables.
        use_model = self.model
        model_tier_enabled = True  # Always enable tiering by default

        EARLY_PASS_MODEL = "gpt-4o-mini"   # Model for non-final passes
        FINAL_PASS_MODEL = "gpt-4o"        # Model for final pass
        FAST_MODE_MODEL = "gpt-4o-mini"    # Model for explicit fast mode
        
        if model_tier_enabled and total_passes > 1:
            # Early passes (all except final): use fast/cheap model
            if pass_num < total_passes:
                use_model = EARLY_PASS_MODEL
            else:
                # Final pass: use premium model for polish
                use_model = FINAL_PASS_MODEL
        elif fast_mode:
            # Explicit fast mode override
            use_model = FAST_MODE_MODEL
        
        try:
            resp = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": attempt_user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._request_timeout,  # Add timeout to prevent hanging
            )
        except Exception as e:
            # Handle context length exceeded by retrying with more aggressive truncation
            err_str = str(e)
            if "context length" in err_str or "maximum context length" in err_str or "context_length_exceeded" in err_str:
                # Retry with 50% size, then 25%, etc.
                for factor in (0.5, 0.25, 0.125):
                    try:
                        reduced = _truncate_text(user, int(max_chars * factor))
                        resp = self.client.chat.completions.create(
                            model=use_model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": reduced},
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=self._request_timeout,
                        )
                        break
                    except Exception as inner_e:
                        if factor == 0.125:
                            raise inner_e
                        continue
            else:
                raise

        content = resp.choices[0].message.content or ""
        # Token usage (OpenAI provides usage fields)
        logger = logging.getLogger(__name__)
        try:
            # Access usage object correctly - OpenAI SDK v1+ uses resp.usage directly
            usage = resp.usage if hasattr(resp, 'usage') else None
            if usage:
                used_in = int(usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else (getattr(usage, 'prompt_tokens', 0) or 0))
                used_out = int(usage.completion_tokens if hasattr(usage, 'completion_tokens') else (getattr(usage, 'completion_tokens', 0) or 0))
                logger.debug(f"Token usage extracted: {used_in} in, {used_out} out")
            else:
                logger.warning("No usage object found in OpenAI response")
                used_in = 0
                used_out = 0
        except Exception as e:
            logger.warning(f"Failed to extract token usage: {e}", exc_info=True)
            used_in = 0
            used_out = 0
        
        # Normalize model name for pricing lookup (handle variations like "gpt-4o", "gpt-4-turbo", etc.)
        model_name = use_model  # Use the actual model that was used (could be different in fast_mode)
        if model_name.startswith("gpt-4o-mini"):
            model_name = "gpt-4o-mini"
        elif model_name.startswith("gpt-4o"):
            model_name = "gpt-4o"
        elif model_name.startswith("gpt-4-turbo"):
            model_name = "gpt-4-turbo"
        elif model_name.startswith("gpt-4"):
            model_name = "gpt-4"
        elif model_name.startswith("gpt-3.5"):
            model_name = "gpt-3.5-turbo"
        
        # Track cost and return cost information
        cost_info = analytics_store.add(used_in, used_out, model_name, job_id, user_id)
        
        # Log analytics tracking
        logger.debug(f"Analytics tracked: {used_in} in, {used_out} out tokens, cost=${cost_info['total_cost']:.6f}, model={model_name}, job_id={job_id}, user_id={user_id}")
        
        return content, cost_info


