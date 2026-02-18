"""Base agent class with LLM client abstraction."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the M3 pipeline.

    Handles LLM client initialization and provides common utilities
    for structured output generation.
    """

    # Subclasses should override these
    agent_name: str = "base"
    default_provider: str = "openai"
    default_model: str = "gpt-4o"

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        env_path: Optional[Path] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the agent with an LLM client.

        Args:
            provider: LLM provider (openai, anthropic, google, moonshot).
            model: Model identifier.
            env_path: Path to environment file with API keys.
            temperature: Sampling temperature for generation.
        """
        self.provider = provider or self.default_provider
        self.model = model or self.default_model
        self.temperature = temperature

        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            # Try default locations
            for path in [Path("keyholder.env"), Path(".env")]:
                if path.exists():
                    load_dotenv(path)
                    break

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self._client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self._client = Anthropic(api_key=api_key)

        elif self.provider == "google":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)

        elif self.provider == "moonshot":
            from openai import OpenAI
            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("MOONSHOT_API_KEY not found in environment")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_llm(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Call the LLM with the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens in response.
            response_format: Optional response format spec (for JSON mode).

        Returns:
            The model's response text.
        """
        if self.provider in ("openai", "moonshot"):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            # Convert messages to Anthropic format
            system_msg = None
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
            }
            if system_msg:
                kwargs["system"] = system_msg

            response = self._client.messages.create(**kwargs)
            return response.content[0].text

        elif self.provider == "google":
            # Combine messages into a single prompt for Gemini
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"Instructions: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n\n".join(prompt_parts)
            response = self._client.generate_content(prompt)
            return response.text

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_structured_output(
        self,
        response: str,
        output_type: Type[T],
    ) -> T:
        """
        Parse LLM response into a Pydantic model.

        Expects the response to contain valid JSON matching the model schema.
        Handles common issues like markdown code blocks.

        Args:
            response: Raw LLM response text.
            output_type: Pydantic model class to parse into.

        Returns:
            Parsed Pydantic model instance.
        """
        import json
        import re

        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            # Remove opening fence (possibly with language specifier)
            text = re.sub(r"^```\w*\n?", "", text)
            # Remove closing fence
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        def sanitize_json_string(s: str) -> str:
            """Replace control characters with their escaped equivalents."""
            # Replace common control characters with escape sequences
            result = []
            i = 0
            in_string = False
            while i < len(s):
                char = s[i]

                # Track if we're inside a JSON string
                if char == '"' and (i == 0 or s[i-1] != '\\'):
                    in_string = not in_string
                    result.append(char)
                elif in_string and ord(char) < 32:
                    # Escape control characters inside strings
                    if char == '\n':
                        result.append('\\n')
                    elif char == '\r':
                        result.append('\\r')
                    elif char == '\t':
                        result.append('\\t')
                    else:
                        result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
                i += 1
            return ''.join(result)

        # Parse JSON with multiple fallback strategies
        parse_error = None

        # Strategy 1: Direct parse
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            parse_error = e
            data = None

        # Strategy 2: Use strict=False to allow control characters
        if data is None:
            try:
                data = json.loads(text, strict=False)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Sanitize control characters and retry
        if data is None:
            try:
                sanitized = sanitize_json_string(text)
                data = json.loads(sanitized)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Extract JSON object from response and try again
        if data is None:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                extracted = match.group()
                try:
                    data = json.loads(extracted, strict=False)
                except json.JSONDecodeError:
                    try:
                        sanitized = sanitize_json_string(extracted)
                        data = json.loads(sanitized)
                    except json.JSONDecodeError:
                        pass

        # Strategy 5: Fix unterminated strings by adding closing quotes
        if data is None:
            def fix_unterminated_strings(s: str) -> str:
                """Try to fix unterminated strings in JSON."""
                lines = s.split('\n')
                fixed_lines = []
                for line in lines:
                    # Count quotes in line
                    quote_count = line.count('"') - line.count('\\"')
                    if quote_count % 2 == 1:
                        # Odd number of quotes - add one at end
                        line = line.rstrip(',') + '",'
                    fixed_lines.append(line)
                return '\n'.join(fixed_lines)

            try:
                fixed = fix_unterminated_strings(text)
                data = json.loads(fixed, strict=False)
            except json.JSONDecodeError:
                pass

        # Strategy 6: Use regex to build JSON from key-value patterns
        if data is None:
            try:
                # Extract fields manually for common model outputs
                framework_match = re.search(r'"framework"\s*:\s*"([^"]*)"', text)
                equations_match = re.search(r'"equations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                rationale_match = re.search(r'"rationale"\s*:\s*"(.*?)"(?:,|\})', text, re.DOTALL)

                if framework_match:
                    manual_data = {
                        "framework": framework_match.group(1) if framework_match else "Unknown",
                        "equations": [],
                        "variables": {},
                        "parameters": {},
                        "boundary_conditions": [],
                        "rationale": rationale_match.group(1) if rationale_match else "See model description"
                    }
                    if equations_match:
                        eq_text = equations_match.group(1)
                        eqs = re.findall(r'"([^"]*)"', eq_text)
                        manual_data["equations"] = eqs
                    data = manual_data
            except Exception:
                pass

        if data is None:
            raise ValueError(f"Could not parse JSON from response: {parse_error}")

        return output_type.model_validate(data)

    def generate_structured(
        self,
        prompt: str,
        output_type: Type[T],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> T:
        """
        Generate a structured output from a prompt.

        Args:
            prompt: User prompt.
            output_type: Pydantic model class for output.
            system_prompt: Optional system instructions.
            max_tokens: Maximum response tokens.

        Returns:
            Parsed Pydantic model instance.
        """
        # Build schema description for the model
        schema = output_type.model_json_schema()
        schema_str = self._format_schema_for_prompt(schema)

        messages = []

        base_system = f"""You are an expert agent in an M3 math modeling competition system.
You must respond with valid JSON matching this schema:

{schema_str}

Respond ONLY with the JSON object, no additional text or markdown."""

        if system_prompt:
            base_system = f"{system_prompt}\n\n{base_system}"

        messages.append({"role": "system", "content": base_system})
        messages.append({"role": "user", "content": prompt})

        # Request JSON response format for OpenAI
        response_format = None
        if self.provider == "openai":
            response_format = {"type": "json_object"}

        response = self._call_llm(
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        return self._parse_structured_output(response, output_type)

    def _format_schema_for_prompt(self, schema: dict) -> str:
        """Format JSON schema as readable text for LLM prompt."""
        import json
        # Simplify schema for prompt
        simplified = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        return json.dumps(simplified, indent=2)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the agent's main task.

        Subclasses must implement this method.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model={self.model})"
