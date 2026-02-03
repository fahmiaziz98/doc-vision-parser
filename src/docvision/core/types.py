import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, List, TypedDict


class ImageFormat(str, Enum):
    """Support image formats"""

    PNG = "png"
    JPEG = "jpeg"


class ParsingMode(str, Enum):
    """Parsing modes"""

    VLM = "parse_with_vlm"  # Fast single-shot parsing
    AGENTIC = "parse_with_agent"


@dataclass
class ParseResult:
    """Result from document parsing"""

    content: str
    page_number: int
    processing_time: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchParseResult:
    """Result from batch parsing"""

    results: List[ParseResult]
    total_pages: int
    total_time: float
    success_count: int
    error_count: int
    errors: List[dict] = field(default_factory=list)


class AgenticParseState(TypedDict):
    """State for agentic parsing workflow"""

    # Input (immutable)
    image_b64: str
    mime_type: str

    accumulated_text: str
    iteration_count: int
    current_prompt: str
    generation_history: Annotated[List[str], operator.add]
