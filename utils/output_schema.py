from pydantic import BaseModel, Field
from typing import Literal, Annotated, List

class ClassificationOutput(BaseModel):
    style_tags: Annotated[
        List[str],
        Field(min_length=1, max_length=4)
    ] = Field(description="Up to 4 descriptive style tags.")

    genre_suggestions: Annotated[
        List[str],
        Field(min_length=1, max_length=3)
    ] = Field(description="Up to 3 genre suggestions.")

    texture: Literal[
        "gritty", "warm", "bright", "dark",
        "percussive", "smooth", "wide"
    ] = Field(description="Texture word.")

    confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0)
    ] = Field(description="Confidence 0–1 score.")


# utils/recommender_schema.py
from typing import Any, Dict

# One recommendation item
class RecommendationItem(BaseModel):
    id: str = Field(..., description="Unique id for the recommendation (string).")
    type: Literal["layer", "fx_chain", "preset_tweak", "sample_keyword", "variation"] = Field(
        ..., description="One of the allowed recommendation types."
    )
    title: str = Field(..., description="Short title for the recommendation.")
    short_description: str = Field(..., description="Concise human-readable description.")
    actionable_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete parameters to apply the recommendation (e.g. cutoff_hz, gain_db, synth name, filter settings...)."
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Confidence score between 0.0 and 1.0."
    )

# Top-level schema: exactly 4 recommendations required
class RecommenderOutput(BaseModel):
    recommendations: Annotated[
        List[RecommendationItem],
        Field(min_length=4, max_length=4, description="Exactly 4 recommendations, ranked highest first.")
    ] = Field(..., description="List of recommendations (length exactly 4).")

    class Config:
        # keep examples, forbid extra? If you want to reject extra top-level keys:
        # extra = "forbid"
        pass


# ========================================== Synthesis Schema ============================================
# utils/synthesis_schema.py
from pydantic import model_validator
from typing import Optional

# Small helper types
AudioPathStr = Annotated[str, Field(
    pattern=r"^tests/sample_audio/[^/\\]+\.(wav|mp3|flac|aiff|ogg|m4a)$",
    description="Relative path under tests/sample_audio/ to an existing audio file"
)]

OutPathStr = Annotated[str, Field(
    pattern=r"^tests/synthesis_demo/.+\.wav$",
    description="Relative output path under tests/synthesis_demo/ and must end with .wav"
)]

MixRatio = Annotated[float, Field(ge=0.0, le=1.0, description="Proportion of original audio in final mix")]


# params submodels
class SubSineParams(BaseModel):
    enabled: bool = Field(..., description="Enable or disable sub sine")
    freq_hz: Annotated[float, Field(ge=20.0, le=20000.0, description="Frequency in Hz")]
    amp: Annotated[float, Field(ge=0.0, le=1.0, description="Amplitude 0-1")]
    lowpass_cutoff: Annotated[float, Field(ge=20.0, le=20000.0, description="Lowpass cutoff Hz")]

    class Config:
        extra = "forbid"


class NoiseParams(BaseModel):
    enabled: bool = Field(..., description="Enable or disable noise")
    amp: Annotated[float, Field(ge=0.0, le=1.0, description="Noise amplitude 0-1")]

    class Config:
        extra = "forbid"


class DistortionParams(BaseModel):
    enabled: bool = Field(..., description="Enable distortion")
    drive: Annotated[float, Field(ge=0.0, le=10.0, description="Distortion drive; typical 0.5-3")]

    class Config:
        extra = "forbid"


class DelayParams(BaseModel):
    enabled: bool = Field(..., description="Enable delay")
    ms: Annotated[int, Field(ge=10, le=600, description="Delay time in ms")]
    feedback: Annotated[float, Field(ge=0.0, le=1.0, description="Delay feedback 0-1")]

    class Config:
        extra = "forbid"


class Params(BaseModel):
    sub_sine: Optional[SubSineParams] = Field(None, description="Sub sine parameters")
    noise: Optional[NoiseParams] = Field(None, description="Noise parameters")
    distortion: Optional[DistortionParams] = Field(None, description="Distortion params")
    delay: Optional[DelayParams] = Field(None, description="Delay params")
    global_lowpass: Optional[Annotated[float, Field(ge=20.0, le=20000.0)]] = Field(
        None, description="Global lowpass cutoff in Hz"
    )
    global_highpass: Optional[Annotated[float, Field(ge=20.0, le=20000.0)]] = Field(
        None, description="Global highpass cutoff in Hz"
    )

    class Config:
        extra = "forbid"


class ArgsModel(BaseModel):
    input_audio_path: AudioPathStr
    out_path: OutPathStr
    sr: Annotated[int, Field(ge=8000, le=192000, description="Sample rate; typically 22050")] = 22050
    mix_ratio: MixRatio
    params: Optional[Params] = Field(default_factory=dict, description="Structured synthesis parameters")

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    def ensure_defaults_for_params(cls, values):
        # Ensure params exists as object (not None) so downstream code has a consistent type
        if "params" not in values or values.get("params") is None:
            values["params"] = {}
        return values


class SynthesisToolCall(BaseModel):
    tool: Literal["synthesis_tool"] = Field(..., description="Tool to call")
    function: Literal["apply_patch"] = Field(..., description="Function to call on the tool")
    args: ArgsModel

    class Config:
        extra = "forbid"
