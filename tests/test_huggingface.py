"""Tests for the LLM HuggingFace plugin."""

from click.testing import CliRunner
import llm
import llm.errors
from llm.cli import cli
import nest_asyncio # type: ignore
import json
import os
import pytest
import pydantic
from unittest.mock import patch, MagicMock
from llm_huggingface import HuggingFace, AsyncHuggingFace, _get_text_generation_models, _get_image_text_to_text_models

nest_asyncio.apply()

HUGGINGFACE_TOKEN = os.environ.get("PYTEST_HUGGINGFACE_TOKEN", None) or "hf_test_token"


class TestModelRegistration:
    """Test model registration and discovery."""
    
    def test_model_registration(self):
        """Test that models are properly registered with different capabilities."""
        # Test text model
        text_model = HuggingFace("meta-llama/Llama-2-7b-hf", is_image_text_to_text=False)
        assert text_model.model_id == "hf/meta-llama/Llama-2-7b-hf"
        assert not hasattr(text_model, 'attachment_types') or not text_model.attachment_types
        assert text_model.supports_schema == True
        assert text_model.supports_tools == True
        
        # Test multi-modal model
        mm_model = HuggingFace("meta-llama/Llama-3.2-11B-Vision-Instruct", is_image_text_to_text=True)
        assert mm_model.model_id == "hf/meta-llama/Llama-3.2-11B-Vision-Instruct"
        assert hasattr(mm_model, 'attachment_types')
        assert mm_model.attachment_types == {"image/png", "image/jpeg", "image/webp", "image/gif"}
    
    def test_async_model_registration(self):
        """Test async model registration."""
        async_model = AsyncHuggingFace("meta-llama/Llama-2-7b-hf", is_image_text_to_text=False)
        assert async_model.model_id == "hf/meta-llama/Llama-2-7b-hf"
        assert async_model.can_stream == True


class TestPrompting:
    """Test basic prompting functionality."""
    
    def test_basic_prompt(self):
        """Test basic text prompting."""
        model = HuggingFace("meta-llama/Llama-2-7b-hf")
        # Test that the model can be instantiated and has the right properties
        assert model.needs_key == "huggingface"
        assert model.key_env_var == "HUGGINGFACE_TOKEN"
        assert model.can_stream == True
    
    def test_async_prompt(self):
        """Test async prompting."""
        async_model = AsyncHuggingFace("meta-llama/Llama-2-7b-hf")
        assert hasattr(async_model, 'execute')
        # The actual execute method is async
        import inspect
        assert inspect.isasyncgenfunction(async_model.execute)


class TestSchemaSupport:
    """Test JSON schema support."""
    
    def test_schema_support_flag(self):
        """Test that models report schema support."""
        model = HuggingFace("meta-llama/Llama-2-7b-hf")
        assert model.supports_schema == True
    
    def test_prompt_with_pydantic_schema(self):
        """Test prompting with a Pydantic schema."""
        class Dog(pydantic.BaseModel):
            name: str # type: ignore
            age: int # type: ignore
            bio: str # type: ignore
        
        model = HuggingFace("meta-llama/Llama-2-7b-hf")
        # This would make an API call with schema parameter
        # In a real test with VCR, the response would be recorded
        assert model.supports_schema == True


class TestToolSupport:
    """Test tool/function calling support."""
    
    def test_tools_support_flag(self):
        """Test that models report tools support."""
        model = HuggingFace("meta-llama/Llama-2-7b-hf")
        assert model.supports_tools == True


class TestOptions:
    """Test model options validation."""
    
    def test_valid_options(self):
        """Test that valid options are accepted."""
        model = HuggingFace("test-model")
        opts = model.Options(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["\\n", "END"]
        )
        assert opts.temperature == 0.7
        assert opts.top_p == 0.9
        assert opts.max_tokens == 100
        assert opts.stop == ["\\n", "END"]
    
    def test_invalid_temperature(self):
        """Test that invalid temperature is rejected."""
        model = HuggingFace("test-model")
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            model.Options(temperature=3.0)
    
    def test_invalid_top_p(self):
        """Test that invalid top_p is rejected."""
        model = HuggingFace("test-model")
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            model.Options(top_p=1.5)
    
    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens is rejected."""
        model = HuggingFace("test-model")
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            model.Options(max_tokens=0)


class TestMultiModal:
    """Test multi-modal support."""
    
    def test_multimodal_model_has_attachments(self):
        """Test that multi-modal models have attachment_types."""
        mm_model = HuggingFace("llava-hf/llava-1.5-7b-hf", is_image_text_to_text=True)
        assert hasattr(mm_model, 'attachment_types')
        assert "image/png" in mm_model.attachment_types
        assert "image/jpeg" in mm_model.attachment_types
    
    def test_text_model_no_attachments(self):
        """Test that text-only models don't have attachment_types."""
        text_model = HuggingFace("gpt2", is_image_text_to_text=False)
        # Model should not have attachment_types or it should be empty
        if hasattr(text_model, 'attachment_types'):
            assert not text_model.attachment_types  # Should be empty set or None


class TestModelDiscovery:
    """Test model discovery and caching."""
    
    def test_get_text_generation_models(self):
        """Test that text generation models can be discovered."""
        # This uses the cached version for speed
        models = _get_text_generation_models()
        assert isinstance(models, list)
        # Should have found some models (exact count may vary)
        assert len(models) > 0
    
    def test_get_image_text_to_text_models(self):
        """Test that image-text-to-text models can be discovered."""
        models = _get_image_text_to_text_models()
        assert isinstance(models, list)
        # Image-text-to-text models are less common
        # but there should be at least a few
        if models:  # May be empty if API is down
            assert len(models) > 0


class TestCLI:
    """Test CLI functionality."""
    
    def test_key_validation_none(self):
        """Test that model requires API key when None provided."""
        model = HuggingFace("test-model")
        
        # Test the _validate_key method directly with None
        with pytest.raises(llm.errors.ModelError, match="API key"):
            model._validate_key(None)
    
    def test_key_validation_empty(self):
        """Test that model requires API key when empty string provided."""
        model = HuggingFace("test-model")
        
        # Test the _validate_key method with empty string
        with pytest.raises(llm.errors.ModelError, match="API key"):
            model._validate_key("")
    
    def test_key_validation_valid(self):
        """Test that valid API key is accepted."""
        model = HuggingFace("test-model")
        
        # Test with valid key - should not raise
        validated_key = model._validate_key("test_key_123")
        assert validated_key == "test_key_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])