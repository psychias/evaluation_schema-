import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from huggingface_hub import model_info
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from eval_converters.common.error import AdapterError, TransformationError
from eval_types import EvaluationLog

@dataclass
class AdapterMetadata:
    name: str
    version: str
    supported_library_versions: List[str]
    description: str

    
class SupportedLibrary(Enum):
    LM_EVAL = "lm-evaluation-harness"
    INSPECT_AI = "inspect-ai"
    HELM = "helm"
    CUSTOM = "custom"


class BaseEvaluationAdapter(ABC):
    """Base class for evaluation library adapters. Transforms library-specific output to EEE schema."""
    
    def __init__(self, strict_validation: bool = True):
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata: ...
    
    @property
    @abstractmethod
    def supported_library(self) -> SupportedLibrary: ...
    
    @abstractmethod
    def _transform_single(
        self, raw_data: Any, metadata_args: Dict[str, Any]
    ) -> EvaluationLog: ...
    
    def transform(
        self, data: Any, metadata_args: Dict[str, Any]
    ) -> Union[
        EvaluationLog,
        List[EvaluationLog]
    ]:
        """Transform evaluation data to EEE schema format."""
        try:
            # Handle both single records and lists
            if isinstance(data, list):
                results = []
                for i, item in enumerate(data):
                    try:
                        result = self._transform_single(item, metadata_args)
                        results.append(result)
                    except Exception as e:
                        self._handle_transformation_error(e, f"item {i}")
                return results
            else:
                return self._transform_single(data, metadata_args)
                
        except Exception as e:
            self._handle_transformation_error(e, "data transformation")
            
    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> Union[
        EvaluationLog,
        List[EvaluationLog]
    ]:
        """Load and transform evaluation data from a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AdapterError(f"File not found: {file_path}")
        
        try:
            data = self._load_file(file_path)
            return self.transform(data, metadata_args)
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)}")
        
    @abstractmethod
    def transform_from_directory(
        self, 
        dir_path: Union[str, Path],
        metadata_args: Dict[str, Any] = None    
    ) -> Union[
        EvaluationLog,
        List[EvaluationLog]
    ]:
        """Load and transform all evaluation files in a directory."""
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise AdapterError(f"Path is not a directory: {dir_path}")
        
        # Subclass must implement this part
        # e.g., how to iterate through files and process them
        pass

    def _load_file(self, file_path: Path) -> Any:
        """Load data from JSON/JSONL file. Override for custom formats."""
        # Default implementation for JSON files
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix.lower() == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            raise AdapterError(f"Unsupported file format: {file_path.suffix}")
    
    def _handle_transformation_error(self, error: Exception, context: str):
        """Handle transformation errors based on strict_validation setting"""
        error_msg = f"Transformation error in {context}: {str(error)}"
        
        if self.strict_validation:
            raise TransformationError(error_msg) from error
        else:
            self.logger.warning(error_msg)

    def _check_if_model_is_on_huggingface(self, model_path):
        try:
            info = model_info(model_path)
            return info
        except Exception:
            # self.logger.warning(f"Model '{model_path}' not found on Hugging Face.")
            pass