# File: utils/response_formatter.py
from flask import jsonify
from typing import Any, Dict, Optional
from datetime import datetime

class ResponseFormatter:
    """Utility class for formatting API responses"""
    
    @staticmethod
    def success(data: Any, message: str = "Success", status_code: int = 200) -> tuple:
        """Format successful response"""
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(response), status_code
    
    @staticmethod
    def error(message: str, status_code: int = 400, details: Optional[Dict] = None) -> tuple:
        """Format error response"""
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            response["details"] = details
            
        return jsonify(response), status_code
    
    @staticmethod
    def validation_error(errors: Dict[str, str]) -> tuple:
        """Format validation error response"""
        return ResponseFormatter.error(
            message="Validation failed",
            status_code=422,
            details={"validation_errors": errors}
        )
# File: utils/response_formatter.py
from flask import jsonify
from typing import Any, Dict, Optional
from datetime import datetime

class ResponseFormatter:
    """Utility class for formatting API responses"""
    
    @staticmethod
    def success(data: Any, message: str = "Success", status_code: int = 200) -> tuple:
        """Format successful response"""
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(response), status_code
    
    @staticmethod
    def error(message: str, status_code: int = 400, details: Optional[Dict] = None) -> tuple:
        """Format error response"""
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            response["details"] = details
            
        return jsonify(response), status_code
    
    @staticmethod
    def validation_error(errors: Dict[str, str]) -> tuple:
        """Format validation error response"""
        return ResponseFormatter.error(
            message="Validation failed",
            status_code=422,
            details={"validation_errors": errors}
        )