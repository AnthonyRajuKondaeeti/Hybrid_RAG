# File: utils/response_formatter.py
from flask import jsonify
from typing import Any, Dict, Optional, List, Union
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
    def validation_error(errors: Union[Dict[str, str], List[str]] = None, message: str = "Validation failed") -> tuple:
        """Format validation error response"""
        if isinstance(errors, list):
            # Handle list of error strings
            return ResponseFormatter.error(
                message=message,
                status_code=400,
                details={"errors": errors}
            )
        elif isinstance(errors, dict):
            # Handle dict of validation errors
            return ResponseFormatter.error(
                message=message,
                status_code=422,
                details={"validation_errors": errors}
            )
        else:
            return ResponseFormatter.error(
                message=message,
                status_code=400
            )
    
    @staticmethod
    def success_response(data: Any, message: str = "Success", status: int = 200) -> tuple:
        """Format successful response - alias for success method"""
        response = {
            "success": True,
            "message": message,
            "data": data,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(response), status
    
    @staticmethod
    def error_response(message: str, status: int = 400, details: Optional[Dict] = None) -> tuple:
        """Format error response - alias for error method"""
        response = {
            "success": False,
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            response["details"] = details
            
        return jsonify(response), status