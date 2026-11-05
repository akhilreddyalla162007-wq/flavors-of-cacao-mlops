
from pydantic import BaseModel, Field


class CocoaInput(BaseModel):
    Company: str = Field(..., description="Chocolate manufacturer")
    Specific_Bean_Origin: str
    Review_Date: int = Field(..., ge=1900, le=2100)
    Cocoa_Percent: float = Field(..., gt=0, le=100)
    Company_Location: str
    Bean_Type: str = ""
    Broad_Bean_Origin: str = ""


class PredictionResponse(BaseModel):
    predicted_rating: float
    message: str
