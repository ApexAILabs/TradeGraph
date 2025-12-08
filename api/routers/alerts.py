from fastapi import APIRouter
from ..models import APIResponse

router = APIRouter()

@router.get("/", response_model=APIResponse)
async def get_alerts():
    return APIResponse(success=True, data={"message": "Alerts endpoint placeholder"}, message="Success")
