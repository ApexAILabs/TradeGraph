from fastapi import APIRouter
from ..models import APIResponse

router = APIRouter()

@router.get("/", response_model=APIResponse)
async def get_portfolio():
    return APIResponse(success=True, data={"message": "Portfolio endpoint placeholder"}, message="Success")
