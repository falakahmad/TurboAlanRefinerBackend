"""
Stripe API Routes
Professional Stripe integration endpoints for subscriptions and payments.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel, EmailStr
from starlette.responses import Response

from app.services.stripe_service import stripe_service
from app.core.logger import get_logger

logger = get_logger('api.stripe')

router = APIRouter(prefix="/stripe", tags=["stripe"])


# --- Request Models ---

class CreateCheckoutRequest(BaseModel):
    price_id: str
    user_id: str
    email: EmailStr
    name: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class CreatePortalRequest(BaseModel):
    user_id: str
    return_url: str


class CancelSubscriptionRequest(BaseModel):
    subscription_id: str
    immediately: bool = False


# --- Endpoints ---

@router.post("/create-checkout-session")
async def create_checkout_session(request: CreateCheckoutRequest):
    """
    Create a Stripe Checkout Session for subscription.
    
    Returns:
        Checkout session with URL for redirect
    """
    if not stripe_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        # Create or get customer
        logger.info(f"Creating checkout session for user {request.user_id}, email: {request.email}")
        customer = stripe_service.create_or_get_customer(
            email=request.email,
            user_id=request.user_id,
            name=request.name
        )
        
        if not customer:
            error_msg = "Failed to create or retrieve customer. Check logs for details."
            logger.error(f"{error_msg} User ID: {request.user_id}, Email: {request.email}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        # Default URLs
        origin = request.success_url or "http://localhost:3000"
        success_url = f"{origin}/dashboard?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = request.cancel_url or f"{origin}/pricing"
        
        # Create checkout session
        session = stripe_service.create_checkout_session(
            customer_id=customer["id"],
            price_id=request.price_id,
            success_url=success_url,
            cancel_url=cancel_url,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        if not session:
            raise HTTPException(
                status_code=500,
                detail="Failed to create checkout session"
            )
        
        logger.info(f"Created checkout session {session['id']} for user {request.user_id}")
        
        return {
            "success": True,
            "session_id": session["id"],
            "url": session["url"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout session: {str(e)}"
        )


@router.post("/create-portal-session")
async def create_portal_session(request: CreatePortalRequest):
    """
    Create a Stripe Customer Portal session.
    
    Allows customers to manage their subscription, payment methods, and billing.
    """
    if not stripe_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        # Get customer record
        customer_record = stripe_service.get_customer_by_user_id(request.user_id)
        
        if not customer_record or not customer_record.get("stripe_customer_id"):
            raise HTTPException(
                status_code=404,
                detail="Customer not found. Please subscribe first."
            )
        
        # Create portal session
        session = stripe_service.create_customer_portal_session(
            customer_id=customer_record["stripe_customer_id"],
            return_url=request.return_url
        )
        
        if not session:
            raise HTTPException(
                status_code=500,
                detail="Failed to create portal session"
            )
        
        logger.info(f"Created portal session for user {request.user_id}")
        
        return {
            "success": True,
            "url": session["url"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating portal session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create portal session: {str(e)}"
        )


@router.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    """
    Get active subscription for a user.
    """
    try:
        subscription = stripe_service.get_active_subscription(user_id)
        
        if not subscription:
            return {
                "success": True,
                "subscription": None
            }
        
        # Get full subscription details from Stripe
        stripe_subscription = stripe_service.get_subscription(
            subscription["subscription_id"]
        )
        
        return {
            "success": True,
            "subscription": {
                **subscription,
                "stripe_details": stripe_subscription
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get subscription: {str(e)}"
        )


@router.post("/cancel-subscription")
async def cancel_subscription(request: CancelSubscriptionRequest):
    """
    Cancel a subscription.
    """
    if not stripe_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured. Please contact support."
        )
    
    try:
        result = stripe_service.cancel_subscription(
            subscription_id=request.subscription_id,
            immediately=request.immediately
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel subscription"
            )
        
        logger.info(f"Cancelled subscription {request.subscription_id}")
        
        return {
            "success": True,
            "subscription": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel subscription: {str(e)}"
        )


@router.get("/payment-history/{user_id}")
async def get_payment_history(user_id: str, limit: int = 10):
    """
    Get payment history for a user.
    """
    try:
        payments = stripe_service.get_payment_history(user_id, limit=limit)
        
        return {
            "success": True,
            "payments": payments
        }
        
    except Exception as e:
        logger.error(f"Error getting payment history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get payment history: {str(e)}"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature")
):
    """
    Handle Stripe webhook events.
    
    This endpoint processes webhook events from Stripe for:
    - Subscription creation/updates/cancellations
    - Payment success/failure
    - Checkout completion
    """
    if not stripe_signature:
        logger.warning("Webhook request missing stripe-signature header")
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")
    
    try:
        # Get raw request body
        body = await request.body()
        
        # Construct and verify webhook event
        event = stripe_service.construct_webhook_event(body, stripe_signature)
        
        if not event:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
        
        # Handle the event
        success = stripe_service.handle_webhook_event(event)
        
        if not success:
            logger.warning(f"Webhook event {event.type} was not handled successfully")
            # Still return 200 to acknowledge receipt
        
        return Response(status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        # Return 400 to indicate we didn't process the event
        raise HTTPException(status_code=400, detail=f"Webhook processing failed: {str(e)}")


