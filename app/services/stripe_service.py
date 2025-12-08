"""
Stripe Service
Professional Stripe integration with comprehensive error handling and type safety.
Handles subscriptions, payments, customers, and webhook events.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import stripe
from stripe.error import StripeError

from app.core.mongodb_db import db
from app.core.logger import get_logger
from app.services.email_service import email_service

logger = get_logger('services.stripe')

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

if not stripe.api_key:
    logger.warning("STRIPE_SECRET_KEY not configured. Stripe features will be disabled.")


class StripeService:
    """
    Professional Stripe service for handling payments, subscriptions, and customers.
    """
    
    @staticmethod
    def is_configured() -> bool:
        """Check if Stripe is properly configured."""
        api_key = stripe.api_key
        if not api_key:
            return False
        # Basic validation - Stripe keys start with sk_ (secret) or pk_ (public)
        if isinstance(api_key, str) and (api_key.startswith("sk_") or api_key.startswith("pk_")):
            return True
        logger.warning(f"Stripe API key format appears invalid (should start with sk_ or pk_)")
        return False
    
    @staticmethod
    def get_stripe_client() -> Optional[stripe.Stripe]:
        """Get Stripe client instance."""
        if not StripeService.is_configured():
            return None
        return stripe
    
    # --- Customer Management ---
    
    @staticmethod
    def create_or_get_customer(email: str, user_id: str, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create or retrieve a Stripe customer.
        
        Args:
            email: Customer email address
            user_id: Internal user ID
            name: Customer name (optional)
            
        Returns:
            Customer object or None if failed
        """
        if not StripeService.is_configured():
            logger.error("Stripe not configured - STRIPE_SECRET_KEY is missing or invalid")
            return None
        
        try:
            # Check MongoDB connection
            if not db.db:
                logger.error("MongoDB database connection is not available")
                # Continue anyway - we can still create Stripe customer without storing in DB
            
            # Check if customer already exists in our database
            customer_record = None
            try:
                customer_record = StripeService.get_customer_by_user_id(user_id)
            except Exception as e:
                logger.warning(f"Failed to check existing customer in database: {e}")
            
            if customer_record and customer_record.get("stripe_customer_id"):
                # Retrieve existing customer from Stripe
                try:
                    customer = stripe.Customer.retrieve(customer_record["stripe_customer_id"])
                    logger.info(f"Retrieved existing Stripe customer {customer.id} for user {user_id}")
                    return {
                        "id": customer.id,
                        "email": customer.email,
                        "name": customer.name,
                        "created": customer.created,
                        "metadata": customer.metadata
                    }
                except StripeError as e:
                    logger.warning(f"Failed to retrieve existing Stripe customer {customer_record['stripe_customer_id']}: {e}")
                    # Continue to create new customer
            
            # Create new customer in Stripe
            logger.info(f"Creating new Stripe customer for user {user_id}, email: {email}")
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    "user_id": user_id,
                    "created_by": "alan_refiner"
                }
            )
            
            logger.info(f"Successfully created Stripe customer {customer.id}")
            
            # Store customer in MongoDB (non-blocking - continue even if this fails)
            try:
                stored = StripeService._store_customer(
                    user_id=user_id,
                    stripe_customer_id=customer.id,
                    email=email,
                    name=name
                )
                if not stored:
                    logger.warning(f"Failed to store customer in MongoDB, but Stripe customer {customer.id} was created")
            except Exception as e:
                logger.warning(f"Exception while storing customer in MongoDB: {e}")
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            
            return {
                "id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "created": customer.created,
                "metadata": customer.metadata
            }
            
        except StripeError as e:
            error_msg = f"Stripe API error creating customer: {str(e)}"
            if hasattr(e, 'user_message'):
                error_msg += f" - {e.user_message}"
            logger.error(error_msg, exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating customer: {str(e)}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def get_customer_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get customer record from MongoDB by user ID."""
        if not db.db:
            logger.debug("MongoDB database not available for customer lookup")
            return None
        
        try:
            collection = db.db.customers
            customer = collection.find_one({"user_id": user_id})
            if customer and "_id" in customer:
                customer["_id"] = str(customer["_id"])
            return customer
        except Exception as e:
            logger.warning(f"Failed to get customer from database (non-critical): {e}")
            return None
    
    @staticmethod
    def _store_customer(user_id: str, stripe_customer_id: str, email: str, name: Optional[str] = None) -> bool:
        """Store customer record in MongoDB."""
        if not db.db:
            return False
        
        try:
            collection = db.db.customers
            collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "stripe_customer_id": stripe_customer_id,
                        "email": email,
                        "name": name,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store customer: {e}")
            return False
    
    # --- Subscription Management ---
    
    @staticmethod
    def create_checkout_session(
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a Stripe Checkout Session for subscription.
        
        Args:
            customer_id: Stripe customer ID
            price_id: Stripe price ID
            success_url: URL to redirect after successful payment
            cancel_url: URL to redirect after cancellation
            user_id: Internal user ID (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Checkout session object or None if failed
        """
        if not StripeService.is_configured():
            logger.error("Stripe not configured")
            return None
        
        try:
            session_metadata = metadata or {}
            if user_id:
                session_metadata["user_id"] = user_id
            
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata=session_metadata,
                subscription_data={
                    "metadata": session_metadata
                },
                allow_promotion_codes=True
            )
            
            logger.info(f"Created checkout session {session.id} for customer {customer_id}")
            
            return {
                "id": session.id,
                "url": session.url,
                "customer": session.customer,
                "subscription": session.subscription
            }
            
        except StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating checkout session: {e}")
            return None
    
    @staticmethod
    def create_customer_portal_session(
        customer_id: str,
        return_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a Stripe Customer Portal session.
        
        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session
            
        Returns:
            Portal session object or None if failed
        """
        if not StripeService.is_configured():
            logger.error("Stripe not configured")
            return None
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            
            logger.info(f"Created customer portal session for customer {customer_id}")
            
            return {
                "id": session.id,
                "url": session.url
            }
            
        except StripeError as e:
            logger.error(f"Stripe error creating portal session: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating portal session: {e}")
            return None
    
    @staticmethod
    def get_subscription(subscription_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a subscription from Stripe."""
        if not StripeService.is_configured():
            return None
        
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return {
                "id": subscription.id,
                "customer": subscription.customer,
                "status": subscription.status,
                "current_period_start": subscription.current_period_start,
                "current_period_end": subscription.current_period_end,
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "items": [
                    {
                        "id": item.id,
                        "price": {
                            "id": item.price.id,
                            "unit_amount": item.price.unit_amount,
                            "currency": item.price.currency,
                            "recurring": {
                                "interval": item.price.recurring.interval if item.price.recurring else None
                            }
                        }
                    }
                    for item in subscription.items.data
                ],
                "metadata": subscription.metadata
            }
        except StripeError as e:
            logger.error(f"Stripe error retrieving subscription: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving subscription: {e}")
            return None
    
    @staticmethod
    def cancel_subscription(subscription_id: str, immediately: bool = False) -> Optional[Dict[str, Any]]:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            immediately: If True, cancel immediately; if False, cancel at period end
            
        Returns:
            Updated subscription object or None if failed
        """
        if not StripeService.is_configured():
            return None
        
        try:
            if immediately:
                subscription = stripe.Subscription.delete(subscription_id)
            else:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            
            # Update subscription in database
            StripeService._update_subscription_status(
                subscription_id=subscription.id,
                status=subscription.status,
                cancel_at_period_end=subscription.cancel_at_period_end
            )
            
            logger.info(f"Subscription {subscription_id} cancelled (immediately={immediately})")
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "cancel_at_period_end": subscription.cancel_at_period_end
            }
            
        except StripeError as e:
            logger.error(f"Stripe error cancelling subscription: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error cancelling subscription: {e}")
            return None
    
    @staticmethod
    def get_active_subscription(user_id: str) -> Optional[Dict[str, Any]]:
        """Get active subscription for a user."""
        if not db.db:
            return None
        
        try:
            collection = db.db.subscriptions
            subscription = collection.find_one({
                "user_id": user_id,
                "status": {"$in": ["active", "trialing", "past_due"]}
            }, sort=[("created_at", -1)])
            
            if subscription and "_id" in subscription:
                subscription["_id"] = str(subscription["_id"])
            
            return subscription
        except Exception as e:
            logger.error(f"Failed to get active subscription: {e}")
            return None
    
    @staticmethod
    def _store_subscription(
        subscription_id: str,
        user_id: str,
        customer_id: str,
        status: str,
        price_id: str,
        current_period_start: int,
        current_period_end: int,
        cancel_at_period_end: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store subscription in MongoDB."""
        if not db.db:
            return False
        
        try:
            collection = db.db.subscriptions
            collection.update_one(
                {"subscription_id": subscription_id},
                {
                    "$set": {
                        "user_id": user_id,
                        "customer_id": customer_id,
                        "status": status,
                        "price_id": price_id,
                        "current_period_start": datetime.fromtimestamp(current_period_start),
                        "current_period_end": datetime.fromtimestamp(current_period_end),
                        "cancel_at_period_end": cancel_at_period_end,
                        "metadata": metadata or {},
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "subscription_id": subscription_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store subscription: {e}")
            return False
    
    @staticmethod
    def _update_subscription_status(
        subscription_id: str,
        status: str,
        cancel_at_period_end: Optional[bool] = None
    ) -> bool:
        """Update subscription status in MongoDB and user plan."""
        if not db.db:
            return False
        
        try:
            collection = db.db.subscriptions
            subscription = collection.find_one({"subscription_id": subscription_id})
            
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found")
                return False
            
            user_id = subscription.get("user_id")
            price_id = subscription.get("price_id")
            
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            if cancel_at_period_end is not None:
                update_data["cancel_at_period_end"] = cancel_at_period_end
            
            collection.update_one(
                {"subscription_id": subscription_id},
                {"$set": update_data}
            )
            
            # Update user plan/status
            if user_id:
                StripeService._update_user_plan(user_id, status, price_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update subscription status: {e}")
            return False
    
    @staticmethod
    def _update_user_plan(user_id: str, subscription_status: str, price_id: Optional[str] = None, plan_name_from_metadata: Optional[str] = None) -> bool:
        """
        Update user plan and subscription status in users collection.
        
        Args:
            user_id: Internal user ID
            subscription_status: Stripe subscription status
            price_id: Stripe price ID (optional, to determine plan name)
            plan_name_from_metadata: Plan name from metadata (optional, takes priority)
        """
        if not db.db:
            return False
        
        try:
            # Determine plan name - prioritize metadata, then price_id
            plan_name = None
            
            # First, use plan name from metadata if provided
            if plan_name_from_metadata:
                plan_name = plan_name_from_metadata.lower()
            elif price_id:
                # Map price_id to plan name by checking if it contains plan keywords
                # This works with any Stripe price ID format
                price_id_lower = price_id.lower()
                if "starter" in price_id_lower:
                    plan_name = "starter"
                elif "professional" in price_id_lower or "pro" in price_id_lower:
                    plan_name = "pro"
                elif "enterprise" in price_id_lower:
                    plan_name = "enterprise"
                else:
                    # Default to pro if we can't determine
                    plan_name = "pro"
            
            # Determine if user has active subscription
            is_paid = subscription_status in ["active", "trialing"]
            
            # Update user document
            users_collection = db.db.users
            update_data = {
                "subscription_status": subscription_status,
                "is_paid": is_paid,
                "updated_at": datetime.utcnow()
            }
            
            if plan_name:
                update_data["plan"] = plan_name
            
            users_collection.update_one(
                {"id": user_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated user {user_id} plan: {plan_name}, status: {subscription_status}, paid: {is_paid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user plan: {e}")
            return False
    
    # --- Payment History ---
    
    @staticmethod
    def get_payment_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get payment history for a user."""
        if not db.db:
            return []
        
        try:
            collection = db.db.payments
            payments = list(
                collection.find({"user_id": user_id})
                .sort("created_at", -1)
                .limit(limit)
            )
            
            for payment in payments:
                if "_id" in payment:
                    payment["_id"] = str(payment["_id"])
            
            return payments
        except Exception as e:
            logger.error(f"Failed to get payment history: {e}")
            return []
    
    @staticmethod
    def _store_payment(
        payment_intent_id: str,
        user_id: str,
        customer_id: str,
        amount: int,
        currency: str,
        status: str,
        subscription_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store payment record in MongoDB."""
        if not db.db:
            return False
        
        try:
            collection = db.db.payments
            collection.insert_one({
                "payment_intent_id": payment_intent_id,
                "user_id": user_id,
                "customer_id": customer_id,
                "subscription_id": subscription_id,
                "amount": amount,
                "currency": currency,
                "status": status,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to store payment: {e}")
            return False
    
    # --- Webhook Event Handling ---
    
    @staticmethod
    def construct_webhook_event(payload: bytes, signature: str) -> Optional[stripe.Event]:
        """
        Construct and verify a webhook event.
        
        Args:
            payload: Raw request body
            signature: Stripe signature header
            
        Returns:
            Stripe event object or None if verification failed
        """
        if not StripeService.is_configured() or not STRIPE_WEBHOOK_SECRET:
            logger.error("Stripe webhook secret not configured")
            return None
        
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                STRIPE_WEBHOOK_SECRET
            )
            return event
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            return None
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            return None
    
    @staticmethod
    def handle_webhook_event(event: stripe.Event) -> bool:
        """
        Handle a Stripe webhook event.
        
        Args:
            event: Stripe event object
            
        Returns:
            True if handled successfully, False otherwise
        """
        try:
            event_type = event.type
            logger.info(f"Handling Stripe webhook event: {event_type}")
            
            if event_type == "checkout.session.completed":
                return StripeService._handle_checkout_session_completed(event)
            elif event_type == "customer.subscription.created":
                return StripeService._handle_subscription_created(event)
            elif event_type == "customer.subscription.updated":
                return StripeService._handle_subscription_updated(event)
            elif event_type == "customer.subscription.deleted":
                return StripeService._handle_subscription_deleted(event)
            elif event_type == "invoice.payment_succeeded":
                return StripeService._handle_invoice_payment_succeeded(event)
            elif event_type == "invoice.payment_failed":
                return StripeService._handle_invoice_payment_failed(event)
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return True  # Return True for unhandled events to acknowledge receipt
            
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return False
    
    @staticmethod
    def _handle_checkout_session_completed(event: stripe.Event) -> bool:
        """Handle checkout.session.completed event."""
        try:
            session = event.data.object
            
            customer_id = session.customer
            subscription_id = session.subscription
            user_id = session.metadata.get("user_id") if session.metadata else None
            
            if not user_id and customer_id:
                # Try to get user_id from customer record
                customer_record = db.db.customers.find_one({"stripe_customer_id": customer_id}) if db.db else None
                if customer_record:
                    user_id = customer_record.get("user_id")
            
            if subscription_id and user_id:
                # Retrieve subscription details
                subscription = StripeService.get_subscription(subscription_id)
                if subscription:
                    # Store subscription
                    price_id = subscription["items"][0]["price"]["id"] if subscription["items"] else None
                    plan_name = session.metadata.get("plan") if session.metadata else None
                    
                    StripeService._store_subscription(
                        subscription_id=subscription_id,
                        user_id=user_id,
                        customer_id=customer_id,
                        status=subscription["status"],
                        price_id=price_id or "",
                        current_period_start=subscription["current_period_start"],
                        current_period_end=subscription["current_period_end"],
                        cancel_at_period_end=subscription["cancel_at_period_end"],
                        metadata={
                            **(subscription.get("metadata") or {}),
                            "plan": plan_name
                        }
                    )
                    
                    # Update user plan/status - pass plan name from metadata
                    StripeService._update_user_plan(user_id, subscription["status"], price_id or "", plan_name)
                    
                    # Send payment confirmation email
                    customer_email = session.customer_details.email if session.customer_details else session.customer_email
                    customer_name = session.customer_details.name if session.customer_details else None
                    if customer_email and subscription.get("items"):
                        try:
                            items = subscription.get("items", [])
                            if items and len(items) > 0:
                                price_info = items[0].get("price", {})
                                amount = price_info.get("unit_amount", 0) / 100  # Convert cents to dollars
                                currency = price_info.get("currency", "usd")
                                email_service.send_payment_confirmation_email(
                                    to_email=customer_email,
                                    plan_name=plan_name or "Pro",
                                    amount=amount,
                                    currency=currency,
                                    customer_name=customer_name
                                )
                        except Exception as e:
                            logger.error(f"Failed to send payment confirmation email: {e}")
            
            logger.info(f"Processed checkout.session.completed for session {session.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling checkout.session.completed: {e}")
            return False
    
    @staticmethod
    def _handle_subscription_created(event: stripe.Event) -> bool:
        """Handle customer.subscription.created event."""
        try:
            subscription = event.data.object
            
            user_id = subscription.metadata.get("user_id") if subscription.metadata else None
            if not user_id:
                # Try to get from customer
                customer_record = db.db.customers.find_one({"stripe_customer_id": subscription.customer}) if db.db else None
                if customer_record:
                    user_id = customer_record.get("user_id")
            
            if user_id:
                price_id = subscription.items.data[0].price.id if subscription.items.data else None
                plan_name = subscription.metadata.get("plan") if subscription.metadata else None
                
                StripeService._store_subscription(
                    subscription_id=subscription.id,
                    user_id=user_id,
                    customer_id=subscription.customer,
                    status=subscription.status,
                    price_id=price_id or "",
                    current_period_start=subscription.current_period_start,
                    current_period_end=subscription.current_period_end,
                    cancel_at_period_end=subscription.cancel_at_period_end,
                    metadata={
                        **(subscription.metadata or {}),
                        "plan": plan_name
                    }
                )
                
                # Update user plan/status - pass plan name from metadata
                StripeService._update_user_plan(user_id, subscription.status, price_id or "", plan_name)
            
            logger.info(f"Processed subscription.created for {subscription.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling subscription.created: {e}")
            return False
    
    @staticmethod
    def _handle_subscription_updated(event: stripe.Event) -> bool:
        """Handle customer.subscription.updated event."""
        try:
            subscription = event.data.object
            
            StripeService._update_subscription_status(
                subscription_id=subscription.id,
                status=subscription.status,
                cancel_at_period_end=subscription.cancel_at_period_end
            )
            
            logger.info(f"Processed subscription.updated for {subscription.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling subscription.updated: {e}")
            return False
    
    @staticmethod
    def _handle_subscription_deleted(event: stripe.Event) -> bool:
        """Handle customer.subscription.deleted event."""
        try:
            subscription = event.data.object
            
            StripeService._update_subscription_status(
                subscription_id=subscription.id,
                status="cancelled"
            )
            
            logger.info(f"Processed subscription.deleted for {subscription.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling subscription.deleted: {e}")
            return False
    
    @staticmethod
    def _handle_invoice_payment_succeeded(event: stripe.Event) -> bool:
        """Handle invoice.payment_succeeded event."""
        try:
            invoice = event.data.object
            
            customer_id = invoice.customer
            subscription_id = invoice.subscription
            amount = invoice.amount_paid
            currency = invoice.currency
            
            # Get user_id from customer
            customer_record = db.db.customers.find_one({"stripe_customer_id": customer_id}) if db.db else None
            user_id = customer_record.get("user_id") if customer_record else None
            
            if user_id:
                StripeService._store_payment(
                    payment_intent_id=invoice.payment_intent or "",
                    user_id=user_id,
                    customer_id=customer_id,
                    amount=amount,
                    currency=currency,
                    status="succeeded",
                    subscription_id=subscription_id,
                    metadata={"invoice_id": invoice.id}
                )
                
                # Send payment confirmation email for recurring payments
                # (Skip if this is the first payment, as checkout.session.completed already sent it)
                if subscription_id:
                    # Check if this is a recurring payment (not the first one)
                    subscription_record = db.db.subscriptions.find_one({"subscription_id": subscription_id}) if db.db else None
                    if subscription_record:
                        # Get customer email and plan
                        customer_email = customer_record.get("email") if customer_record else None
                        customer_name = customer_record.get("name") if customer_record else None
                        plan_name = subscription_record.get("metadata", {}).get("plan", "Pro") if subscription_record else "Pro"
                        
                        if customer_email:
                            try:
                                amount_dollars = amount / 100  # Convert cents to dollars
                                email_service.send_payment_confirmation_email(
                                    to_email=customer_email,
                                    plan_name=plan_name,
                                    amount=amount_dollars,
                                    currency=currency,
                                    customer_name=customer_name
                                )
                            except Exception as e:
                                logger.error(f"Failed to send payment confirmation email: {e}")
            
            logger.info(f"Processed invoice.payment_succeeded for invoice {invoice.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling invoice.payment_succeeded: {e}")
            return False
    
    @staticmethod
    def _handle_invoice_payment_failed(event: stripe.Event) -> bool:
        """Handle invoice.payment_failed event."""
        try:
            invoice = event.data.object
            
            customer_id = invoice.customer
            subscription_id = invoice.subscription
            
            # Get user_id from customer
            customer_record = db.db.customers.find_one({"stripe_customer_id": customer_id}) if db.db else None
            user_id = customer_record.get("user_id") if customer_record else None
            
            if user_id:
                # Update subscription status if needed
                if subscription_id:
                    StripeService._update_subscription_status(
                        subscription_id=subscription_id,
                        status="past_due"
                    )
            
            logger.info(f"Processed invoice.payment_failed for invoice {invoice.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling invoice.payment_failed: {e}")
            return False


# Global instance
stripe_service = StripeService()

