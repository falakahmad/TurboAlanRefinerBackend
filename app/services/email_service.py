"""
Email service for sending OTP and password reset emails via Gmail SMTP.
"""
import smtplib
import os
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via Gmail SMTP."""
    
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('SMTP_FROM_EMAIL', self.smtp_user)
        
        if not self.smtp_user or not self.smtp_password:
            logger.warning("SMTP credentials not configured. Email service will not work.")
    
    def generate_otp(self) -> str:
        """Generate a 6-digit OTP code."""
        return ''.join([str(secrets.randbelow(10)) for _ in range(6)])
    
    def send_otp_email(self, to_email: str, otp_code: str) -> bool:
        """
        Send OTP email to user.
        
        Args:
            to_email: Recipient email address
            otp_code: 6-digit OTP code
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.smtp_user or not self.smtp_password:
            logger.error("SMTP credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = 'Password Reset - Turbo Alan Refiner'
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            # Create HTML content with yellow/golden theme
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
              <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Password Reset - Turbo Alan Refiner</title>
              </head>
              <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
                <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f5f5f5;">
                  <tr>
                    <td align="center" style="padding: 40px 20px;">
                      <table role="presentation" style="max-width: 600px; width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <!-- Header with Golden Gradient -->
                        <tr>
                          <td style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); padding: 40px 30px; text-align: center;">
                            <div style="display: inline-block; width: 60px; height: 60px; background-color: rgba(255, 255, 255, 0.2); border-radius: 50%; margin-bottom: 15px; display: flex; align-items: center; justify-content: center;">
                              <span style="color: #ffffff; font-size: 28px; font-weight: bold;">T</span>
                            </div>
                            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">Turbo Alan Refiner</h1>
                            <p style="color: rgba(255, 255, 255, 0.9); margin: 8px 0 0 0; font-size: 14px; font-weight: 400;">AI-Powered Text Refinement</p>
                          </td>
                        </tr>
                        
                        <!-- Content Section -->
                        <tr>
                          <td style="padding: 40px 30px; background-color: #ffffff;">
                            <h2 style="color: #1f2937; margin: 0 0 16px 0; font-size: 24px; font-weight: 600;">Password Reset Request</h2>
                            <p style="color: #4b5563; font-size: 16px; line-height: 1.6; margin: 0 0 30px 0;">
                              You requested to reset your password. Use the following verification code to proceed:
                            </p>
                            
                            <!-- OTP Code Box -->
                            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #fbbf24; border-radius: 12px; padding: 30px; text-align: center; margin: 30px 0;">
                              <p style="color: #92400e; margin: 0 0 12px 0; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Your Verification Code</p>
                              <div style="background-color: #ffffff; border-radius: 8px; padding: 20px; margin: 15px 0; display: inline-block; min-width: 280px;">
                                <h1 style="color: #f59e0b; font-size: 48px; letter-spacing: 12px; margin: 0; font-weight: 700; font-family: 'Courier New', monospace;">{otp_code}</h1>
                              </div>
                              <p style="color: #92400e; margin: 15px 0 0 0; font-size: 12px; font-weight: 500;">
                                ⏰ Valid for 10 minutes
                              </p>
                            </div>
                            
                            <!-- Security Notice -->
                            <div style="background-color: #fffbeb; border-left: 4px solid #fbbf24; padding: 16px; border-radius: 6px; margin: 30px 0;">
                              <p style="color: #78350f; font-size: 14px; line-height: 1.6; margin: 0;">
                                <strong style="color: #92400e;">🔒 Security Notice:</strong> If you didn't request this password reset, please ignore this email. Your account remains secure.
                              </p>
                            </div>
                            
                            <!-- Instructions -->
                            <div style="margin-top: 30px; padding-top: 30px; border-top: 1px solid #e5e7eb;">
                              <p style="color: #6b7280; font-size: 14px; line-height: 1.6; margin: 0 0 12px 0;">
                                <strong style="color: #374151;">How to use this code:</strong>
                              </p>
                              <ol style="color: #6b7280; font-size: 14px; line-height: 1.8; margin: 0; padding-left: 20px;">
                                <li>Return to the password reset page</li>
                                <li>Enter the 6-digit code shown above</li>
                                <li>Create your new password</li>
                              </ol>
                            </div>
                          </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                          <td style="background-color: #f9fafb; padding: 30px; text-align: center; border-top: 1px solid #e5e7eb;">
                            <p style="color: #9ca3af; font-size: 12px; line-height: 1.6; margin: 0 0 8px 0;">
                              Turbo Alan Refiner - AI-Powered Text Refinement
                            </p>
                            <p style="color: #d1d5db; font-size: 11px; margin: 0;">
                              This is an automated message. Please do not reply to this email.
                            </p>
                            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                              <p style="color: #9ca3af; font-size: 11px; margin: 0;">
                                Need help? Contact our support team or visit our website.
                              </p>
                            </div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                </table>
              </body>
            </html>
            """
            
            # Create plain text version
            text = f"""
Turbo Alan Refiner - Password Reset Request

You requested to reset your password. Use the following verification code to proceed:

Your Verification Code: {otp_code}

This code will expire in 10 minutes.

Security Notice: If you didn't request this password reset, please ignore this email. Your account remains secure.

How to use this code:
1. Return to the password reset page
2. Enter the 6-digit code shown above
3. Create your new password

---
Turbo Alan Refiner - AI-Powered Text Refinement
This is an automated message. Please do not reply to this email.
            """.strip()
            
            # Attach both HTML and plain text versions
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"OTP email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send OTP email: {e}")
            return False


# Global email service instance
email_service = EmailService()
