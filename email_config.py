# Email Configuration for OTP Verification
# Configure these settings according to your email provider

# Gmail Configuration (recommended)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'pjmulat@gmail.com'  # Replace with your actual Gmail address
SMTP_PASSWORD = '0T&NT@66bux'
# For Gmail:
# 1. Enable 2-Factor Authentication
# 2. Generate an App Password: https://myaccount.google.com/apppasswords
# 3. Use the App Password (not your regular Gmail password)

# Alternative SMTP Providers:
# 
# Outlook/Hotmail:
# SMTP_SERVER = 'smtp-mail.outlook.com'
# SMTP_PORT = 587
# 
# Yahoo Mail:
# SMTP_SERVER = 'smtp.mail.yahoo.com'
# SMTP_PORT = 587
# 
# Custom SMTP Server:
# SMTP_SERVER = 'your-smtp-server.com'
# SMTP_PORT = 587  # or 465 for SSL

# Email Configuration
SMTP_SENDER_NAME = "CBK ADMINISTRATOR ACCOUNT"

# Email Templates
OTP_EMAIL_SUBJECT = "CBK System - Email Verification Code"

OTP_EMAIL_TEMPLATE = """
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
        <h2 style="color: #2c5aa0; text-align: center;">CBK Hydroelectric Monitoring System</h2>
        
        <p>Dear User,</p>
        
        <p>Your verification code is:</p>
        
        <div style="text-align: center; margin: 20px 0;">
            <span style="font-size: 32px; font-weight: bold; color: #2c5aa0; background-color: #f0f8ff; padding: 15px 25px; border: 2px solid #2c5aa0; border-radius: 8px; letter-spacing: 3px;">{otp}</span>
        </div>
        
        <p><strong>Important:</strong></p>
        <ul>
            <li>This code will expire in <strong>10 minutes</strong></li>
            <li>Do not share this code with anyone</li>
            <li>If you did not request this code, please ignore this email</li>
        </ul>
        
        <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
        
        <p style="text-align: center; color: #666; font-size: 14px;">
            Best regards,<br>
            <strong>CBK System Administrator</strong><br>
            Hydroelectric Power Plant Monitoring System
        </p>
    </div>
</body>
</html>
"""

WELCOME_EMAIL_TEMPLATE = """
Welcome to CBK Hydroelectric Monitoring System!

Your account has been created with the following details:
- Email: {email}
- Role: {role}
- Access Locations: {locations}

Please log in using your email address and the temporary password provided by your administrator.
You will be required to change your password on first login.

Login URL: {login_url}

Best regards,
CBK System Administrator
"""
