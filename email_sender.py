import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def send_digest_email(html_content: str, recipient_email: str = None) -> bool:
    """
    Sends an HTML digest email using Gmail SMTP.

    Args:
        html_content: The HTML newsletter body to send.
        recipient_email: The target recipient. Defaults to GMAIL_ADDRESS (self).

    Returns:
        True if sent successfully, False otherwise.
    """
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_address or not gmail_app_password:
        print("[Email] ERROR: GMAIL_ADDRESS or GMAIL_APP_PASSWORD is not set in environment.")
        return False

    if recipient_email is None:
        recipient_email = gmail_address

    # Format today's date (e.g. "May 29, 2026")
    # Using the current system date
    today_str = datetime.now().strftime("%B %d, %Y")

    # Create message container - "alternative" is standard for HTML mails
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🤖 AI Daily Digest — {today_str}"
    msg["From"] = f"AI Daily Digest <{gmail_address}>"
    msg["To"] = recipient_email

    # Record the HTML MIME body
    html_part = MIMEText(html_content, "html")
    msg.attach(html_part)

    try:
        print(f"[Email] Connecting to Gmail SMTP server (smtp.gmail.com:465) via SSL...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_app_password)
            print(f"[Email] Sending email to {recipient_email}...")
            server.sendmail(gmail_address, recipient_email, msg.as_string())
        print("[Email] ✓ Success: Email sent successfully.")
        return True
    except Exception as e:
        print(f"[Email] ✗ Failure: Failed to send email. Error: {e}")
        return False

if __name__ == "__main__":
    # Main block: send a simple test email
    print("=" * 60)
    print("  AI Daily Digest — Testing Email Delivery")
    print("=" * 60)
    
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Email Test</title>
    </head>
    <body>
      <h1>Test Digest</h1>
      <p>If you see this, email delivery works!</p>
    </body>
    </html>
    """
    send_digest_email(test_html)
