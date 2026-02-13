"""
Utility Functions
Common helper functions used across the pipeline
"""

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


logger = logging.getLogger(__name__)


def send_email_alert(subject, body, recipients, smtp_config=None):
    """
    Send email alert
    
    Args:
        subject (str): Email subject
        body (str): Email body
        recipients (list): List of recipient email addresses
        smtp_config (dict): SMTP configuration
    """
    if not smtp_config:
        smtp_config = {
            'host': 'smtp.gmail.com',
            'port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password'
        }
    
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['username']
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
        server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent to {recipients}")
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")


def send_slack_alert(message, webhook_url):
    """
    Send Slack alert
    
    Args:
        message (str): Message to send
        webhook_url (str): Slack webhook URL
    """
    try:
        payload = {'text': message}
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            logger.info("Slack alert sent successfully")
        else:
            logger.error(f"Failed to send Slack alert: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {str(e)}")


def format_anomaly_alert(anomalies):
    """
    Format anomalies into alert message
    
    Args:
        anomalies (list): List of detected anomalies
        
    Returns:
        str: Formatted alert message
    """
    if not anomalies:
        return "No anomalies detected"
    
    message = f"ALERT: Data Pipeline - {len(anomalies)} Anomalies Detected\n\n"
    
    for i, anomaly in enumerate(anomalies, 1):
        message += f"{i}. [{anomaly['severity'].upper()}] {anomaly['type']}\n"
        message += f"   {anomaly['message']}\n\n"
    
    return message
