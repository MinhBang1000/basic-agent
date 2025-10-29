from langchain_core.tools import tool

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle
import base64, email

@tool("get_weather", description="Get weather in a city")
def get_weather(city: str):
    return f"It is always sunny in {city}."

@tool("add", description="Add two integers.")
def add(a: int, b: int):
    return a+b

# Gmail features
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
def get_gmail_service():
    creds = None
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)

    service = build("gmail", "v1", credentials=creds)
    return service

@tool("read_emails", description="Read latest Gmail messages. Optional query: search string.")
def read_emails(query: str = "", number_of_emails: int = 5) -> str:
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", q=query, maxResults=number_of_emails).execute()
    messages = results.get("messages", [])

    emails = []
    for m in messages:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        snippet = msg.get("snippet", "")
        payload = msg.get("payload", {})
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
        subject = headers.get("Subject", "(no subject)")
        sender = headers.get("From", "(unknown)")
        emails.append(f"From: {sender}\nSubject: {subject}\nSnippet: {snippet}\n")

    if not emails:
        return "No emails found."
    return "\n---\n".join(emails)

@tool("summarize_emails", description="Summarize the latest Gmail messages.")
def summarize_emails(number_of_emails: int = 5) -> str:
    content = read_emails.invoke({
        "query": "",
        "number_of_emails": number_of_emails,
    })
    lines = content.splitlines()
    summary = []
    for line in lines:
        if line.startswith("Subject:") or line.startswith("From:"):
            summary.append(line)
    return "\n".join(summary)

@tool("send_email", description="Send an email via Gmail. Args: to, subject, body")
def send_email(to: str, subject: str, body: str) -> str:
    service = get_gmail_service()
    from email.mime.text import MIMEText
    import base64

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {"raw": encoded}

    service.users().messages().send(userId="me", body=create_message).execute()
    return f"Email sent to {to} with subject '{subject}'."


TOOLS = [get_weather, add, read_emails, summarize_emails, send_email]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}