from typing import List, Dict, Any, Optional

from langchain_core.tools import tool

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle
from email.mime.text import MIMEText
from email.utils import getaddresses
import base64, email
import re


# Constraints
DEFAULT_MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 800
EMAIL_SEPARATOR = "\n---\n"

# Scoring & flag rules
URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "please respond", "action required",
    "deadline", "due by", "important", "as soon as possible", "priority"
]

DEADLINE_PATTERNS = [
    re.compile(r"\bdue (by|on)\b[:\s]*(\w+\s+\d{1,2}(?:,?\s*\d{4})?)", re.IGNORECASE),
    re.compile(r"\bdeadline\b[:\s]*(\w+\s+\d{1,2}(?:,?\s*\d{4})?)", re.IGNORECASE),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")  # ISO format YYYY-MM-DD
]

# Helpers

def _safe_extract_headers_from_msg(msg: Dict[str, Any]) -> Dict[str, str]:
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers") or []
    return {h.get("name"): h.get("value") for h in headers if h.get("name")}

def _get_message_id_header(headers: Dict[str, str]) -> Optional[str]:
    return headers.get("Message-ID") or headers.get("Message-Id") or headers.get("MessageID")

def _extract_headers(headers: List[Dict[str, str]]):
    return {h["name"]: h["value"] for h in headers}

def _parse_payload_snippet(payload: Dict[str, Any]) -> str:
    """
    Best-effort extraction of a text snippet from Gmail payload parts.
    """
    parts = payload.get("parts") or []

    def _walk(parts_list):
        for p in parts_list:
            mime = p.get("mimeType", "")
            body = p.get("body", {})
            data = body.get("data")
            if data:
                try:
                    raw = base64.urlsafe_b64decode(data.encode("utf-8"))
                    txt = raw.decode("utf-8", errors="replace")
                    if mime == "text/html":
                        txt = re.sub(r"<[^>]+>", "", txt)
                    return txt
                except Exception:
                    continue
            if p.get("parts"):
                res = _walk(p.get("parts"))
                if res:
                    return res
        return ""

    if parts:
        return _walk(parts)[:MAX_SNIPPET_CHARS]

    # fallback to top-level body
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        try:
            raw = base64.urlsafe_b64decode(data.encode("utf-8"))
            return raw.decode("utf-8", errors="replace")[:MAX_SNIPPET_CHARS]
        except Exception:
            return ""

    return ""

def score_and_flag(subject: str, snippet: str) -> Dict[str, Any]:
    text = f"{subject or ''} {snippet or ''}".lower()
    score = 0
    flags = []

    # Urgent keyword scoring
    for k in URGENT_KEYWORDS:
        if k in text:
            score += 18

    # Deadline date detection
    for pat in DEADLINE_PATTERNS:
        if pat.search(text):
            score += 35
            flags.append("deadline")
            break

    # Short imperative sentence boost (ex: "Send this now")
    if len(text.split()) < 9 and re.search(r"\b(send|reply|confirm|check|review)\b", text):
        score += 20
        flags.append("action_short")

    # Attachment mention bonus
    if "attach" in text or "attached" in text or "attachment" in text:
        score += 10
        flags.append("has_attachment_mention")

    # Cap at 100
    score = min(100, score)
    if score >= 60:
        flags.append("urgent")

    if score >= 30 and "deadline" in flags:
        flags.append("important")

    return {"score": score, "flags": list(set(flags))}

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

@tool("search_emails", description=(
    "Use this tool to RETRIEVE a list of recent emails. "
    "After getting the emails, the agent (you) can then perform tasks like summarization or analysis. "
    "The 'query' argument accepts standard Gmail search queries (e.g., 'from:boss', 'is:unread', 'newer_than:2d')."
))
def search_emails(query: str = "", number_of_emails: int = DEFAULT_MAX_RESULTS, label_ids: Optional[List[str]] = None):
    """
    Returns a single string containing up to `max_results` emails, each formatted as:
      From: <sender>
      Subject: <subject>
      Date: <date>
      Snippet: <snippet>
    separated by a line '---'.

    On error, returns a string that starts with '[ToolError] ' followed by the error message.
    """
    try:
        service = get_gmail_service()
        params = {
            "userId": "me",
            "q": query,
            "maxResults": number_of_emails
        }
        if label_ids:
            params["labelIds"] = label_ids

        resp = service.users().messages().list(**params).execute()
        msgs = resp.get("messages", [])
        if not msgs:
            return "No emails found."

        pieces = []
        for m in msgs:
            mid = m.get("id")
            msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
            payload = msg.get("payload", {})
            headers = _extract_headers(payload.get("headers", []))
            snippet = msg.get("snippet", "") or _parse_payload_snippet(payload) or ""
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS]+"..."
            sender = headers.get("From", "(unknown)")
            subject = headers.get("Subject", "(no subject)")
            date = headers.get("Date", "(no date)")
            has_attach = any(p.get("filename") for p in payload.get("parts") or [])
            info = score_and_flag(subject, snippet)
            piece = (
                f"ID: {mid}\n"
                f"From: {sender}\n"
                f"Subject: {subject}\n"
                f"Date: {date}\n"
                f"Snippet: {snippet[:MAX_SNIPPET_CHARS]}\n"
                f"Has Attachment: {has_attach}\n"
                f"Importance Score: {info['score']}/100\n"
                f"Flags: {', '.join(info['flags']) if info['flags'] else 'None'}"
            )
            pieces.append(piece)
        result = EMAIL_SEPARATOR.join(pieces)
        print("Search tool: \n", result)
        return result

    except Exception as e:
        return f"[ToolError] {type(e).__name__}: {e}"

@tool("send_email", description="Send an email via Gmail. Args: to, subject, body")
def send_email(to: str, subject: str, body: str) -> str:
    try:
        service = get_gmail_service()

        # Build MIME email
        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to
        message["subject"] = subject

        # Encode message
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded}

        # Send via Gmail API
        sent = service.users().messages().send(userId="me", body=create_message).execute()

        # Return simple string for agent
        return f"✅ Sent to {to} | subject: '{subject}' | id: {sent.get('id')}"

    except Exception as e:
        return f"[ToolError] send_email failed: {type(e).__name__}: {str(e)}"

@tool("reply_email", description="Reply to one email by message_id. Args: message_id, body, dry_run=True")
def reply_email(message_id: str, body: str) -> str:
    try:
        service = get_gmail_service()
        # fetch full to get headers and threadId
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = _safe_extract_headers_from_msg(msg)

        to_email = headers.get("Reply-To") or headers.get("From")
        if not to_email:
            return "[ToolError] Cannot determine reply recipient (no From/Reply-To)."

        subject = headers.get("Subject", "")
        if not subject.lower().startswith("re:"):
            subject = "Re: " + subject

        mime = MIMEText(body, "plain", "utf-8")
        mime["To"] = to_email
        mime["Subject"] = subject

        # Use actual Message-ID header for threading
        orig_msgid = _get_message_id_header(headers)
        if orig_msgid:
            mime["In-Reply-To"] = orig_msgid
            refs = headers.get("References", "")
            mime["References"] = (refs + " " + orig_msgid).strip() if refs else orig_msgid

        print(f"[reply_email DEBUG] Trying to fetch Gmail message_id={message_id!r}")
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        thread_id = msg.get("threadId")
        sent = service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id} ).execute()
        print(f"OK|replied_id:{sent.get('id')}")
        return f"OK|replied_id:{sent.get('id')}"

    except Exception as e:
        print(e)
        return f"[ToolError] reply_email failed: {type(e).__name__}: {e}"


from email.utils import getaddresses

@tool("reply_all_email", description="Reply to all recipients of an email. Args: message_id, body, my_email, dry_run=True")
def reply_all_email(message_id: str, body: str, my_email: str = "", dry_run: bool = True) -> str:
    try:
        service = get_gmail_service()
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = _safe_extract_headers_from_msg(msg)

        # gather addresses robustly
        raw_from = headers.get("Reply-To") or headers.get("From") or ""
        raw_to = headers.get("To", "")
        raw_cc = headers.get("Cc", "")

        all_pairs = getaddresses([raw_from, raw_to, raw_cc])  # returns list of (name, email)
        all_emails = [email_addr for (_, email_addr) in all_pairs if email_addr]

        # remove our own email (if provided) and dedupe preserve order
        seen = set()
        recipients = []
        my_email_l = (my_email or "").lower()
        for e in all_emails:
            el = e.lower()
            if my_email_l and my_email_l == el:
                continue
            if el in seen:
                continue
            seen.add(el)
            recipients.append(e)

        if not recipients:
            return "[ToolError] No recipients to reply-all to after filtering."

        to_field = ", ".join(recipients)

        subject = headers.get("Subject", "")
        if not subject.lower().startswith("re:"):
            subject = "Re: " + subject

        mime = MIMEText(body, "plain", "utf-8")
        mime["To"] = to_field
        mime["Subject"] = subject

        orig_msgid = _get_message_id_header(headers)
        if orig_msgid:
            mime["In-Reply-To"] = orig_msgid
            refs = headers.get("References", "")
            mime["References"] = (refs + " " + orig_msgid).strip() if refs else orig_msgid

        preview = f"[DryRun ReplyAll] To: {to_field}\nSubject: {subject}\n\n{body[:2000]}"
        if dry_run:
            return preview

        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        thread_id = msg.get("threadId")
        sent = service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id}).execute()

        return f"OK|replied_all_id:{sent.get('id')}"

    except Exception as e:
        return f"[ToolError] reply_all_email failed: {type(e).__name__}: {e}"



TOOLS = [search_emails, send_email, reply_email, reply_all_email]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}