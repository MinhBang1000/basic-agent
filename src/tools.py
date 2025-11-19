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

import json
@tool(
    "is_reply_or_reply_all",
    description=(
        "Given a Gmail message_id, determine whether the correct reply action is "
        "'reply' or 'reply_all'. Returns a JSON object: "
        "{ 'should_reply_all': bool, 'from': [...], 'to': [...], 'cc': [...], 'all_recipients': [...] }."
    )
)
def is_reply_or_reply_all(message_id: str) -> str:
    """
    Determine whether a message should use reply or reply-all behavior.
    Also returns extracted recipients for LLM to display and confirm with the user.
    """
    try:
        service = get_gmail_service()

        # Fetch the email
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        headers = _safe_extract_headers_from_msg(msg)

        raw_from = headers.get("Reply-To") or headers.get("From") or ""
        raw_to = headers.get("To", "")
        raw_cc = headers.get("Cc", "")

        from_list = [e for _, e in getaddresses([raw_from])]
        to_list = [e for _, e in getaddresses([raw_to])]
        cc_list = [e for _, e in getaddresses([raw_cc])]

        # Rule: If CC exists → should reply all
        should_reply_all = len(cc_list) > 0 or len(to_list) > 1

        result = {
            "should_reply_all": should_reply_all,
            "from": from_list,
            "to": to_list,
            "cc": cc_list,
            "all_recipients": list({*from_list, *to_list, *cc_list}),
            "subject": headers.get("Subject", "(no subject)")
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return f"[ToolError] is_reply_or_reply_all failed: {type(e).__name__}: {e}"

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

        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        thread_id = msg.get("threadId")
        sent = service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id} ).execute()
        print(f"OK|replied_id:{sent.get('id')}")
        return f"OK|replied_id:{sent.get('id')}"

    except Exception as e:
        print(e)
        return f"[ToolError] reply_email failed: {type(e).__name__}: {e}"

@tool(
    "reply_all_email",
    description="Reply to all recipients of an email. Args: message_id, body, my_email"
)
def reply_all_email(message_id: str, body: str, my_email: str = "") -> str:
    """
    Reply-all to an existing Gmail message while preserving threading and recipients.

    Args:
        message_id: Gmail 'id' of the message to reply to (NOT the Message-ID header).
        body:       Plain-text body of the reply.
        my_email:   Your own email address (to avoid replying to yourself).
                    If omitted, we try to guess it from the original To header.

    Returns:
        - "OK|replied_all_id:<id>" on success
        - "[ToolError] <reason>" on failure
    """
    try:
        service = get_gmail_service()

        # 1) Fetch original message (headers + threadId)
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        headers = _safe_extract_headers_from_msg(msg)

        # 2) Raw header strings
        raw_from = headers.get("Reply-To") or headers.get("From") or ""
        raw_to = headers.get("To", "")
        raw_cc = headers.get("Cc", "")

        # 3) Parse into (name, email) pairs
        from_addrs = [e for _, e in getaddresses([raw_from])]
        to_addrs   = [e for _, e in getaddresses([raw_to])]
        cc_addrs   = [e for _, e in getaddresses([raw_cc])]

        # 4) If my_email not provided, guess from the To list (heuristic)
        if not my_email and to_addrs:
            my_email = to_addrs[0]
        my_email_l = (my_email or "").lower()

        def _sanitize_email(addr: str) -> str:
            """
            Basic cleanup: strip spaces and remove CR/LF to avoid header injection
            or invalid formatting that Gmail might reject.
            """
            if not addr:
                return ""
            addr = addr.strip()
            addr = addr.replace("\r", "").replace("\n", "")
            return addr

        def _is_probably_valid_email(addr: str) -> bool:
            """
            Very simple validation: contains exactly one '@' and some dot after it.
            This is not full RFC validation, but good enough to avoid obvious errors.
            """
            if "@" not in addr:
                return False
            local, _, domain = addr.partition("@")
            if not local or not domain:
                return False
            if "." not in domain:
                return False
            return True

        def _clean_recipients(addresses):
            """
            - Sanitize addresses (strip, remove newlines).
            - Drop obviously invalid ones.
            - Remove our own email.
            - Remove duplicates while preserving order.
            """
            seen = set()
            result = []
            for addr in addresses:
                addr = _sanitize_email(addr)
                if not addr:
                    continue
                addr_l = addr.lower()
                # Skip ourselves
                if my_email_l and addr_l == my_email_l:
                    continue
                # Simple validity check
                if not _is_probably_valid_email(addr):
                    continue
                if addr_l in seen:
                    continue
                seen.add(addr_l)
                result.append(addr)
            return result

        # 5) Clean each group
        clean_from = _clean_recipients(from_addrs)
        clean_to   = _clean_recipients(to_addrs)
        clean_cc   = _clean_recipients(cc_addrs)

        # 6) Build final To and Cc (Gmail-like reply-all behavior)
        #    To: original sender + original To
        #    Cc: original Cc
        to_recipients = _clean_recipients(clean_from + clean_to)
        cc_recipients = clean_cc

        if not to_recipients and not cc_recipients:
            return "[ToolError] No valid recipients for reply-all after filtering."

        # If somehow To is empty but Cc has addresses, promote one Cc into To
        if not to_recipients and cc_recipients:
            to_recipients = [cc_recipients[0]]
            cc_recipients = cc_recipients[1:]

        # 7) Subject: ensure "Re:" prefix
        subject = headers.get("Subject", "") or ""
        if not subject.lower().startswith("re:"):
            subject = "Re: " + subject

        # 8) Build MIME message
        mime = MIMEText(body, "plain", "utf-8")
        mime["Subject"] = subject
        if to_recipients:
            mime["To"] = ", ".join(to_recipients)
        if cc_recipients:
            mime["Cc"] = ", ".join(cc_recipients)

        # 9) Threading headers for proper conversation grouping
        orig_msgid = _get_message_id_header(headers)
        if orig_msgid:
            mime["In-Reply-To"] = orig_msgid
            refs = headers.get("References", "")
            mime["References"] = (refs + " " + orig_msgid).strip() if refs else orig_msgid

        # 10) Encode + send in same Gmail thread
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
        thread_id = msg.get("threadId")

        sent = (
            service.users()
            .messages()
            .send(
                userId="me",
                body={"raw": raw, "threadId": thread_id},
            )
            .execute()
        )

        return f"OK|replied_all_id:{sent.get('id')}"

    except Exception as e:
        return f"[ToolError] reply_all_email failed: {type(e).__name__}: {e}"

@tool(
    "get_all_emails",
    description="Load all saved emails from docs/emails.txt and return them as a JSON list."
)
def get_all_emails() -> str:
    path = "docs/emails.txt"
    try:
        if not os.path.exists(path):
            return json.dumps({"emails": []}, ensure_ascii=False)

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        # Clean and dedupe
        emails = []
        seen = set()
        for e in lines:
            if not e:
                continue
            if e not in seen:
                seen.add(e)
                emails.append(e)

        return json.dumps({"emails": emails}, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"error": True, "type": type(e).__name__, "message": str(e)},
            ensure_ascii=False
        )

@tool(
    "update_emails",
    description=(
        "Modify docs/emails.txt. Args:\n"
        "- action: 'add' or 'remove'\n"
        "- email: the email address to add/remove.\n"
        "Maintains one email per line, deduplicated."
    )
)
def update_emails(action: str, email: str) -> str:
    path = "docs/emails.txt"
    email = email.strip()

    try:
        # Load existing emails
        emails = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                emails = [line.strip() for line in f.readlines() if line.strip()]

        # Deduplicate
        emails = list(dict.fromkeys(emails))

        action = action.lower()

        if action == "add":
            if email not in emails:
                emails.append(email)
                message = f"Added: {email}"
            else:
                message = f"Email already exists: {email}"

        elif action == "remove":
            if email in emails:
                emails.remove(email)
                message = f"Removed: {email}"
            else:
                message = f"Email not found: {email}"

        else:
            return json.dumps(
                {"error": True, "message": "Invalid action (use 'add' or 'remove')."},
                ensure_ascii=False
            )

        # Ensure docs/ exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write back to file
        with open(path, "w", encoding="utf-8") as f:
            for e in emails:
                f.write(e + "\n")

        return json.dumps(
            {"success": True, "message": message, "emails": emails},
            ensure_ascii=False
        )

    except Exception as e:
        return json.dumps(
            {"error": True, "type": type(e).__name__, "message": str(e)},
            ensure_ascii=False
        )


TOOLS = [search_emails, send_email, reply_email, reply_all_email, is_reply_or_reply_all, get_all_emails, update_emails]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}