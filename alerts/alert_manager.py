"""
alerts/alert_manager.py — v3.3
===============================================================
SMS delivery options (in priority order):
  1. Fast2SMS  — free Indian SMS API, no Twilio needed
                 Sign up at fast2sms.com → get API key (free ₹50 credits)
  2. Twilio    — if credentials present in .env
  3. Console   — always logs full alert to terminal

To get Fast2SMS key (2 minutes):
  → https://www.fast2sms.com/
  → Register → Dashboard → Dev API → Copy API key
  → Paste in .env as FAST2SMS_API_KEY=your_key
===============================================================
"""

import os, logging, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AlertManager:

    def __init__(self):
        # Fast2SMS (Indian SMS — free tier)
        self.fast2sms_key = os.getenv("FAST2SMS_API_KEY", "").strip()

        # Twilio (optional)
        self.sid      = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.token    = os.getenv("TWILIO_AUTH_TOKEN",  "").strip()
        self.from_num = os.getenv("TWILIO_FROM_NUMBER", "").strip()
        self.client   = None

        # Recipients
        raw = os.getenv("ALERT_PHONE_NUMBERS", "+919096568129")
        self.recipients = [n.strip() for n in raw.split(",") if n.strip()]
        # Indian numbers without country code (for Fast2SMS)
        self.indian_numbers = [
            n.replace("+91", "").replace(" ", "")
            for n in self.recipients if "91" in n
        ]

        # Init Twilio if credentials present
        if self.sid and self.token and self.from_num:
            try:
                from twilio.rest import Client
                self.client = Client(self.sid, self.token)
                logger.info(f"✅ Twilio ready → {self.recipients}")
            except Exception as e:
                logger.warning(f"Twilio init failed: {e}")

        # Status
        if self.fast2sms_key:
            logger.info(f"✅ Fast2SMS ready → {self.indian_numbers}")
        elif not self.client:
            logger.info(
                "⚠  No SMS provider configured.\n"
                "   For FREE Indian SMS: sign up at fast2sms.com → add key to .env as FAST2SMS_API_KEY=...\n"
                f"   Console alerts will print for: {self.recipients}"
            )

    # ── PUBLIC ────────────────────────────────

    def send_alert(self, event) -> list:
        msg = self._build_message(event)
        self._log_alert(event, msg)     # always log to terminal
        results = []

        sent = False

        # Try Fast2SMS first (free, works in India)
        if self.fast2sms_key and self.indian_numbers:
            r = self._send_fast2sms(msg, event)
            results.extend(r)
            sent = True

        # Try Twilio if available
        if self.client:
            for num in self.recipients:
                try:
                    m = self.client.messages.create(
                        body=msg, from_=self.from_num, to=num)
                    results.append(m.sid)
                    logger.info(f"✅ Twilio SMS → {num}  SID={m.sid}")
                    sent = True
                except Exception as e:
                    logger.error(f"❌ Twilio failed → {num}: {e}")

        if not sent:
            logger.info(
                "\n" + "━"*50 +
                "\n  ⚠  NO SMS PROVIDER CONFIGURED" +
                "\n  Get free SMS: https://www.fast2sms.com/" +
                "\n  Add FAST2SMS_API_KEY=your_key to .env file" +
                "\n  Alert details printed above in terminal" +
                "\n" + "━"*50
            )
            results.append("CONSOLE_ONLY")

        return results

    def send_voice_call(self, event) -> list:
        """Voice call via Twilio for Critical severity only."""
        if not self.client or event.severity < 3:
            if event.severity >= 3:
                logger.warning("⚠  Critical accident — no Twilio configured for voice call")
            return []
        loc   = event.location.get("name", "the monitored location")
        twiml = (
            f"<Response><Say voice='alice' loop='2'>"
            f"Emergency. Critical road accident at {loc}. Respond immediately."
            f"</Say></Response>"
        )
        results = []
        for num in self.recipients:
            try:
                c = self.client.calls.create(
                    twiml=twiml, from_=self.from_num, to=num)
                results.append(c.sid)
                logger.info(f"✅ Voice call → {num}  SID={c.sid}")
            except Exception as e:
                logger.error(f"❌ Call failed → {num}: {e}")
        return results

    # ── FAST2SMS ─────────────────────────────

    def _send_fast2sms(self, message: str, event) -> list:
        """
        Send SMS via Fast2SMS DLT/Quick Send API.
        Free tier: ~50 SMS credits on signup.
        """
        # Fast2SMS has a 160-char limit on quick send — use short message
        short = self._build_short_message(event)
        numbers = ",".join(self.indian_numbers)

        try:
            resp = requests.post(
                "https://www.fast2sms.com/dev/bulkV2",
                headers={
                    "authorization": self.fast2sms_key,
                    "Content-Type": "application/json",
                },
                json={
                    "route":   "q",          # quick route
                    "message": short,
                    "numbers": numbers,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("return"):
                logger.info(f"✅ Fast2SMS sent → {numbers}  | {data.get('message','OK')}")
                return [f"FAST2SMS:{numbers}"]
            else:
                logger.error(f"❌ Fast2SMS error: {data}")
                return [f"FAST2SMS_ERROR:{data}"]
        except Exception as e:
            logger.error(f"❌ Fast2SMS request failed: {e}")
            return [f"ERROR:{e}"]

    # ── MESSAGE BUILDERS ─────────────────────

    def _build_short_message(self, event) -> str:
        """Short message for Fast2SMS (under 160 chars)."""
        loc   = event.location
        lat   = loc.get("lat",  19.9975)
        lng   = loc.get("lng",  73.7898)
        name  = loc.get("name", "Nashik")
        score = round(getattr(event, "accident_score", 0.75) * 100)
        emoji = {1:"[MINOR]", 2:"[MODERATE]", 3:"[CRITICAL]"}.get(event.severity, "[ALERT]")
        ts    = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")

        return (
            f"ACCIDENT ALERT {emoji} {ts} | {name} | "
            f"Score:{score}% | {len(event.vehicles_involved)} vehicles | "
            f"Maps:https://maps.google.com/?q={lat},{lng}"
        )

    def _build_message(self, event) -> str:
        """Full message for Twilio / console log."""
        ts    = datetime.fromtimestamp(event.timestamp).strftime("%d-%b-%Y %H:%M:%S")
        loc   = event.location
        lat   = loc.get("lat",  19.9975)
        lng   = loc.get("lng",  73.7898)
        name  = loc.get("name", "Nashik, Maharashtra, India")
        conf  = round(event.confidence * 100)
        score = round(getattr(event, "accident_score", 0.75) * 100)
        maps  = f"https://maps.google.com/?q={lat},{lng}"
        emoji = {1:"🟡", 2:"🟠", 3:"🔴"}.get(event.severity, "⚠️")
        n_veh = len(event.vehicles_involved)

        return (
            f"{emoji} ROAD ACCIDENT ALERT\n"
            f"{'━'*32}\n"
            f"Severity    : {event.severity_label.upper()}\n"
            f"Time        : {ts}\n"
            f"Location    : {name}\n"
            f"Coordinates : {lat}°N, {lng}°E\n"
            f"Vehicles    : {n_veh} involved\n"
            f"AI Score    : {score}%\n"
            f"Confidence  : {conf}%\n"
            f"{'━'*32}\n"
            f"📍 Google Maps:\n{maps}\n"
            f"{'━'*32}\n"
            f"AccidentWatch v3 · Nashik Traffic AI"
        )

    def _log_alert(self, event, msg: str):
        border = "═" * 58
        logger.info(
            f"\n{border}\n"
            f"  🚨 ACCIDENT ALERT  →  {self.recipients}\n"
            f"{border}\n"
            f"{msg}\n"
            f"{border}"
        )
