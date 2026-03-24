import json
import os
from http.server import BaseHTTPRequestHandler
from openai import OpenAI


def build_prompt(mission_name: str, context: str) -> str:
    return f"""Je helpt bij het ontwerpen van een innovatie-workshop over AI.

Missienaam: "{mission_name}"
Context: "{context}"

Genereer precies 5 processtappen voor dit proces/uitdaging. Elke stap heeft:
- Een korte naam (1-3 woorden, Nederlands)
- Een bondige omschrijving (max 12 woorden, wat er in deze stap gebeurt)
- Een prikkelende AI-missievraag in het Nederlands (1 zin, begint met "Hoe zou je AI kunnen..." of vergelijkbaar, specifiek voor dit domein)

Reageer ALLEEN met geldig JSON, geen uitleg, geen markdown, geen backticks:
[
  {{"name":"...","desc":"...","q":"..."}},
  {{"name":"...","desc":"...","q":"..."}},
  {{"name":"...","desc":"...","q":"..."}},
  {{"name":"...","desc":"...","q":"..."}},
  {{"name":"...","desc":"...","q":"..."}}
]"""


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            mission_name = body.get("mission_name", "").strip()
            context = body.get("context", "").strip()

            if not mission_name or not context:
                self._send_error(400, "mission_name en context zijn verplicht")
                return

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self._send_error(500, "OPENAI_API_KEY is niet geconfigureerd")
                return

            client = OpenAI(api_key=api_key)
            prompt = build_prompt(mission_name, context)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7,
            )

            raw = response.choices[0].message.content.strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            steps = json.loads(clean)

            if not isinstance(steps, list) or len(steps) == 0:
                self._send_error(500, "Geen stappen ontvangen van AI")
                return

            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(steps).encode())

        except json.JSONDecodeError:
            self._send_error(400, "Ongeldige JSON in request of AI-response")
        except Exception as e:
            self._send_error(500, f"Server fout: {str(e)}")

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_error(self, code, message):
        self.send_response(code)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
