import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


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


@app.post("/api/generate")
async def generate(request: Request):
    try:
        body = await request.json()

        mission_name = body.get("mission_name", "").strip()
        context = body.get("context", "").strip()

        if not mission_name or not context:
            return JSONResponse(
                status_code=400,
                content={"error": "mission_name en context zijn verplicht"},
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"error": "OPENAI_API_KEY is niet geconfigureerd"},
            )

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
            return JSONResponse(
                status_code=500,
                content={"error": "Geen stappen ontvangen van AI"},
            )

        return JSONResponse(content=steps)

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Ongeldige JSON in request of AI-response"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Server fout: {str(e)}"},
        )
