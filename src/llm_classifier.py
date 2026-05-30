from jinja2 import Template
import re
import json

from openai import OpenAI
from config import Config


_ZERO_SHOT_TEMPLATE = """Considérez l'avis suivant:

"{{text}}"

Quelle est la valeur de l'opinion exprimée sur chacun des aspects suivants : Prix, Cuisine, Service?

La valeur d'une opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "Non exprimée".

La réponse doit se limiter au format json suivant:
{ "Prix": opinion, "Cuisine": opinion, "Service": opinion}."""


_FEW_SHOT_TEMPLATE = """Votre tâche est d'analyser les opinions exprimées dans des avis de restaurants sur trois aspects : Prix, Cuisine, Service.
Pour chaque aspect, la valeur doit être : "Positive", "Négative", "Neutre", ou "Non exprimée".

Voici quelques exemples:

Avis: "La cuisine était excellente, des saveurs raffinées et des portions généreuses. Le service était attentionné et rapide. Par contre, l'addition était salée pour ce qu'on a eu."
Réponse: {"Prix": "Négative", "Cuisine": "Positive", "Service": "Positive"}

Avis: "Bon rapport qualité-prix, on ne peut pas se plaindre. Les plats étaient corrects sans plus, rien d'exceptionnel. Le serveur n'était pas très aimable."
Réponse: {"Prix": "Positive", "Cuisine": "Neutre", "Service": "Négative"}

Avis: "J'ai adoré les desserts, vraiment délicieux. Je n'ai aucun commentaire sur le reste."
Réponse: {"Prix": "Non exprimée", "Cuisine": "Positive", "Service": "Non exprimée"}

Maintenant analysez cet avis:

Avis: "{{text}}"
Réponse:"""


def normalize_label(label: str) -> str:
    if not label:
        return "NE"
    l = label.lower()
    if "posit" in l:
        return "Positive"
    if "neg" in l:
        return "Négative"
    if "neutr" in l:
        return "Neutre"
    if "non" in l or label.upper() == "NE":
        return "NE"
    return "NE"


class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llmclient = OpenAI(
            base_url=cfg.ollama_url,
            api_key='EMPTY'
        )
        self.model_name = 'qwen2.5:7b-instruct-q4_K_M'
        self.model_params = {
            'max_tokens': 1000,
            'temperature': 0.1,
            'top_p': 0.9,
        }
        template_str = _FEW_SHOT_TEMPLATE if cfg.few_shot else _ZERO_SHOT_TEMPLATE
        self.jtemplate = Template(template_str)

    def predict(self, text: str) -> dict[str, str]:
        prompt = self.jtemplate.render(text=text)
        try:
            response = self.llmclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Vous êtes un assistant expert en analyse de sentiments. Répondez uniquement en JSON."},
                    {"role": "user", "content": prompt}
                ],
                **self.model_params
            )
            content = response.choices[0].message.content
            jresp = self.parse_json_response(content)
            if jresp is None:
                jresp = {"Prix": "NE", "Cuisine": "NE", "Service": "NE"}
            for aspect in jresp:
                jresp[aspect] = normalize_label(jresp[aspect])
            return jresp
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"Prix": "NE", "Cuisine": "NE", "Service": "NE"}

    def parse_json_response(self, response: str) -> dict[str, str] | None:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except Exception:
                return None
        return None
