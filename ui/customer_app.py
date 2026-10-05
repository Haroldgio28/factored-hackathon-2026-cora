"""Customer chat view: a thin Streamlit client over the CORA FastAPI service (task 5.4, REQ-05/REQ-49).

A bank customer authenticates (mock OTP) and converses with CORA. This view holds NO business logic
(ponytail: identity, policy, grounding and masking all live behind the API) - it only calls the 5.3
service over stdlib `urllib` and renders the returned customer-facing text plus its trace id.

Language theme (input side): the customer picks es/pt EXPLICITLY with a widget. That is the user's
explicit choice, not language detection, and it is sent to `/chat` as `language` so the turn runs
in the chosen language. The server treats it as a UI PREFERENCE (never identity) and validates it
fail-closed: an invalid value is ignored and the orchestrator falls back to per-turn detection.

Run with: `uv run streamlit run ui/customer_app.py` (needs the `ui` dependency group). It needs the
API running: `uv run uvicorn cora.api.app:create_app --factory`. The API base URL comes from
`CORA_API_BASE_URL` (default `http://127.0.0.1:8000`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

# Customer-facing UI copy. BOTH languages must carry EVERY key; a missing translation fails the
# test (mirrors how templates/agent_console enforce es+pt completeness). Spanish is a neutral LATAM
# register; Portuguese is pt-BR.
UI_COPY: dict[str, dict[str, str]] = {
    "es": {
        "title": "CORA - asistente de tu banco",
        "language_label": "Idioma",
        "customer_id_label": "Identificador de cliente",
        "login_button": "Ingresar",
        "chat_placeholder": "Escribe tu mensaje",
        "send_button": "Enviar",
        "greeting": "Hola, soy CORA. ¿En qué puedo ayudarte hoy?",
        "login_prompt": "Ingresa tu identificador de cliente para empezar.",
        "login_failed": "No pudimos verificar tu identidad. Inténtalo de nuevo.",
        "server_unreachable": "No pudimos conectar con el servicio. Inténtalo más tarde.",
        "turn_failed": "Ocurrió un problema al procesar tu mensaje. Inténtalo de nuevo.",
        "thinking": "Un momento, estoy revisando...",
    },
    "pt": {
        "title": "CORA - assistente do seu banco",
        "language_label": "Idioma",
        "customer_id_label": "Identificador do cliente",
        "login_button": "Entrar",
        "chat_placeholder": "Escreva sua mensagem",
        "send_button": "Enviar",
        "greeting": "Olá, sou a CORA. Como posso ajudar você hoje?",
        "login_prompt": "Informe seu identificador de cliente para começar.",
        "login_failed": "Não conseguimos verificar sua identidade. Tente novamente.",
        "server_unreachable": "Não conseguimos conectar ao serviço. Tente mais tarde.",
        "turn_failed": "Ocorreu um problema ao processar sua mensagem. Tente novamente.",
        "thinking": "Um momento, estou verificando...",
    },
}

LANGUAGES: tuple[str, ...] = ("es", "pt")
DEFAULT_LANGUAGE = "es"


def ui_text(language: str, key: str) -> str:
    """Look up one UI-copy string, falling back to the default language if the lang is unknown.

    Pure so the copy path is testable without Streamlit. A `KeyError` on `key` is intentional: a
    missing copy key is a bug the test must catch, not something to paper over at runtime.
    """
    return UI_COPY.get(language, UI_COPY[DEFAULT_LANGUAGE])[key]


def trace_label(trace_id: object) -> str:
    """Format a turn's `trace_id` (task 5.1) into the visible, copyable reference line.

    Pure so the UI retrieval path is testable without Streamlit. Mirrors the agent console's
    `trace_label`: an empty/missing id reads as a plain "none" rather than a blank line.
    """
    trace_id = str(trace_id or "").strip()
    return f"trace: {trace_id}" if trace_id else "trace: (none)"


def login_submission(submitted: bool, customer_id: str) -> bool:
    """Decide whether to attempt auth this run: only on an atomic submit with a non-blank id.

    Pure so the one-click login decision is testable without Streamlit. Driving auth off this
    predicate (fed by `st.form_submit_button`, which submits the typed value atomically on the
    first click) fixes the stale-`st.button` rerun bug where the first click did nothing.
    """
    return bool(submitted and customer_id.strip())


def login_succeeded(token: object) -> bool:
    """Decide whether the just-run auth attempt produced a usable session token.

    Pure so the "repaint the authenticated view now" decision is testable without Streamlit.
    Storing the token does not restart `main()` by itself, so a successful first-click submit must
    trigger an explicit rerun; a `None`/blank token (error path) stays on the login screen.
    """
    return bool(token)


def _parse_chat_response(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract `(response_text, trace_id)` from a `/chat` JSON body (the 5.3 ChatResponse shape).

    Pure so the response-parsing path is testable with a sample dict and no network. Missing fields
    degrade to empty strings rather than raising, so a malformed body cannot crash the render.
    """
    return str(payload.get("response_text") or ""), str(payload.get("trace_id") or "")


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON body and return the decoded JSON response (stdlib urllib; no new HTTP dep).

    Raises `urllib.error.URLError` when the server is unreachable and `urllib.error.HTTPError` on a
    4xx/5xx - the caller maps both to a friendly es/pt message so the UI never crashes.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # A real Bedrock turn (polish + grounding + bounded retries) can take well over 30s on a cold
    # start, so the client timeout is generous; a genuine hang still surfaces as TimeoutError and
    # the UI degrades to a friendly message rather than crashing.
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed http(s) API base URL
        return json.loads(resp.read().decode("utf-8"))


def authenticate(api_base_url: str, customer_id: str) -> str:
    """Run the 5.3 two-step OTP flow against `/auth` and return a session token.

    The mock IdP returns the one-time code in the challenge, so the demo completes the flow without
    an out-of-band channel. No business logic here - the server owns identity.
    """
    base = api_base_url.rstrip("/")
    start = _post_json(f"{base}/auth", {"customer_id": customer_id})
    verify = _post_json(
        f"{base}/auth",
        {
            "customer_id": customer_id,
            "challenge_id": start["challenge_id"],
            "otp_code": start["otp_code"],
        },
    )
    return str(verify["token"])


def _chat_body(token: str, utterance: str, language: str) -> dict[str, Any]:
    """Build the `/chat` request body (pure, so request-building is testable without a live call).

    Identity travels ONLY as the session token (the body carries no `customer_id`; the server
    forbids unknown fields). `language` is the customer's explicit es/pt toggle: a UI preference,
    not identity, that the server validates fail-closed (an invalid value is ignored there).
    """
    return {"token": token, "utterance": utterance, "language": language}


def send_turn(api_base_url: str, token: str, utterance: str, language: str) -> tuple[str, str]:
    """Send one chat turn to `/chat` and return the parsed `(response_text, trace_id)`.

    The customer's explicit es/pt choice is sent as `language` so the turn runs in it (the server
    validates it fail-closed and falls back to detection on an invalid value); identity still
    travels only as the session token.
    """
    base = api_base_url.rstrip("/")
    payload = _post_json(f"{base}/chat", _chat_body(token, utterance, language))
    return _parse_chat_response(payload)


def main() -> None:
    import streamlit as st

    from cora.settings import get_settings

    api_base_url = get_settings().api_base_url

    # `set_page_config` MUST be the first Streamlit command on the page (Streamlit raises
    # StreamlitSetPageConfigMustBeFirstCommandError otherwise), so it runs BEFORE the language
    # selectbox. The page title is the language-neutral product name ("CORA"); the per-language
    # title/greeting are rendered after the toggle below.
    st.set_page_config(page_title="CORA", layout="centered")

    language = st.sidebar.selectbox(
        ui_text(DEFAULT_LANGUAGE, "language_label"),
        LANGUAGES,
        index=LANGUAGES.index(DEFAULT_LANGUAGE),
    )

    st.title(ui_text(language, "title"))
    st.caption(ui_text(language, "greeting"))

    if "token" not in st.session_state:
        st.session_state.token = None
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.token is None:
        st.info(ui_text(language, "login_prompt"))
        # An `st.form` submits the id + button atomically on the FIRST click, fixing the
        # stale-`st.button` bug (see `login_submission`). Storing the token is not enough: it does
        # not restart `main()`, so a successful submit must `st.rerun()` to repaint the chat view
        # on the same first click, while an error stays on the login screen.
        with st.form("login"):
            customer_id = st.text_input(ui_text(language, "customer_id_label"))
            submitted = st.form_submit_button(ui_text(language, "login_button"))
        if login_submission(submitted, customer_id):
            with st.spinner(ui_text(language, "thinking")):
                try:
                    st.session_state.token = authenticate(api_base_url, customer_id)
                except urllib.error.HTTPError:
                    st.error(ui_text(language, "login_failed"))
                except urllib.error.URLError:
                    st.error(ui_text(language, "server_unreachable"))
            if login_succeeded(st.session_state.token):
                st.rerun()
        return

    for turn in st.session_state.history:
        st.chat_message(turn["role"]).write(turn["text"])
        if turn.get("trace"):
            st.chat_message(turn["role"]).caption(trace_label(turn["trace"]))

    utterance = st.chat_input(ui_text(language, "chat_placeholder"))
    if utterance:
        # Record and PAINT the user's message immediately, so it is visible while the (possibly
        # slow) real LLM turn runs - not only after the next rerun.
        st.session_state.history.append({"role": "user", "text": utterance})
        st.chat_message("user").write(utterance)
        # Any transport failure degrades to a friendly es/pt message instead of crashing the app:
        # HTTPError (4xx/5xx), URLError (unreachable), and a raw socket TimeoutError/OSError (a
        # slow turn that exceeds the client timeout) are all handled. `thinking` is shown meanwhile.
        with st.chat_message("assistant"), st.spinner(ui_text(language, "thinking")):
            try:
                response_text, trace_id = send_turn(api_base_url, st.session_state.token, utterance, language)
                st.session_state.history.append(
                    {"role": "assistant", "text": response_text, "trace": trace_id}
                )
            except urllib.error.HTTPError:
                st.session_state.history.append(
                    {"role": "assistant", "text": ui_text(language, "turn_failed")}
                )
            except (urllib.error.URLError, TimeoutError, OSError):
                st.session_state.history.append(
                    {"role": "assistant", "text": ui_text(language, "server_unreachable")}
                )
        st.rerun()


if __name__ == "__main__":
    main()
