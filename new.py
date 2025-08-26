import json
import textwrap
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

"""
Adaptive 15-question career counseling bot using Groq Llama 4.
- Direct API key in code (as you requested)
- Robust JSON extraction for model outputs
- Skill → Numeracy → Literacy phases
- Final report auto-saved as PDF with line wrapping
"""

# ======== Config ========
API_KEY = "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"  # ⚠️ replace with your real key
TRADES = ["Electrician", "Carpenter", "Plumber", "Mason", "Painter"]
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS_DEFAULT = 300

# ======== Initialize Client ========
client = Groq(api_key=API_KEY)

# ======== Prompts ========
TRANSLATION_PROMPT = """
Translate this question and options into {language}.
Return ONLY a readable multiple-choice question in plain text.

Question: {question_text}
Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
""".strip()

SKILL_DETERMINATION_PROMPT = """
You are an expert career counselor AI specializing in trade skills assessment.
Phase: SKILL DETERMINATION (Questions 1–5).

Instructions:
- Persona: Friendly and encouraging counselor.
- Language: {language}.
- Based on conversation history, ask a scenario-based MCQ revealing trade aptitude.
- Consider these trades: {trades}.
- Output ONLY JSON: {{"question_text": "...", "options": {{"A": "...","B":"...","C":"...","D":"..."}}}}
- No extra text, no backticks.

Conversation History:
{conversation_history}

Current Question: {question_number}/5
""".strip()

NUMERACY_ASSESSMENT_PROMPT = """
You are conducting the NUMERACY ASSESSMENT phase (Q6–10) for {trade}.
Language: {language}.
Ask a math-related MCQ relevant to {trade}.
Output ONLY JSON: {{"question_text": "...", "options": {{"A": "...","B":"...","C":"...","D":"..."}}}}
No extra text, no backticks.

Conversation History:
{conversation_history}

Current Question: {question_number}/10
""".strip()

LITERACY_ASSESSMENT_PROMPT = """
You are conducting the LITERACY ASSESSMENT phase (Q11–15) for {trade}.
Language: {language}.
Ask a reading comprehension MCQ relevant to {trade}.
Output ONLY JSON: {{"question_text": "...", "options": {{"A": "...","B":"...","C":"...","D":"..."}}}}
No extra text, no backticks.

Conversation History:
{conversation_history}

Current Question: {question_number}/15
""".strip()

SKILL_DETERMINATION_AI_PROMPT = """
Analyze the first 5 Q&A to identify the most suitable trade from {trades}.
Reply ONLY JSON: {{"determined_skill": "<trade>", "confidence": 0.0-1.0}}
No extra text.
""".strip()

REPORT_GENERATOR_PROMPT = """
You are a master career analyst AI.

Analyze the full 15-question conversation transcript below, calculating scores for:
- skill (Q1–5),
- numeracy (Q6–10),
- literacy (Q11–15).

Identify strengths and recommend the most suitable trade.
Return ONLY JSON with keys:
- scores: {{"skill": number, "numeracy": number, "literacy": number, "overall": number}}
- strengths: [string, ...]
- recommended_trade: string
- is_final_report: true
- rationale: string

Transcript:
{conversation_history}
""".strip()

FIRST_QUESTION = {
    "question_text": (
        "You're working on a construction site and need to repair a faulty circuit that controls the lights "
        "in a large hall. You're given a choice of tools and materials. What would you prefer to use to "
        "diagnose and fix the issue?"
    ),
    "options": {
        "A": "A multimeter to measure voltage and a wire stripper to access the wires",
        "B": "A hammer to tap on the walls and a level to ensure the circuit box is straight",
        "C": "A pipe wrench to grip the circuit box and a hacksaw to cut new wires",
        "D": "A paintbrush to inspect the circuit box and a putty knife to clean out old paint"
    }
}

# ======== Helpers ========
def call_groq_ai(prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def translate_question(language: str, question_data: dict) -> str:
    prompt = TRANSLATION_PROMPT.format(
        language=language,
        question_text=question_data["question_text"],
        option_a=question_data["options"]["A"],
        option_b=question_data["options"]["B"],
        option_c=question_data["options"]["C"],
        option_d=question_data["options"]["D"],
    )
    return call_groq_ai(prompt).strip()

def extract_json_block(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1].strip()
        # crude but effective balance check
        if candidate.count("{") == candidate.count("}"):
            return candidate
    return None

def parse_question_json(text: str) -> dict | None:
    block = extract_json_block(text) or text
    try:
        data = json.loads(block)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "question_text" not in data or "options" not in data:
        return None
    opts = data["options"]
    if not isinstance(opts, dict):
        return None
    if not {"A", "B", "C", "D"}.issubset(opts.keys()):
        return None
    return data

def get_next_question(phase, qnum, convo_text, language, skill=None) -> dict:
    if phase == "skill_determination":
        prompt = SKILL_DETERMINATION_PROMPT.format(
            language=language,
            trades=", ".join(TRADES),
            conversation_history=convo_text,
            question_number=qnum
        )
    elif phase == "numeracy":
        prompt = NUMERACY_ASSESSMENT_PROMPT.format(
            language=language,
            conversation_history=convo_text,
            question_number=qnum,
            trade=skill or TRADES[0]
        )
    elif phase == "literacy":
        prompt = LITERACY_ASSESSMENT_PROMPT.format(
            language=language,
            conversation_history=convo_text,
            question_number=qnum,
            trade=skill or TRADES[0]
        )
    else:
        raise ValueError("Invalid phase")

    for _ in range(2):
        raw = call_groq_ai(prompt)
        parsed = parse_question_json(raw)
        if parsed:
            return parsed
        prompt += "\nReturn ONLY valid minified JSON."
    raise RuntimeError(f"Model did not return valid JSON for phase={phase}, q={qnum}.")

def pretty_print_question(qnum: int, qdata):
    print(f"\nQuestion {qnum}:")
    if isinstance(qdata, dict):
        print(qdata["question_text"])
        for k in ["A", "B", "C", "D"]:
            print(f"{k}) {qdata['options'][k]}")
    else:
        print(qdata)

# ======== PDF Report Generator (with line wrapping) ========
def save_report_as_pdf(report_text: str, filename: str = "final_report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    title = "Career Counseling Final Report"
    margin_left = 50
    margin_right = 40
    top = height - 50
    line_height = 16

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, top, title)

    # Body
    c.setFont("Helvetica", 11)
    y = top - 30

    # wrap lines to fit page width
    max_width = width - margin_left - margin_right
    for raw_line in report_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            y -= line_height
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            continue

        # word-wrap manually
        words = line.split()
        current = ""
        for w in words:
            test = (current + " " + w).strip()
            if stringWidth(test, "Helvetica", 11) <= max_width:
                current = test
            else:
                c.drawString(margin_left, y, current)
                y -= line_height
                if y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = height - 50
                current = w
        if current:
            c.drawString(margin_left, y, current)
            y -= line_height
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50

    c.save()
    print(f"\n✅ Final Report saved as: {filename}")

# ======== Main ========
def main():
    print("Welcome to the Career Counseling Assessment")
    name = input("Enter your name: ").strip()
    language = input("Preferred language (e.g., Hindi, English): ").strip()

    conversation = []
    current_question = 1
    phase = "skill_determination"
    skill = None

    # Q1 (translated plain text)
    translated_first = translate_question(language, FIRST_QUESTION)
    pretty_print_question(current_question, translated_first)

    while current_question <= 15:
        answer = input("Your answer (A/B/C/D): ").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            print("Please enter A, B, C, or D.")
            continue

        last_q = translated_first if current_question == 1 else last_q_json["question_text"]
        conversation.append({"question": last_q, "answer": answer})
        convo_text = "\n".join(
            [f"Q{i+1}: {c['question']}\nA{i+1}: {c['answer']}" for i, c in enumerate(conversation)]
        )

        if phase == "skill_determination":
            if current_question < 5:
                current_question += 1
                last_q_json = get_next_question("skill_determination", current_question, convo_text, language)
                pretty_print_question(current_question, last_q_json)
            else:
                # after Q5 -> determine skill
                analysis_prompt = SKILL_DETERMINATION_AI_PROMPT.format(trades=", ".join(TRADES)) + "\n\n" + convo_text
                result = call_groq_ai(analysis_prompt)
                try:
                    skill = json.loads(extract_json_block(result) or "{}").get("determined_skill")
                except Exception:
                    skill = None
                phase = "numeracy"
                current_question += 1
                last_q_json = get_next_question("numeracy", current_question, convo_text, language, skill)
                pretty_print_question(current_question, last_q_json)

        elif phase == "numeracy":
            if current_question < 10:
                current_question += 1
                last_q_json = get_next_question("numeracy", current_question, convo_text, language, skill)
                pretty_print_question(current_question, last_q_json)
            else:
                phase = "literacy"
                current_question += 1
                last_q_json = get_next_question("literacy", current_question, convo_text, language, skill)
                pretty_print_question(current_question, last_q_json)

        elif phase == "literacy":
            if current_question < 15:
                current_question += 1
                last_q_json = get_next_question("literacy", current_question, convo_text, language, skill)
                pretty_print_question(current_question, last_q_json)
            else:
                # Build final report
                report_prompt = REPORT_GENERATOR_PROMPT.format(conversation_history=convo_text)
                final_report = call_groq_ai(report_prompt, 700)  # a bit more room
                print("\nFinal Report:\n", final_report)

                # Save as PDF
                safe_name = "".join(ch for ch in name if ch.isalnum() or ch in (" ", "_", "-")).strip() or "candidate"
                pdf_name = f"{safe_name}_career_report.pdf"
                save_report_as_pdf(final_report, pdf_name)
                break

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")