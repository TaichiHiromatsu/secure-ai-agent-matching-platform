from google.genai import types

# Relaxed safety for A2A pipelines (intentionally permissive for judge/debate)
SAFETY_SETTINGS_RELAXED = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.OFF),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.OFF),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.OFF),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.OFF),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.OFF),
    # Jailbreak category is present in newer GenAI releases; included for completeness
    types.SafetySetting(category=getattr(types.HarmCategory, "HARM_CATEGORY_JAILBREAK", types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT),
                        threshold=types.HarmBlockThreshold.OFF),
]
