# propts will go here

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an expert educational content creator specializing in exam design.
    Using the provided context, generate up to 3 multiple-choice questions (MCQs).
    Context: {context}
    Topic: {topic}
    Instructions:
    - If topic is empty, create MCQs covering the document's core concepts.
    - Each MCQ must have 1 correct answer, 3 distractors, and a short explanation.
    - Return **strictly JSON** in this format:
    [
      {{
        "question": "...",
        "options": {{
            "A": "...",
            "B": "...",
            "C": "...",
            "D": "..."
        }},
        "correct_answer": "...",
        "explanation": "..."
      }},
      ...
    ]
    """
)

ai_prompt = AIMessagePromptTemplate.from_template(
    """
    [
      {{
        "question": "What is the main concept discussed in the provided context?",
        "options": {{
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D"
        }},
        "correct_answer": "A",
        "explanation": "Explanation about why A is correct."
      }}
    ]
    """
)

human_prompt = HumanMessagePromptTemplate.from_template(
    """
    Generate multiple-choice questions (MCQs) based on the following information.
    Context:
    {context}

    Topic or Task:
    {topic}
    Please follow the system instructions and return the output in the required JSON format.
    """
)

PROMPT_REGISTRY = {
    "system_prompt": system_prompt,
    "ai_prompt": ai_prompt,
    "human_prompt": human_prompt

}


prompt = """
            You are an expert educational content creator specializing in exam design.
                Using the provided context, generate up to 10 multiple-choice questions (MCQs).
                Instructions:

                - If topic is empty, create MCQs covering the document's core concepts.
                - Each MCQ must have 1 correct answer, 3 distractors, and a short explanation.
                - Return **strictly JSON** in this format:
                [
                {{
                    "question": "...",
                    "options": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }},
                    "correct_answer": "...",
                    "explanation": "..."
                }},
                ...
                ]
        """


custom_prompt = PromptTemplate(
    template = (
        """
            You are an expert educational content creator specializing in exam design.
                Using the provided context, generate up to 10 multiple-choice questions (MCQs).
                Instructions:
                - If topic is empty, create MCQs covering the document's core concepts.
                - Each MCQ must have 1 correct answer, 3 distractors, and a short explanation.
                - Return **strictly JSON** in this format:
                [
                {{
                    "question": "...",
                    "options": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }},
                    "correct_answer": "...",
                    "explanation": "..."
                }},
                ...
                ]
        """
    )
)