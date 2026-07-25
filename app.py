import streamlit as st
from google import genai
import re
import time
from PyPDF2 import PdfReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib import styles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if "optimized_resume" not in st.session_state:
    st.session_state.optimized_resume = ""

GEMINI_API_KEY = "AIzaSyBMbsFIaYG5WQKCwuWEcyCYZgxeTemQD6Q"

client = genai.Client(
    api_key=GEMINI_API_KEY
)
# CSS Styling
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f5f7fb;
}

/* Main Title */
h1 {
    text-align: center;
    color: #1E293B;
    font-size: 42px !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
}

/* Input Boxes */
.stTextArea textarea {
    border-radius: 12px;
}

section[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E2E8F0;
}

/* Metric / Success Boxes */
[data-testid="stAlert"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:

    st.header("About Project")

    st.write("""
This system helps:

**Recruiters**
- Analyze resumes quickly
- Match candidates with job roles

**Applicants**
- Identify missing skills
- Improve resume alignment
- Explore suitable career paths
""")

SKILLS_DATABASE = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "artificial intelligence",
    "data structures",
    "algorithms",
    "git",
    "github",
    "html",
    "css",
    "javascript",

    # ECE Skills
    "embedded systems",
    "electronics",
    "microcontrollers",
    "communication systems",
    "signal processing",
    "pcb design",
    "vlsi",

    # Civil Engineering Skills
    "autocad",
    "revit",
    "qgis",
    "staad pro",
    "surveying",
    "construction management",
    "structural analysis",
    "building design"
]

# Page Configuration
st.set_page_config(
    page_title="Resume Management System",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 Intelligent Resume Lifecycle Management")

st.markdown("""
Analyze resumes intelligently using **NLP-based matching**.

Upload your resume and compare it with a job description to receive:
- ✅ Match Percentage  
- ✅ Skill Gap Analysis  
- ✅ Feedback Suggestions  
- ✅ Career Recommendations  
""")

# Upload Resume
uploaded_resume = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# Job Description
job_description = st.text_area(
    "Enter Job Description",
    placeholder="Paste the job description here..."
)

# Function to extract text from PDF
def extract_resume_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def calculate_match_score(resume_text, job_description):

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return round(similarity_score[0][0] * 100, 2)

def calculate_skill_score(
    resume_skills,
    job_skills
):

    if len(job_skills) == 0:
        return 0

    matched_count = len(
        set(resume_skills) &
        set(job_skills)
    )

    skill_score = (
        matched_count /
        len(job_skills)
    ) * 100

    return round(skill_score, 2)

def extract_skills(text):

    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    found_skills = []

    for skill in SKILLS_DATABASE:

        skill_words = skill.split()

        if len(skill_words) == 1:

            if skill in words:
                found_skills.append(skill)

        else:

            if skill in text:
                found_skills.append(skill)

    return found_skills

def generate_feedback(
    resume_text,
    job_description,
    match_score,
    missing_skills,
    suggested_role
):

    prompt = f"""
    You are an AI recruitment assistant.

    Resume Match Percentage:
    {match_score}

    Suggested Career Path:
    {suggested_role}

    Missing Skills:
    {', '.join(missing_skills)}

    Resume:
    {resume_text[:3000]}

    Job Description:
    {job_description[:2000]}

    Generate professional recruiter-style
    feedback in 3–5 lines in third-person tone.

    Mention:
    - overall alignment
    - strengths
    - missing skills
    - improvement suggestions
    """

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Error: {str(e)}"

def suggest_role(resume_text):

    prompt = f"""
    You are a career recommendation AI.

    Analyze this resume and suggest:

    1. The BEST suited career role
    2. A short explanation (1–2 lines)

    Format EXACTLY like this:

    Career Role:
    [role]

    Why:
    Provide a short professional explanation
    in third-person tone.

    Use professional third-person wording.
    Do NOT use "you" or "your".

    Resume:
    {resume_text[:3000]}
    """

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        return (
            "General Engineering Roles\n\n"
            "Could not generate AI recommendation."
        )
    
def generate_optimized_resume(
    resume_text,
    job_description
):

    prompt = f"""
    You are an expert ATS resume optimizer.

    Given the resume and job description:

    TASK:
    Rewrite and improve the resume
    to better align with the job description.

    RULES:
    - Keep ALL information truthful
    - Do NOT invent fake projects,
      experience, or skills
    - Improve wording professionally
    - Add relevant keywords naturally
    - Reorder skills based on relevance
    - Highlight relevant experience
    - Maintain professional structure

    OUTPUT FORMAT:
    Return a properly structured resume.

    Include sections like:
    - Professional Summary
    - Technical Skills
    - Projects
    - Experience
    - Education

    Resume:
    {resume_text[:5000]}

    Job Description:
    {job_description[:3000]}
    """

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        return f"Error: {str(e)}"

# Analyze Button
analyze = st.button(
    "🔍 Analyze Resume",
    use_container_width=True
)

from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib import styles


import re
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib import styles


def create_resume_pdf(
    optimized_resume
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer
    )

    style_sheet = (
        styles.getSampleStyleSheet()
    )

    heading_style = (
        style_sheet["Heading2"]
    )

    normal_style = (
        style_sheet["BodyText"]
    )

    content = []

    # Remove markdown symbols
    cleaned_text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        optimized_resume
    )

    cleaned_text = re.sub(
        r"#+\s*",
        "",
        cleaned_text
    )

    for line in (
        cleaned_text.split("\n")
    ):

        line = line.strip()

        if not line:
            continue

        # Detect headings
        if line.isupper():

            content.append(
                Paragraph(
                    line,
                    heading_style
                )
            )

        else:

            content.append(
                Paragraph(
                    line.replace(
                        "&",
                        "&amp;"
                    ),
                    normal_style
                )
            )

        content.append(
            Spacer(1, 4)
        )

    doc.build(content)

    pdf_data = (
        buffer.getvalue()
    )

    buffer.close()

    return pdf_data

if analyze or (
    "analysis_done"
    in st.session_state
):

    if analyze:

        with st.spinner(
            "Analyzing resume..."
        ):

            if uploaded_resume is not None:

                resume_text = (
                    extract_resume_text(
                        uploaded_resume
                    )
                )

                if (
                    job_description
                    .strip() != ""
                ):

                    # Text Similarity
                    text_score = (
                        calculate_match_score(
                            resume_text,
                            job_description
                        )
                    )

                    # Extract Skills
                    resume_skills = (
                        extract_skills(
                            resume_text
                        )
                    )

                    job_skills = (
                        extract_skills(
                            job_description
                        )
                    )

                    # Skill Score
                    skill_score = (
                        calculate_skill_score(
                            resume_skills,
                            job_skills
                        )
                    )

                    # Final Score
                    match_score = round(
                        (
                            0.3 * text_score
                        )
                        +
                        (
                            0.7
                            * skill_score
                        ),
                        2
                    )

                    # Matched Skills
                    matched_skills = list(
                        set(
                            resume_skills
                        )
                        &
                        set(
                            job_skills
                        )
                    )

                    # Missing Skills
                    missing_skills = list(
                        set(
                            job_skills
                        )
                        -
                        set(
                            resume_skills
                        )
                    )

                    # Career
                    suggested_role = (
                        suggest_role(
                            resume_text
                        )
                    )

                    # Feedback
                    feedback = (
                        generate_feedback(
                            resume_text,
                            job_description,
                            match_score,
                            missing_skills,
                            suggested_role
                        )
                    )

                    # Save everything
                    st.session_state[
                        "analysis_done"
                    ] = True

                    st.session_state[
                        "resume_text"
                    ] = resume_text

                    st.session_state[
                        "job_description"
                    ] = (
                        job_description
                    )

                    st.session_state[
                        "match_score"
                    ] = (
                        match_score
                    )

                    st.session_state[
                        "matched_skills"
                    ] = (
                        matched_skills
                    )

                    st.session_state[
                        "missing_skills"
                    ] = (
                        missing_skills
                    )

                    st.session_state[
                        "feedback"
                    ] = feedback

                    st.session_state[
                        "suggested_role"
                    ] = (
                        suggested_role
                    )

                else:

                    st.warning(
                        "Please enter a job description."
                    )

            else:

                st.warning(
                    "Please upload a resume first."
                )

    # Pull stored values
    match_score = (
        st.session_state[
            "match_score"
        ]
    )

    matched_skills = (
        st.session_state[
            "matched_skills"
        ]
    )

    missing_skills = (
        st.session_state[
            "missing_skills"
        ]
    )

    feedback = (
        st.session_state[
            "feedback"
        ]
    )

    suggested_role = (
        st.session_state[
            "suggested_role"
        ]
    )

    # ---------------- DISPLAY ---------------- #

    st.divider()

    st.subheader(
        "📊 Resume Match Percentage"
    )

    st.progress(
        match_score / 100
    )

    st.success(
        f"{match_score}% Match"
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(
            border=True
        ):

            st.subheader(
                "📌 Matched Skills"
            )

            if matched_skills:

                for skill in (
                    matched_skills
                ):

                    st.write(
                        f"✅ {skill.title()}"
                    )

            else:

                st.write(
                    "No matched skills found."
                )

    with col2:

        with st.container(
            border=True
        ):

            st.subheader(
                "⚠️ Missing Skills"
            )

            if missing_skills:

                for skill in (
                    missing_skills
                ):

                    st.write(
                        f"❌ {skill.title()}"
                    )

            else:

                st.write(
                    "No missing skills."
                )

    col3, col4 = st.columns(2)

    with col3:

        with st.container(
            border=True
        ):

            st.subheader(
                "💬 Feedback"
            )

            st.info(
                feedback
            )

    with col4:

        with st.container(
            border=True
        ):

            st.subheader(
                "🎯 Suggested Career Path"
            )

            st.success(
                suggested_role
            )

    # BUTTON BELOW OUTPUTS
    if st.button(
        "✨ Generate Optimized Resume",
        use_container_width=True
    ):

        with st.spinner(
            "Generating optimized resume..."
        ):

            st.session_state[
                "optimized_resume"
            ] = (
                generate_optimized_resume(
                    st.session_state[
                        "resume_text"
                    ],
                    st.session_state[
                        "job_description"
                    ]
                )
            )

    st.divider()

    with st.container(
        border=True
    ):

        st.subheader(
            "✨ Optimized Resume"
        )

        if (
            "optimized_resume"
            in st.session_state
            and
            st.session_state[
                "optimized_resume"
            ]
        ):

            st.markdown(
                st.session_state[
                    "optimized_resume"
                ]
            )

            pdf_file = (
                create_resume_pdf(
                    st.session_state[
                        "optimized_resume"
                    ]
                )
            )

            st.download_button(
                label=(
                    "📥 Download "
                    "Optimized Resume PDF"
                ),
                data=pdf_file,
                file_name=(
                    "optimized_resume.pdf"
                ),
                mime=(
                    "application/pdf"
                )
            )

        else:

            st.info(
                "Click "
                "'✨ Generate "
                "Optimized Resume'"
            )