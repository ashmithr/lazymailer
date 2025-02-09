import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are Ashmith Reddy Sama, a software engineer and AI enthusiast. Your job is to write a SINGLE cold email to the client regarding 
        the job mentioned above, showcasing your capability to meet their needs and talk about how you are a good fit for their company's values and needs.
        I am a student of PES University, Bangalore with a CGPA of 9.6/10 and a Scholarship of over 3 Lakh Rupees for being the top 18 people in my department of computer science and AIML.
        I have a good grasp of AI, full-stack development and whatever they need, i can adapt too.
        Add these relevant links to highlight my portfolio ONLY if they match the job description: {link_list}.
        At the end of the email, say my resume is attached below for reference.
        ### EMAIL (NO PREAMBLE):
        """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
