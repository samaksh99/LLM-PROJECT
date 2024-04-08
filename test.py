from PyPDF2 import PdfReader
from rich.console import Console
console = Console()
import re
class QuestionAnswerText:
    def read_pdf(self,file_path):
        text = ""
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PdfReader(file)
            # Iterate through each page of the PDF
            for page_num in range(len(pdf_reader.pages)):
                # Extract text from the page
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text

    def question_answer_list(self,text):
        pattern = r'\d+\.\s*(.*?)\n(.*?)(?=\d+\.|$)'

        # Extract questions and answers using regex
        matches = re.findall(pattern, text, re.DOTALL)

        # Store questions and answers in a list
        qa_list = ['\n'.join(match) for match in matches]

        # Print the list of questions and answers
        return qa_list


