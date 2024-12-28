from django.shortcuts import render, redirect

from book_assistant.ai_module.db_handler import load_answer
from book_assistant.ai_module.db_handler import answer_question
from book_assistant.ai_module.db_handler import format_answer_django


from .forms import QuestionForm
from book_assistant.ai_module.logger import log_question


def generate_answer(question):
    
    answer = answer_question(question)
    #answer = load_answer()
    return format_answer_django(answer)
  

def index(request):
    if request.method == 'POST':
        if 'clear' in request.POST:
            return redirect('index')
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            log_question(question)
            answer = generate_answer(question)
            # pass the answer to the context
            return render(request, 'phase1/index.html', {'form': form, 'answer': answer})
    else:
        form = QuestionForm()
    
    return render(request, 'phase1/index.html', {'form': form})