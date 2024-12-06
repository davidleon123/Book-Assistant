from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader

import db_handler as db

from .forms import QuestionForm
import logger

def generate_answer(question):
    
    #answer = db.answer_question(question)
    answer = db.load_answer()
    return db.format_answer_django(answer)
  


def index(request):
    if request.method == 'POST':
        if 'clear' in request.POST:
            return redirect('index')
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            logger.log_question(question)
            answer = generate_answer(question)
            # pass the answer to the context
            return render(request, 'phase1/index.html', {'form': form, 'answer': answer})
    else:
        form = QuestionForm()
    
    return render(request, 'phase1/index.html', {'form': form})