from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader

from .forms import QuestionForm

#def index(request) -> HttpResponse:
#    return render(request, 'phase1/index.html')


def helper():
    return "B"


def home(request):
    
    context = {"a": helper()}
    return render(request, "phase1/home.html", context)


def generate_answer(question):
    # Your logic to generate an answer based on the question
    answer = f"Answer to: {question}"
    return answer + "XYZ"

def index(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            answer = generate_answer(question)
            # You can pass the answer to the context if needed
            return render(request, 'phase1/index.html', {'form': form, 'answer': answer})
    else:
        form = QuestionForm()
    
    return render(request, 'phase1/index.html', {'form': form})