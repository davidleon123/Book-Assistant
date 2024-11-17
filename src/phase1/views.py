from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def helper():
    return "B"


def home(request):
    
    context = {"a": helper()}
    return render(request, "phase1/home.html", context)


