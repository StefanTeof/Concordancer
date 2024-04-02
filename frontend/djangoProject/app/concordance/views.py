from django.shortcuts import render
from .models import Word


# Create your views here.
def index(request):
    user = request.user
    categories_list = ["Adjective", "Noun", "Verb", "Numeral", "Adposition"]
    words_list = Word.objects.all()

    context = {'categories': categories_list, 'words': words_list}

    return render(request, 'index.html', context)


# attach file
def attachFile(request):
    user = request.user
    categories_list = ["Adjective", "Noun", "Verb", "Numeral", "Adposition"]
    words_list = Word.objects.all()

    context = {'categories': categories_list, 'words': words_list}

    return render(request, 'attachFile.html', context)

