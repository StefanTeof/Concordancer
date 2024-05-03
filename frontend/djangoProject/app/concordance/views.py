from django.shortcuts import render
import requests
from .models import Word

# Create your views here.

def left_right_context(keyword, search_results):
    context_results = []

    # Iterate through each entry in the search results
    for result in search_results:
        filename = result['file_name']
        sentence = result['matching_sentence']

        # Find the position of the keyword in the sentence
        idx = sentence.lower().find(keyword.lower())
        
        if idx != -1:
            left_context = sentence[:idx].strip()
            right_context = sentence[idx + len(keyword):].strip()
            keyword_in_context = sentence[idx:idx + len(keyword)].strip()

            context_results.append((filename, left_context, keyword_in_context, right_context))

    return context_results
    

def index(request):
    categories_list = ["Adjective", "Noun", "Verb", "Numeral", "Adposition"]
    search_results = [] 
    context_results= []
    
    # The form submission via GET request will include 'keyword' and 'category'
    if request.method == 'GET' and 'keyword' in request.GET and 'category' in request.GET:
        keyword = request.GET['keyword']
        category = request.GET['category']

        # Proceed with the API call only if both keyword and category are provided
        if keyword and category:
            payload = {
                'keyword': keyword,
                'pos_category': category
            }
            # URL of the FastAPI endpoint
            url = 'http://localhost:8000/search/'  # Change to your actual FastAPI server URL

            # Making a POST request to the FastAPI backend
            response = requests.post(url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                search_results = response.json()['results']
                context_results = left_right_context(keyword=keyword, search_results=search_results)
            else:
                print("Failed to fetch results")
        else:
            # If keyword or category are empty, do not call API and possibly handle user notification
            print("Keyword and category must be provided")


    context = {
        'categories': categories_list,
        'search_results': search_results,
        'context_results': context_results
    }

    return render(request, 'index.html', context)


# attach file
def attachFile(request):
    user = request.user
    categories_list = ["Adjective", "Noun", "Verb", "Numeral", "Adposition"]
    words_list = Word.objects.all()

    context = {'categories': categories_list, 'words': words_list}

    return render(request, 'attachFile.html', context)

