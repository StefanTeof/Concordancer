from django.http import HttpResponse
from django.shortcuts import render
from flask import redirect
import requests
from .models import Word

# Create your views here.

def convert_to_dict(data):
    if not data:
        return []

    if isinstance(data[0], list):
        result = [
            {
                'file_name': item[0],
                'left_context': item[1].strip(),
                'keyword': item[2].strip(),
                'right_context': item[3].strip()
            }
            for item in data
        ]
    elif isinstance(data[0], dict):
        result = data
    else:
        result = []

    return result

def upload_to_fastapi(file):
    url = 'http://localhost:8000/upload_file/'  # URL of your FastAPI endpoint
    files = {'file': (file.name, file, file.content_type)}
    return requests.post(url, files=files)
    

def index(request):
    categories_list = ["Adjective", "Noun", "Verb", "Numeral", "Adposition"]

    # Initialize search results, context results, and category from session if available
    search_results = request.session.get('search_results', [])
    context_results = request.session.get('context_results', [])
    selected_category = request.session.get('selected_category', '')  # Initialize selected category from session

    # The form submission via GET request will include 'keyword' and 'category'
    if request.method == 'GET' and 'keyword' in request.GET:
        
        request.session['search_results'] = []
        request.session['context_results'] = []
        
        keyword = request.GET['keyword']
        
        if 'category' in request.GET:
            category = request.GET['category']

            if keyword and category:
                search_payload = {
                    'keyword': keyword,
                    'pos_category': category
                }
                # URL of the FastAPI endpoint
                search_url = 'http://localhost:8000/search/'

                try:
                    search_response = requests.post(search_url, json=search_payload, timeout=20)
                    
                    if search_response.status_code == 200:
                        search_api_results = search_response.json()['results']
                        
                        unique_results = list({(result['file_name'], result['left_context'], result['keyword'], result['right_context']) for result in search_api_results})

                        unique_results_dicts = [
                            {
                                "file_name": file_name,
                                "left_context": left_context,
                                "keyword": keyword,
                                "right_context": right_context
                            }
                            for (file_name, left_context, keyword, right_context) in unique_results
                        ]

                        # print(f"SEARCH API RESULTS ======== {search_api_results}")
                        context_results.extend(unique_results_dicts)
                        request.session['search_results'] = unique_results_dicts
                        request.session['selected_category'] = category
                    else:
                        print("Failed to fetch results from search API")
                except requests.Timeout:
                    print("The request timed out")
                except requests.RequestException as e:
                    print(f"An error occurred: {e}")
            else:
                # If keyword or category are empty, do not call API and possibly handle user notification
                print("Keyword and category must be provided")
        else:
            simple_search_payload = {
                'keyword': keyword
            }
            simple_search_url = 'http://localhost:8000/simple_search/'
            try:
                simple_search_response = requests.post(simple_search_url, json=simple_search_payload, timeout=10)
                
                if simple_search_response.status_code == 200:
                    simple_search_api_results = simple_search_response.json()
                    context_results.extend(simple_search_api_results)
                else:
                    print("Failed to fetch results from simple search API")
            except requests.Timeout:
                print("The request timed out")
            except requests.RequestException as e:
                print(f"An error occurred: {e}")
    
    
    context_results = convert_to_dict(context_results)
    print(f"SEARCH API RESULTS ======== {context_results}")
                        
    
    context = {
        'categories': categories_list,
        'search_results': search_results,
        'context_results': context_results,
        'selected_category': selected_category  # Pass selected category to the context
    }

    return render(request, 'index.html', context)

# attach file
def attachFile(request):
    message = ''
    if request.method == 'POST':
        # Use .get() to safely access the 'file' key
        file = request.FILES.get('file')
        if file:
            response = upload_to_fastapi(file)
            if response.ok:
                message = 'Your file has been uploaded successfully'
            else:
                message = 'Error while uploading file: ' + response.text  # Include the error message from the response
        else:
            message = "No file was uploaded."

    context = {'message': message}
    return render(request, 'attachFile.html', context)
   

