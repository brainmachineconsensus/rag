from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import chatbot
from django.shortcuts import render

@csrf_exempt
def chat(request):
    if request.method == "POST":
        question = request.POST.get("question", "")
        response = chatbot.invoke(input=question)
        answer = response.get("answer")
        return JsonResponse({"response": answer})
    return JsonResponse({"error": "Invalid request method"}, status=400)

def chat_interface(request):
    return render(request, "chat.html")