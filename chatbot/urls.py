# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("chat/", views.chat, name="chat"),
    path("", views.chat_interface, name="chat_interface"),
]
