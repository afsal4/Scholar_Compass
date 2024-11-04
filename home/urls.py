from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat_post', views.chat_post, name='chat_post'),
    path('view-pdf/', views.view_pdf_page, name='view_pdf_page'),
    ]


