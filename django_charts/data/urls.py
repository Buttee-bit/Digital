from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='homepage'),
    path('results', views.results, name='analiz_data'),
    path('NLP_results',views.nlp_results, name='nlp_results' )
]

