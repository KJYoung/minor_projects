from django.urls import path
from core.views import (
    general_core,
)

urlpatterns = [
    path('', general_core, name="core"),
]
